import argparse
import torch
import utils
from torch.backends import cudnn
from dataset2 import MultiModalDataset
import torch.utils.data
from dscan import DSCAN
import torch.nn as nn
from info_nce import InfoNCE
import os
import datetime
import time
import json
from pathlib import Path
import pandas as pd


def get_args_parser():
    parser = argparse.ArgumentParser('DSCAN', add_help=False)

    # data parameters
    parser.add_argument('--spot_img_path', default='/mnt/public/luoling/9-23-BEGIN_FROM_HEAD/data/111_spot_view_img',
                        type=str, help="spot level image root path.")
    parser.add_argument('--spot_gene_path',
                        default='/mnt/public/luoling/9-23-BEGIN_FROM_HEAD/data/111_spot_view_gene', type=str,
                        help="spot level gene root path.")
    parser.add_argument('--global_img_path',
                        default='/mnt/public/luoling/9-23-BEGIN_FROM_HEAD/data/111_global_view_img', type=str,
                        help="global level image path.")
    parser.add_argument('--global_gene_path',
                        default='/mnt/public/luoling/9-23-BEGIN_FROM_HEAD/data/111_global_view_gene', type=str,
                        help='global level gene root path.')
    # Model parameters
    parser.add_argument('--alpha', type=float, default=0.5, help="""alpha of loss.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
            weight decay.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
            weight decay.""")
    
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=2, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    
    parser.add_argument('--batch_size_per_gpu', default=24, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=['adamw', 'sgd'], help="""Type of optimizer.""")

    # Misc
    parser.add_argument('--output_dir', default="/mnt/public/luoling/9-23-BEGIN_FROM_HEAD/code_space/DSCAN/log_10_10_model/", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--seed', default=42, type=int, help='Random seed.')
    parser.add_argument('--device', default='cuda', type=str, help='device.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
            distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--saveckp_freq', default=1, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    

    return parser


def train_dscan(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    cluster_labels = []
    dataset_name = pd.read_csv('/mnt/public/luoling/9-23-BEGIN_FROM_HEAD/data/use_dataset.csv')['DATA_BASE_NAME'].tolist()

    for item in [os.path.join('/mnt/public/luoling/9-23-BEGIN_FROM_HEAD/data/4_init_cluster', item+'.csv') for item in dataset_name]:
        cluster_labels.append(pd.read_csv(item)['cluster_label'].tolist())

    dict_cluster = dict(zip(dataset_name, cluster_labels))

    dataset = MultiModalDataset(args.spot_img_path, args.spot_gene_path, args.global_img_path, args.global_gene_path,
                                dict_cluster)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} data.")

    # ============ building student and teacher networks ... ============
    model = DSCAN()
    # move to device
    model = model.to(args.device)
    # synchronize batch norms (if any)
    if utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=False)

    # ============ preparing loss ... ============
    spot_loss = InfoNCE(negative_mode='paired')
    fusion_loss = InfoNCE(negative_mode='paired')

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(model)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with Dscan
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    print(f"Loss, optimizer and schedulers ready.")

    to_restore = {"epoch": 0}

    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        model=model,
        optimizer=optimizer,
        spot_loss=spot_loss,
        fusion_loss=fusion_loss
    )

    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(model, spot_loss, fusion_loss, args.alpha, data_loader, optimizer, lr_schedule,
                                      wd_schedule, epoch, device='cuda')

        # ============ writing logs ... ============
        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'spot_loss': spot_loss.state_dict(),
            'fusion_loss': fusion_loss.state_dict()
        }

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))

        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}

        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model, loss_spot, loss_fusion, loss_alpha, data_loader,
                    optimizer, lr_schedule, wd_schedule, epoch, device='cuda'):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    # update weight decay and learning rate according to their schedule
    for it, batch in enumerate(metric_logger.log_every(data_loader, 1, header)):
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # to device
        batch = [tensor.to(device) for tensor in batch]
        # *batch
        spot_img, spot_gene, global_img, global_gene, \
            neg_spot_img_1, neg_spot_gene_1, neg_global_img_1, neg_global_gene_1, \
            neg_spot_img_2, neg_spot_gene_2, neg_global_img_2, neg_global_gene_2, \
            neg_spot_img_3, neg_spot_gene_3, neg_global_img_3, neg_global_gene_3, \
            neg_spot_img_4, neg_spot_gene_4, neg_global_img_4, neg_global_gene_4, \
            neg_spot_img_5, neg_spot_gene_5, neg_global_img_5, neg_global_gene_5 = batch

        local_img_feature, local_gene_feature, local_feature, global_feature = model(spot_img, spot_gene, global_img,
                                                                                     global_gene)

        neg_local_img_feature_1, neg_local_gene_feature_1, neg_local_feature_1, neg_global_feature_1 = model(
            neg_spot_img_1, neg_spot_gene_1, neg_global_img_1, neg_global_gene_1)
        neg_local_img_feature_2, neg_local_gene_feature_2, neg_local_feature_2, neg_global_feature_2 = model(
            neg_spot_img_2, neg_spot_gene_2, neg_global_img_2, neg_global_gene_2)
        neg_local_img_feature_3, neg_local_gene_feature_3, neg_local_feature_3, neg_global_feature_3 = model(
            neg_spot_img_3, neg_spot_gene_3, neg_global_img_3, neg_global_gene_3)
        neg_local_img_feature_4, neg_local_gene_feature_4, neg_local_feature_4, neg_global_feature_4 = model(
            neg_spot_img_4, neg_spot_gene_4, neg_global_img_4, neg_global_gene_4)
        neg_local_img_feature_5, neg_local_gene_feature_5, neg_local_feature_5, neg_global_feature_5 = model(
            neg_spot_img_5, neg_spot_gene_5, neg_global_img_5, neg_global_gene_5)

        neg_spot_gene_stack = torch.stack((neg_local_gene_feature_1, neg_local_gene_feature_2, neg_local_gene_feature_3,
                                           neg_local_gene_feature_4, neg_local_gene_feature_5), dim=1)
        neg_global_stack = torch.stack((neg_global_feature_1, neg_global_feature_2, neg_global_feature_3,
                                        neg_global_feature_4, neg_global_feature_5), dim=1)

        # update
        spot_img_gene_loss = loss_spot(local_img_feature, local_gene_feature, neg_spot_gene_stack)
        local_global_loss = loss_fusion(local_feature, global_feature, neg_global_stack)
        loss_all = loss_alpha * spot_img_gene_loss + (1 - loss_alpha) * local_global_loss

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_all.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    parsers = argparse.ArgumentParser('DSCAN', parents=[get_args_parser()])
    args = parsers.parse_args()
    train_dscan(args)
