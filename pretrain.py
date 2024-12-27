import argparse
import os
import json
import time
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data
from torch.backends import cudnn

import pandas as pd

import utils
from dataset import MultiModalDataset
from stalign import STAlign
from info_nce import InfoNCE


def get_args_parser():
    """
    Creates and returns an argument parser for the STAlign training script.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser('STAlign', add_help=False)

    # Data parameters
    parser.add_argument('--spot_img_path', default='',
                        type=str, help="Spot-level image root path.")
    parser.add_argument('--spot_gene_path', default='',
                        type=str, help="Spot-level gene root path.")
    parser.add_argument('--global_img_path', default='',
                        type=str, help="Global-level image path.")
    parser.add_argument('--global_gene_path', default='',
                        type=str, help='Global-level gene root path.')

    # Model parameters
    parser.add_argument('--alpha', type=float, default=0.3,
                        help="Alpha coefficient for the loss function.")
    parser.add_argument('--beta', type=float, default=0.3,
                        help="Beta coefficient for the loss function.")
    parser.add_argument('--weight_decay', type=float, default=0.04,
                        help="Initial value of the weight decay.")
    parser.add_argument('--weight_decay_end', type=float, default=0.4,
                        help="Final value of the weight decay.")

    parser.add_argument("--lr", default=0.0005, type=float,
                        help=("Learning rate at the end of linear warmup "
                              "(highest LR used during training). The learning rate "
                              "is linearly scaled with the batch size, and specified "
                              "here for a reference batch size of 256."))
    parser.add_argument("--warmup_epochs", default=2, type=int,
                        help="Number of epochs for the linear learning-rate warmup.")
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help=("Target LR at the end of optimization. "
                              "We use a cosine LR schedule with linear warmup."))

    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
                        help='Per-GPU batch size: number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of epochs for training.')
    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=['adamw', 'sgd'], help="Type of optimizer.")

    # Miscellaneous
    parser.add_argument('--output_dir', default="", type=str,
                        help='Path to save logs and checkpoints.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed for reproducibility.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Device to use for training (e.g., "cuda" or "cpu").')
    parser.add_argument('--num_workers', default=10, type=int,
                        help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str,
                        help=("URL used to set up distributed training; see "
                              "https://pytorch.org/docs/stable/distributed.html"))
    parser.add_argument('--saveckp_freq', default=1, type=int,
                        help='Frequency (in epochs) to save checkpoints.')
    parser.add_argument("--local_rank", default=0, type=int,
                        help="Please ignore and do not set this argument.")

    return parser


def train_stalign(args):
    """
    Main training loop for the STAlign model.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # Initialize distributed training and set random seeds
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)

    # Print all arguments
    print("\n".join(f"{k}: {v}" for k, v in sorted(dict(vars(args)).items())))

    # Enable CuDNN benchmark for optimized performance
    cudnn.benchmark = True

    # ============ Preparing Data ============
    cluster_labels = []
    dataset_names = pd.read_csv('use_dataset.csv')['DATA_BASE_NAME'].tolist()

    # Load cluster labels for each dataset
    for name in dataset_names:
        cluster_file = os.path.join('init_cluster', f"{name}.csv")
        cluster_data = pd.read_csv(cluster_file)['cluster_label'].tolist()
        cluster_labels.append(cluster_data)

    # Create a dictionary mapping dataset names to their cluster labels
    dict_cluster = dict(zip(dataset_names, cluster_labels))

    # Initialize the dataset
    dataset = MultiModalDataset(
        args.spot_img_path,
        args.spot_gene_path,
        args.global_img_path,
        args.global_gene_path,
        dict_cluster
    )

    # Setup distributed sampler
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)

    # Create data loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} samples.")

    # ============ Building Model ============
    model = STAlign().to(args.device)

    # Convert batch norms to synchronized batch norms if applicable
    if utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Wrap model for distributed training
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu],
        broadcast_buffers=False
    )

    # ============ Preparing Loss Functions ============
    spot_loss = InfoNCE(negative_mode='paired')
    global_loss = InfoNCE(negative_mode='paired')
    fusion_loss = InfoNCE(negative_mode='paired')

    # ============ Preparing Optimizer ============
    params_groups = utils.get_params_groups(model)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params_groups,
            lr=0,
            momentum=0.9
        )  # Learning rate is managed by the scheduler

    # ============ Initializing Schedulers ============
    world_size = utils.get_world_size()
    lr = args.lr * (args.batch_size_per_gpu * world_size) / 256.  # Linear scaling rule

    lr_schedule = utils.cosine_scheduler(
        base_value=lr,
        final_value=args.min_lr,
        epochs=args.epochs,
        niter_per_ep=len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        base_value=args.weight_decay,
        final_value=args.weight_decay_end,
        epochs=args.epochs,
        niter_per_ep=len(data_loader),
    )
    print("Loss functions, optimizer, and schedulers are ready.")

    # ============ Loading Checkpoint (if any) ============
    to_restore = {"epoch": 0}
    checkpoint_path = os.path.join(args.output_dir, "checkpoint.pth")

    utils.restart_from_checkpoint(
        checkpoint_path,
        run_variables=to_restore,
        model=model,
        optimizer=optimizer,
        spot_loss=spot_loss,
        fusion_loss=fusion_loss
    )

    start_epoch = to_restore["epoch"]

    # ============ Training Loop ============
    start_time = time.time()
    print("Starting STAlign training!")

    for epoch in range(start_epoch, args.epochs):
        # Set epoch for distributed sampler
        data_loader.sampler.set_epoch(epoch)

        # Train for one epoch
        train_stats = train_one_epoch(
            model=model,
            loss_spot=spot_loss,
            loss_global=global_loss,
            loss_fusion=fusion_loss,
            loss_alpha=args.alpha,
            loss_beta=args.beta,
            data_loader=data_loader,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            wd_schedule=wd_schedule,
            epoch=epoch,
            device=args.device
        )

        # ============ Saving Checkpoints ============
        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'spot_loss': spot_loss.state_dict(),
            'fusion_loss': fusion_loss.state_dict()
        }

        # Ensure the output directory exists
        os.makedirs(args.output_dir, exist_ok=True)

        # Save the latest checkpoint
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))

        # Save additional checkpoints based on frequency
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            checkpoint_filename = f'checkpoint{epoch:04}.pth'
            utils.save_on_master(save_dict, os.path.join(args.output_dir, checkpoint_filename))

        # ============ Logging ============
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            'epoch': epoch
        }

        if utils.is_main_process():
            log_file = Path(args.output_dir) / "log.txt"
            with log_file.open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    # ============ Training Completion ============
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training completed in {total_time_str}')


def train_one_epoch(model, loss_spot, loss_global, loss_fusion, loss_alpha, loss_beta,
                   data_loader, optimizer, lr_schedule, wd_schedule, epoch, device='cuda'):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The STAlign model.
        loss_spot (InfoNCE): Loss function for spot-level features.
        loss_global (InfoNCE): Loss function for global-level features.
        loss_fusion (InfoNCE): Loss function for fused features.
        loss_alpha (float): Weight for the spot-level loss.
        loss_beta (float): Weight for the global-level loss.
        data_loader (torch.utils.data.DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer.
        lr_schedule (list): Learning rate schedule.
        wd_schedule (list): Weight decay schedule.
        epoch (int): Current epoch number.
        device (str): Device to perform computations on.

    Returns:
        dict: Dictionary of averaged training statistics.
    """
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch}/{args.epochs}]'

    # Iterate over batches
    for it, batch in enumerate(metric_logger.log_every(data_loader, 1, header)):
        # Calculate global training iteration
        it = len(data_loader) * epoch + it

        # Update learning rate and weight decay
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # Only apply weight decay to the first parameter group
                param_group["weight_decay"] = wd_schedule[it]

        # Move all tensors in the batch to the specified device
        batch = [tensor.to(device) for tensor in batch]

        # Unpack the batch
        (spot_img, spot_gene, global_img, global_gene,
         neg_spot_img_1, neg_spot_gene_1, neg_global_img_1, neg_global_gene_1,
         neg_spot_img_2, neg_spot_gene_2, neg_global_img_2, neg_global_gene_2,
         neg_spot_img_3, neg_spot_gene_3, neg_global_img_3, neg_global_gene_3,
         neg_spot_img_4, neg_spot_gene_4, neg_global_img_4, neg_global_gene_4,
         neg_spot_img_5, neg_spot_gene_5, neg_global_img_5, neg_global_gene_5) = batch

        # Forward pass through the model
        local_img_feature, local_gene_feature, global_img_feature, global_gene_feature, \
            local_feature, global_feature = model(spot_img, spot_gene, global_img, global_gene)

        # Forward pass for negative samples
        _, neg_local_gene_feature_1, _, neg_global_gene_feature_1, _, neg_global_feature_1 = model(
            neg_spot_img_1, neg_spot_gene_1, neg_global_img_1, neg_global_gene_1)
        _, neg_local_gene_feature_2, _, neg_global_gene_feature_2, _, neg_global_feature_2 = model(
            neg_spot_img_2, neg_spot_gene_2, neg_global_img_2, neg_global_gene_2)
        _, neg_local_gene_feature_3, _, neg_global_gene_feature_3, _, neg_global_feature_3 = model(
            neg_spot_img_3, neg_spot_gene_3, neg_global_img_3, neg_global_gene_3)
        _, neg_local_gene_feature_4, _, neg_global_gene_feature_4, _, neg_global_feature_4 = model(
            neg_spot_img_4, neg_spot_gene_4, neg_global_img_4, neg_global_gene_4)
        _, neg_local_gene_feature_5, _, neg_global_gene_feature_5, _, neg_global_feature_5 = model(
            neg_spot_img_5, neg_spot_gene_5, neg_global_img_5, neg_global_gene_5)

        # Stack negative samples
        neg_spot_gene_stack = torch.stack((
            neg_local_gene_feature_1,
            neg_local_gene_feature_2,
            neg_local_gene_feature_3,
            neg_local_gene_feature_4,
            neg_local_gene_feature_5
        ), dim=1)

        neg_global_gene_stack = torch.stack((
            neg_global_gene_feature_1,
            neg_global_gene_feature_2,
            neg_global_gene_feature_3,
            neg_global_gene_feature_4,
            neg_global_gene_feature_5
        ), dim=1)

        neg_global_stack = torch.stack((
            neg_global_feature_1,
            neg_global_feature_2,
            neg_global_feature_3,
            neg_global_feature_4,
            neg_global_feature_5
        ), dim=1)

        # Compute losses
        spot_img_gene_loss = loss_spot(local_img_feature, local_gene_feature, neg_spot_gene_stack)
        global_img_gene_loss = loss_global(global_img_feature, global_gene_feature, neg_global_gene_stack)
        local_global_loss = loss_fusion(local_feature, global_feature, neg_global_stack)

        # Total loss with weighted coefficients
        loss_all = (
            loss_alpha * spot_img_gene_loss +
            loss_beta * global_img_gene_loss +
            (1 - loss_alpha - loss_beta) * local_global_loss
        )

        # Backpropagation
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        # Synchronize CUDA operations
        torch.cuda.synchronize()

        # Update metrics
        metric_logger.update(loss=loss_all.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # Gather the statistics from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # Return averaged metrics
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    """
    Entry point for the STAlign training script.
    """
    parser = argparse.ArgumentParser('STAlign', parents=[get_args_parser()])
    args = parser.parse_args()
    train_stalign(args)
