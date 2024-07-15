import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"

import torch
import random
import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, random_split
from STFounda import STFounda
from dataset import MultiModalDataset
from Loss import multimodal_loss, InfoNCE
import matplotlib.pyplot as plt

# 设置随机种子
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 模型和训练参数
num_epochs = 30
batch_size = 16  # 每个小批次的大小
global_batch_size = 258  # 全局批次大小
accumulation_steps = global_batch_size // batch_size  # 梯度累计步数
eps = 1e-5

# 初始化模型
model = STFounda()
model = model.to('cuda')

# 优化器和损失函数
lr = 1e-3
initial_lr = 1e-5
weight_decay = 1e-5
infoNCE = InfoNCE(temperature=1.0, reduction='mean', negative_mode='paired')
optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=5, factor=0.5
)

# 数据集和 DataLoader
dataset = MultiModalDataset('/mnt/raid5_30/luoling/ST/code/STFounda/data/data_6_20_img', '/mnt/raid5_30/luoling/ST/code/STFounda/data/data_6_20')
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# Warmup Scheduler
class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, base_lr, target_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.target_lr = target_lr
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr = self.base_lr + (self.target_lr - self.base_lr) * (self.last_epoch + 1) / self.warmup_epochs
        else:
            lr = self.target_lr
        return [lr for _ in self.optimizer.param_groups]

warmup_scheduler = WarmupScheduler(optimizer, warmup_epochs=5, base_lr=initial_lr, target_lr=lr)

# 创建保存图像的文件夹
if not os.path.exists('img'):
    os.makedirs('img')

# 记录训练过程中的损失和学习率
train_m_losses = []
train_n_losses = []
val_m_losses = []
val_n_losses = []
learning_rates = []

# 训练和验证函数
def train_one_epoch(epoch):
    model.train()
    epoch_m_loss = 0
    epoch_n_loss = 0
    optimizer.zero_grad()

    with tqdm(train_loader, desc="Batches", leave=False) as pbar:
        for i, batch in enumerate(pbar):
            batch_cuda = [item.to('cuda') for item in batch]
            (st100, stEmb100, gene100, geneEmb100,
             pos_st, pos_stEmb, pos_gene, pos_geneEmb,
             neg_st_1, neg_stEmb_1, neg_gene_1, neg_geneEmb_1,
             neg_st_2, neg_stEmb_2, neg_gene_2, neg_geneEmb_2,
             neg_st_3, neg_stEmb_3, neg_gene_3, neg_geneEmb_3,
             neg_st_4, neg_stEmb_4, neg_gene_4, neg_geneEmb_4,
             neg_st_5, neg_stEmb_5, neg_gene_5, neg_geneEmb_5,
             neg_st_1_100, neg_stEmb_1_100, neg_gene_1_100, neg_geneEmb_1_100,
             neg_st_2_100, neg_stEmb_2_100, neg_gene_2_100, neg_geneEmb_2_100,
             neg_st_3_100, neg_stEmb_3_100, neg_gene_3_100, neg_geneEmb_3_100,
             neg_st_4_100, neg_stEmb_4_100, neg_gene_4_100, neg_geneEmb_4_100,
             neg_st_5_100, neg_stEmb_5_100, neg_gene_5_100, neg_geneEmb_5_100,
             ) = batch_cuda

            st_feature, exp_feature, fusion_feature = model(st100, stEmb100, gene100, geneEmb100)
            _, _, pos_fusion_feature = model(pos_st, pos_stEmb, pos_gene, pos_geneEmb)
            _, _, neg_fusion_feature_1 = model(neg_st_1, neg_stEmb_1, neg_gene_1, neg_geneEmb_1)
            _, _, neg_fusion_feature_2 = model(neg_st_2, neg_stEmb_2, neg_gene_2, neg_geneEmb_2)
            _, _, neg_fusion_feature_3 = model(neg_st_3, neg_stEmb_3, neg_gene_3, neg_geneEmb_3)
            _, _, neg_fusion_feature_4 = model(neg_st_4, neg_stEmb_4, neg_gene_4, neg_geneEmb_4)
            _, _, neg_fusion_feature_5 = model(neg_st_5, neg_stEmb_5, neg_gene_5, neg_geneEmb_5)

            
            _, neg_exp_1, _ = model(neg_st_1_100, neg_stEmb_1_100, neg_gene_1_100, neg_geneEmb_1_100)
            _, neg_exp_2, _ = model(neg_st_2_100, neg_stEmb_2_100, neg_gene_2_100, neg_geneEmb_2_100)
            _, neg_exp_3, _ = model(neg_st_3_100, neg_stEmb_3_100, neg_gene_3_100, neg_geneEmb_3_100)
            _, neg_exp_4, _ = model(neg_st_4_100, neg_stEmb_4_100, neg_gene_4_100, neg_geneEmb_4_100)
            _, neg_exp_5, _ = model(neg_st_5_100, neg_stEmb_5_100, neg_gene_5_100, neg_geneEmb_5_100)

            neg_fusion_feature_stack = torch.stack((neg_fusion_feature_1, neg_fusion_feature_2, neg_fusion_feature_3, neg_fusion_feature_4, neg_fusion_feature_5), dim=1)
            neg_exp_stack = torch.stack((neg_exp_1, neg_exp_2, neg_exp_3, neg_exp_4, neg_exp_5), dim=1)

            m_loss = infoNCE(st_feature, exp_feature, neg_exp_stack)
            n_loss = infoNCE(fusion_feature, pos_fusion_feature, neg_fusion_feature_stack)
            total_loss = m_loss / (m_loss.detach() + eps) + n_loss / (n_loss.detach() + eps)

            (total_loss / accumulation_steps).backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_m_loss += m_loss.item()
            epoch_n_loss += n_loss.item()
            pbar.set_postfix({"m_loss": m_loss.item(), "n_loss": n_loss.item(), "total_loss": total_loss.item()})
    
    warmup_scheduler.step()
    lr_scheduler.step((epoch_m_loss + epoch_n_loss) / len(train_loader))
    return epoch_m_loss / len(train_loader), epoch_n_loss / len(train_loader)

def validate():
    model.eval()
    val_m_loss = 0
    val_n_loss = 0

    with torch.no_grad():
        with tqdm(val_loader, desc="Validation Batches", leave=False) as pbar:
            for batch in pbar:
                batch_cuda = [item.to('cuda') for item in batch]
                (st100, stEmb100, gene100, geneEmb100,
                 pos_st, pos_stEmb, pos_gene, pos_geneEmb,
                 neg_st_1, neg_stEmb_1, neg_gene_1, neg_geneEmb_1,
                 neg_st_2, neg_stEmb_2, neg_gene_2, neg_geneEmb_2,
                 neg_st_3, neg_stEmb_3, neg_gene_3, neg_geneEmb_3,
                 neg_st_4, neg_stEmb_4, neg_gene_4, neg_geneEmb_4,
                 neg_st_5, neg_stEmb_5, neg_gene_5, neg_geneEmb_5,
                 neg_st_1_100, neg_stEmb_1_100, neg_gene_1_100, neg_geneEmb_1_100,
                 neg_st_2_100, neg_stEmb_2_100, neg_gene_2_100, neg_geneEmb_2_100,
                 neg_st_3_100, neg_stEmb_3_100, neg_gene_3_100, neg_geneEmb_3_100,
                 neg_st_4_100, neg_stEmb_4_100, neg_gene_4_100, neg_geneEmb_4_100,
                 neg_st_5_100, neg_stEmb_5_100, neg_gene_5_100, neg_geneEmb_5_100,
                 ) = batch_cuda

                st_feature, exp_feature, fusion_feature = model(st100, stEmb100, gene100, geneEmb100)


                _, _, pos_fusion_feature = model(pos_st, pos_stEmb, pos_gene, pos_geneEmb)
                _, _, neg_fusion_feature_1 = model(neg_st_1, neg_stEmb_1, neg_gene_1, neg_geneEmb_1)
                _, _, neg_fusion_feature_2 = model(neg_st_2, neg_stEmb_2, neg_gene_2, neg_geneEmb_2)
                _, _, neg_fusion_feature_3 = model(neg_st_3, neg_stEmb_3, neg_gene_3, neg_geneEmb_3)
                _, _, neg_fusion_feature_4 = model(neg_st_4, neg_stEmb_4, neg_gene_4, neg_geneEmb_4)
                _, _, neg_fusion_feature_5 = model(neg_st_5, neg_stEmb_5, neg_gene_5, neg_geneEmb_5)

                _, neg_exp_1, _ = model(neg_st_1_100, neg_stEmb_1_100, neg_gene_1_100, neg_geneEmb_1_100)
                _, neg_exp_2, _ = model(neg_st_2_100, neg_stEmb_2_100, neg_gene_2_100, neg_geneEmb_2_100)
                _, neg_exp_3, _ = model(neg_st_3_100, neg_stEmb_3_100, neg_gene_3_100, neg_geneEmb_3_100)
                _, neg_exp_4, _ = model(neg_st_4_100, neg_stEmb_4_100, neg_gene_4_100, neg_geneEmb_4_100)
                _, neg_exp_5, _ = model(neg_st_5_100, neg_stEmb_5_100, neg_gene_5_100, neg_geneEmb_5_100)

                neg_fusion_feature_stack = torch.stack((neg_fusion_feature_1, neg_fusion_feature_2, neg_fusion_feature_3, neg_fusion_feature_4, neg_fusion_feature_5), dim=1)
                neg_exp_stack = torch.stack((neg_exp_1, neg_exp_2, neg_exp_3, neg_exp_4, neg_exp_5), dim=1)

                m_loss = infoNCE(st_feature, exp_feature, neg_exp_stack)
                n_loss = infoNCE(fusion_feature, pos_fusion_feature, neg_fusion_feature_stack)
                total_loss = m_loss / (m_loss.detach() + eps) + n_loss / (n_loss.detach() + eps)

                val_m_loss += m_loss.item()
                val_n_loss += n_loss.item()
                pbar.set_postfix({"m_loss": m_loss.item(), "n_loss": n_loss.item(), "total_loss": total_loss.item()})

    return val_m_loss / len(val_loader), val_n_loss / len(val_loader)

# 训练过程
for epoch in trange(num_epochs, desc="Epochs"):
    train_m_loss, train_n_loss = train_one_epoch(epoch)
    val_m_loss, val_n_loss = validate()
    train_m_losses.append(train_m_loss)
    train_n_losses.append(train_n_loss)
    val_m_losses.append(val_m_loss)
    val_n_losses.append(val_n_loss)
    learning_rates.append(optimizer.param_groups[0]['lr'])
    print(f'Epoch {epoch + 1}/{num_epochs}, Train m_loss: {train_m_loss}, Train n_loss: {train_n_loss}, Validation m_loss: {val_m_loss}, Validation n_loss: {val_n_loss}')

# 绘制并保存损失曲线
plt.figure()
plt.plot(range(1, num_epochs + 1), train_m_losses, label='Train m_loss')
plt.plot(range(1, num_epochs + 1), train_n_losses, label='Train n_loss')
plt.plot(range(1, num_epochs + 1), val_m_losses, label='Validation m_loss')
plt.plot(range(1, num_epochs + 1), val_n_losses, label='Validation n_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Validation m_loss and n_loss')
plt.legend()
plt.savefig('img/loss_curve.png')

# 绘制并保存学习率变化曲线
plt.figure()
plt.plot(range(1, num_epochs + 1), learning_rates, label='Learning Rate')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.legend()
plt.savefig('img/lr_schedule.png')
