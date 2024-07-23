import os

os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import pickle
import torch
import random
import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, random_split
from STFounda import STFounda  # Assuming this is a custom model defined elsewhere
from dataset import MultiModalDataset
from Loss import InfoNCE  # Assuming this is a custom loss defined elsewhere
import matplotlib.pyplot as plt
from listdata import list_data
import pandas as pd
import torch.nn as nn

# 设置随机种子
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 参数初始化函数
def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

# Function to load DataLoader from a .pkl file
def load_dataloader(file_path):
    with open(file_path, 'rb') as f:
        dataloader = pickle.load(f)
    return dataloader

# 模型和训练参数
num_epochs = 25
batch_size = 16  # 每个小批次的大小
global_batch_size = 256  # 全局批次大小
accumulation_steps = global_batch_size // batch_size  # 梯度累计步数
eps = 1e-5
save_interval = 5  # 每5个epoch保存一次

# 初始化模型
model = STFounda()
model.apply(initialize_weights)
model = model.to('cuda')

# 优化器和损失函数
initial_lr = 2e-5
weight_decay = 1e-5
infoNCE = InfoNCE(temperature=1.0, reduction='mean', negative_mode='paired')
optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=5, factor=0.5
)

def train_one_epoch(model, optimizer, train_loader_paths, epoch, accumulation_steps, device):
    model.train()
    train_m_loss, train_n_loss = 0, 0
    optimizer.zero_grad()
    
    total_batches = 0
    pbar_dataloader = tqdm(train_loader_paths, desc=f"DataLoader Progress Epoch {epoch + 1}")
    
    for loader_path in pbar_dataloader:
        train_loader = load_dataloader(loader_path)
        pbar_batch = tqdm(train_loader, leave=False, desc=f"Batch Progress")
        for step, batch in enumerate(pbar_batch):
            total_batches += 1  # Count total batches
            batch = [tensor.to(device) for tensor in batch]
            st100, stEmb100, gene100, geneEmb100, pos_st, pos_stEmb, pos_gene, pos_geneEmb, neg_st_1, neg_stEmb_1, neg_gene_1, neg_geneEmb_1, neg_st_2, neg_stEmb_2, neg_gene_2, neg_geneEmb_2, neg_st_3, neg_stEmb_3, neg_gene_3, neg_geneEmb_3, neg_st_4, neg_stEmb_4, neg_gene_4, neg_geneEmb_4, neg_st_5, neg_stEmb_5, neg_gene_5, neg_geneEmb_5, neg_st_1_100, neg_stEmb_1_100, neg_gene_1_100, neg_geneEmb_1_100, neg_st_2_100, neg_stEmb_2_100, neg_gene_2_100, neg_geneEmb_2_100, neg_st_3_100, neg_stEmb_3_100, neg_gene_3_100, neg_geneEmb_3_100, neg_st_4_100, neg_stEmb_4_100, neg_gene_4_100, neg_geneEmb_4_100, neg_st_5_100, neg_stEmb_5_100, neg_gene_5_100, neg_geneEmb_5_100 = batch

            st_feature, exp_feature, fusion_feature = model(st100, stEmb100, gene100, geneEmb100)
            _, _, pos_fusion_feature = model(pos_st, pos_stEmb, pos_gene, pos_geneEmb)

            neg_exp_1 = model(neg_st_1_100, neg_stEmb_1_100, neg_gene_1_100, neg_geneEmb_1_100)[1]
            neg_exp_2 = model(neg_st_2_100, neg_stEmb_2_100, neg_gene_2_100, neg_geneEmb_2_100)[1]
            neg_exp_3 = model(neg_st_3_100, neg_stEmb_3_100, neg_gene_3_100, neg_geneEmb_3_100)[1]
            neg_exp_4 = model(neg_st_4_100, neg_stEmb_4_100, neg_gene_4_100, neg_geneEmb_4_100)[1]
            neg_exp_5 = model(neg_st_5_100, neg_stEmb_5_100, neg_gene_5_100, neg_geneEmb_5_100)[1]

            neg_fusion_feature_1 = model(neg_st_1, neg_stEmb_1, neg_gene_1, neg_geneEmb_1)[2]
            neg_fusion_feature_2 = model(neg_st_2, neg_stEmb_2, neg_gene_2, neg_geneEmb_2)[2]
            neg_fusion_feature_3 = model(neg_st_3, neg_stEmb_3, neg_gene_3, neg_geneEmb_3)[2]
            neg_fusion_feature_4 = model(neg_st_4, neg_stEmb_4, neg_gene_4, neg_geneEmb_4)[2]
            neg_fusion_feature_5 = model(neg_st_5, neg_stEmb_5, neg_gene_5, neg_geneEmb_5)[2]

            neg_exp_stack = torch.stack((neg_exp_1, neg_exp_2, neg_exp_3, neg_exp_4, neg_exp_5), dim=1)
            neg_fusion_feature_stack = torch.stack((neg_fusion_feature_1, neg_fusion_feature_2, neg_fusion_feature_3, neg_fusion_feature_4, neg_fusion_feature_5), dim=1)

            m_loss = infoNCE(st_feature, exp_feature, neg_exp_stack)
            n_loss = infoNCE(fusion_feature, pos_fusion_feature, neg_fusion_feature_stack)
            total_loss = m_loss / (m_loss.detach() + eps) + n_loss / (n_loss.detach() + eps)

            total_loss.backward()
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_m_loss += m_loss.item()
            train_n_loss += n_loss.item()
            pbar_batch.set_postfix({"m_loss": m_loss.item(), "n_loss": n_loss.item()})
        
        pbar_batch.close()
        del train_loader
        torch.cuda.empty_cache()

    pbar_dataloader.close()
    return train_m_loss / total_batches, train_n_loss / total_batches, total_batches

def validate(model, val_loader_paths, device):
    model.eval()
    val_m_loss, val_n_loss = 0, 0
    total_batches = 0
    with torch.no_grad():
        pbar_dataloader = tqdm(val_loader_paths, desc="DataLoader Progress")
        for loader_path in pbar_dataloader:
            val_loader = load_dataloader(loader_path)
            pbar_batch = tqdm(val_loader, leave=False, desc=f"Batch Progress")
            for batch in val_loader:
                total_batches += 1  # Count total batches
                batch = [tensor.to(device) for tensor in batch]
                st100, stEmb100, gene100, geneEmb100, pos_st, pos_stEmb, pos_gene, pos_geneEmb, neg_st_1, neg_stEmb_1, neg_gene_1, neg_geneEmb_1, neg_st_2, neg_stEmb_2, neg_gene_2, neg_geneEmb_2, neg_st_3, neg_stEmb_3, neg_gene_3, neg_geneEmb_3, neg_st_4, neg_stEmb_4, neg_gene_4, neg_geneEmb_4, neg_st_5, neg_stEmb_5, neg_gene_5, neg_geneEmb_5, neg_st_1_100, neg_stEmb_1_100, neg_gene_1_100, neg_geneEmb_1_100, neg_st_2_100, neg_stEmb_2_100, neg_gene_2_100, neg_geneEmb_2_100, neg_st_3_100, neg_stEmb_3_100, neg_gene_3_100, neg_geneEmb_3_100, neg_st_4_100, neg_stEmb_4_100, neg_gene_4_100, neg_geneEmb_4_100, neg_st_5_100, neg_stEmb_5_100, neg_gene_5_100, neg_geneEmb_5_100 = batch

                st_feature, exp_feature, fusion_feature = model(st100, stEmb100, gene100, geneEmb100)
                pos_st_feature, pos_exp_feature, pos_fusion_feature = model(pos_st, pos_stEmb, pos_gene, pos_geneEmb)

                neg_exp_1 = model(neg_st_1_100, neg_stEmb_1_100, neg_gene_1_100, neg_geneEmb_1_100)[1]
                neg_exp_2 = model(neg_st_2_100, neg_stEmb_2_100, neg_gene_2_100, neg_geneEmb_2_100)[1]
                neg_exp_3 = model(neg_st_3_100, neg_stEmb_3_100, neg_gene_3_100, neg_geneEmb_3_100)[1]
                neg_exp_4 = model(neg_st_4_100, neg_stEmb_4_100, neg_gene_4_100, neg_geneEmb_4_100)[1]
                neg_exp_5 = model(neg_st_5_100, neg_stEmb_5_100, neg_gene_5_100, neg_geneEmb_5_100)[1]

                neg_fusion_feature_1 = model(neg_st_1, neg_stEmb_1, neg_gene_1, neg_geneEmb_1)[2]
                neg_fusion_feature_2 = model(neg_st_2, neg_stEmb_2, neg_gene_2, neg_geneEmb_2)[2]
                neg_fusion_feature_3 = model(neg_st_3, neg_stEmb_3, neg_gene_3, neg_geneEmb_3)[2]
                neg_fusion_feature_4 = model(neg_st_4, neg_stEmb_4, neg_gene_4, neg_geneEmb_4)[2]
                neg_fusion_feature_5 = model(neg_st_5, neg_stEmb_5, neg_gene_5, neg_geneEmb_5)[2]

                neg_exp_stack = torch.stack((neg_exp_1, neg_exp_2, neg_exp_3, neg_exp_4, neg_exp_5), dim=1)
                neg_fusion_feature_stack = torch.stack((neg_fusion_feature_1, neg_fusion_feature_2, neg_fusion_feature_3, neg_fusion_feature_4, neg_fusion_feature_5), dim=1)

                m_loss = infoNCE(st_feature, exp_feature, neg_exp_stack)
                n_loss = infoNCE(fusion_feature, pos_fusion_feature, neg_fusion_feature_stack)

                val_m_loss += m_loss.item()
                val_n_loss += n_loss.item()
                pbar_batch.set_postfix({"m_loss": m_loss.item(), "n_loss": n_loss.item()})

            pbar_batch.close()
            del val_loader
            torch.cuda.empty_cache()
        pbar_dataloader.close()

    return val_m_loss / total_batches, val_n_loss / total_batches, total_batches

# 数据集路径
train_loader_paths = [os.path.join('/mnt/public/luoling/FoundaST/code/dataloader/train', f) for f in os.listdir('/mnt/public/luoling/FoundaST/code/dataloader/train') if f.endswith('.pkl')]
val_loader_paths = [os.path.join('/mnt/public/luoling/FoundaST/code/dataloader/valid', f) for f in os.listdir('/mnt/public/luoling/FoundaST/code/dataloader/valid') if f.endswith('.pkl')]

device = torch.device('cuda')

# 训练过程
train_m_losses, train_n_losses = [], []
val_m_losses, val_n_losses = [], []
learning_rates = []
best_total_loss = float('inf')


os.makedirs('/mnt/public/luoling/FoundaST/code/experimental1-init-param/model', exist_ok=True)

excel_writer = pd.ExcelWriter('/mnt/public/luoling/FoundaST/code/experimental1-init-param/loss_data.xlsx', engine='openpyxl')

for epoch in trange(num_epochs, desc="Epochs"):
    train_m_loss, train_n_loss, total_train_batches = train_one_epoch(model, optimizer, train_loader_paths, epoch, accumulation_steps, device)
    val_m_loss, val_n_loss, total_val_batches = validate(model, val_loader_paths, device)
    train_m_losses.append(train_m_loss)
    train_n_losses.append(train_n_loss)
    val_m_losses.append(val_m_loss)
    val_n_losses.append(val_n_loss)
    learning_rates.append(optimizer.param_groups[0]['lr'])
    total_loss = train_m_loss + train_n_loss
    val_total_loss = val_m_loss + val_n_loss
    
    print(f'Epoch {epoch + 1}/{num_epochs}, Train m_loss: {train_m_loss}, Train n_loss: {train_n_loss}, Validation m_loss: {val_m_loss}, Validation n_loss: {val_n_loss}')
    lr_scheduler.step(val_total_loss)

    if (epoch + 1) % save_interval == 0:
        torch.save(model.state_dict(), f'/mnt/public/luoling/FoundaST/code/experimental1-init-param/model/m_loss_{train_m_loss:.4f}_n_loss_{train_n_loss:.4f}_epoch_{epoch+1}.pt')

    if val_total_loss < best_total_loss:
        best_total_loss = val_total_loss
        torch.save(model.state_dict(), f'/mnt/public/luoling/FoundaST/code/experimental1-init-param/model/best_model_m_loss_{val_m_loss:.4f}_n_loss_{val_n_loss:.4f}.pt')
    
    epoch_data = pd.DataFrame({
        'Epoch': [epoch + 1],
        'Train m_loss': [train_m_loss],
        'Validation m_loss': [val_m_loss],
        'Train n_loss': [train_n_loss],
        'Validation n_loss': [val_n_loss],
        'Learning Rate': [optimizer.param_groups[0]['lr']]
    })
    epoch_data.to_excel(excel_writer, index=False, header=epoch == 0, startrow=epoch)

excel_writer.save()

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
plt.savefig('/mnt/public/luoling/FoundaST/code/experimental1-init-param/img/loss_curve.png')

# 绘制并保存学习率变化曲线
plt.figure()
plt.plot(range(1, num_epochs + 1), learning_rates, label='Learning Rate')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.legend()
plt.savefig('/mnt/public/luoling/FoundaST/code/experimental1-init-param/img/lr_schedule.png')
