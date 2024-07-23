import os

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import multiprocessing
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
import dill
import pickle

# 设置随机种子
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# data_paths = list_data()
# batch_size = 16  # 每个小批次的大小
# loader_id = 1
# # 为每个数据集创建DataLoader
# for path in data_paths:
#     dataset = MultiModalDataset(path, path)
#     train_size = int(0.8 * len(dataset))
#     val_size = len(dataset) - train_size
#     train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

#     with open(f'/mnt/public/luoling/FoundaST/code/dataloader/train/train_loader{loader_id}.pkl','wb') as f:
#         dill.dump(train_loader, f)

#     with open(f'/mnt/public/luoling/FoundaST/code/dataloader/valid/valid_loader{loader_id}.pkl','wb') as f:
#         dill.dump(val_loader, f)
#     loader_id += 1

with open('/mnt/public/luoling/FoundaST/code/dataloader/valid/valid_loader200.pkl', 'rb') as f:
    dataloader = pickle.load(f)

data = next(iter(dataloader))
print(len(data[0]))