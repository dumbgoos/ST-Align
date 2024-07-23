import os
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
import torch.nn as nn
import pandas as pd

path = '/mnt/public/luoling/FoundaST/train_data/'

path_children = [os.path.join(path, item) for item in os.listdir(path)]


def list_data(path_children=path_children):
    all_path = []
    for path_children_item in path_children:
        temp = [os.path.join(path_children_item, item) for item in os.listdir(path_children_item)]
        all_path.extend(temp)
    return all_path

sodb_img_path = '/mnt/public/luoling/FoundaST/train_data/SODB'
sodb_seq_path = '/mnt/public/luoling/FoundaST/train_data/SODB'

def list_data_sodb(img_path_children=sodb_img_path):
    all_path = []
    dataset_list = os.listdir(img_path_children)
    for item in dataset_list:
        all_path.append(os.path.join(img_path_children, item))
    return all_path

if __name__ == '__main__':
    batch_size = 16
    data_paths = list_data_sodb()
    train_loaders = []
    val_loaders = []
    # 为每个数据集创建DataLoader
    for path in data_paths:
        dataset = MultiModalDataset(path, path)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)


    print(len(train_loaders))
    print(len(val_loaders))

    manager = multiprocessing.Manager()
    shared_train_loaders = manager.list()
    shared_val_loaders = manager.list()

   
    # 多线程函数
    def create_dataloaders(path, train_loaders, val_loaders):
        dataset = MultiModalDataset(path, path)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    # 使用进程池加速DataLoader的创建过程
    with multiprocessing.Pool(processes=30) as pool:
        results = []
        for path in data_paths:
            result = pool.apply_async(create_dataloaders, args=(path, shared_train_loaders, shared_val_loaders))
            results.append(result)

        for result in tqdm(results, total=len(data_paths), desc="Creating DataLoaders"):
            result.wait()
    
    train_loaders = list(shared_train_loaders)
    val_loaders = list(shared_val_loaders)
    print(train_loaders)
    print(len(train_loaders))
    print(len(val_loaders))
