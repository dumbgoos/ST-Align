import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import csv
import random
from typing import List, Tuple, Union


def cross_entropy(preds, targets, reduction='none'):
    """
    @author Ling LUO
    @description Computes the cross-entropy loss between the predictions and targets.

    Parameters:
    preds (torch.Tensor): Predicted values, typically the output from a neural network.
    targets (torch.Tensor): Ground truth values, should be the same shape as preds.
    reduction (str): Specifies the reduction to apply to the output. Can be 'none' or 'mean'. Defaults to 'none'.

    Returns:
    torch.Tensor: The computed cross-entropy loss. If reduction is 'none', returns the loss for each instance.
                 If reduction is 'mean', returns the mean loss.
    """
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


def multimodal_loss(st_embeddings, exp_embeddings, temperature=1.0):
    """
    @author Ling LUO
    @description Computes the multimodal loss between st and exp embeddings.

    Parameters:
    st_embeddings (torch.Tensor): Embeddings of st, typically the output from an st encoder.
    exp_embeddings (torch.Tensor): Embeddings of exp, typically the output from a exp encoder.
    temperature (float): Temperature parameter to scale the logits.

    Returns:
    torch.Tensor: The mean multimodal loss.
    """
    logits = (exp_embeddings @ st_embeddings.T) / temperature
    st_similarity = st_embeddings @ st_embeddings.T
    exp_similarity = exp_embeddings @ exp_embeddings.T
    targets = F.softmax(
        (st_similarity + exp_similarity) / 2 * temperature, dim=-1
    )
    exp_loss = cross_entropy(logits, targets, reduction='none')
    st_loss = cross_entropy(logits.T, targets.T, reduction='none')
    loss = (st_loss + exp_loss) / 2.0  # shape: (batch_size)
    return loss.mean()


def euclidean_distance(p: Union[List[float], Tuple[float, ...]], q: Union[List[float], Tuple[float, ...]]):
    """
    @author Ling LUO
    @description Calculate the Euclidean distance between two points.

    Parameters:
    p (Union[List[float], Tuple[float, ...]]): Coordinates of point p.
    q (Union[List[float], Tuple[float, ...]]): Coordinates of point q.

    Returns:
    float: Euclidean distance between points p and q.
    """
    if len(p) != len(q):
        raise ValueError("Points p and q must have the same dimension")

    distance = math.sqrt(sum((pi - qi) ** 2 for pi, qi in zip(p, q)))
    return distance


class InfoNCE(nn.Module):
    """
    @author Ling LUO
    @description Computes the InfoNCE loss.

    Parameters:
    temperature (float): Temperature parameter for the InfoNCE loss. Default is 0.1.
    reduction (str): Reduction method for the loss. Options are 'mean' or 'sum'. Default is 'mean'.
    negative_mode (str): Mode for handling negative keys. Options are 'unpaired' or 'paired'. Default is 'unpaired'.

    Methods:
    forward(query, positive_key, negative_keys=None):
        Computes the InfoNCE loss.
        - query (Tensor): The query tensor.
        - positive_key (Tensor): The positive key tensor.
        - negative_keys (Tensor, optional): The negative keys tensor.
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


def info_nce(query, positive_key,
             negative_keys=None, temperature=1.0,
             reduction='mean', negative_mode='unpaired'):
    """
    Computes the InfoNCE loss.

    Parameters:
    query (Tensor): The query tensor (N, D).
    positive_key (Tensor): The positive key tensor (N, D).
    negative_keys (Tensor, optional): The negative keys tensor. If negative_mode is 'unpaired', it should be (M, D). If negative_mode is 'paired', it should be (N, M, D).
    temperature (float): Temperature parameter for scaling logits.
    reduction (str): Reduction method for the loss. Options are 'mean' or 'sum'.
    negative_mode (str): Mode for handling negative keys. Options are 'unpaired' or 'paired'.

    Returns:
    Tensor: The computed InfoNCE loss.

    Raises:
    ValueError: If input tensors do not meet the required dimensions or conditions.
    """

    if query.dim() != 2:
        raise ValueError('query must be 2D tensor')
    if positive_key.dim() != 2:
        raise ValueError('positive_key must be 2D tensor')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError('negative_keys must be 2D tensor for negative_mode=unpaired')
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError('negative_keys must be 3D tensor for negative_mode=paired')

    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError(
                "If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)

    if negative_keys is not None:
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            negative_logits = query @ transpose(negative_keys)
        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        logits = query @ transpose(positive_key)
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def read_csv(file_path: str) -> List[List[float]]:
    """
    @author Ling LUO
    @description Reads data from a CSV file and returns a list of data points.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    List[List[float]]: List of data points, where each sublist contains coordinates of a point.
    """
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            data.append([float(row[1]), float(row[2])])
    return data


def find_positive_negative_samples(file1: str, file2: str, distance_threshold: float, num_negative_samples: int) -> \
        Tuple[List[int], List[List[int]]]:
    """
    @author Ling LUO
    @description Finds the indices of positive and negative samples.

    Parameters:
    file1 (str): Path to the first CSV file.
    file2 (str): Path to the second CSV file.
    distance_threshold (float): Distance threshold for negative samples.
    num_negative_samples (int): Number of negative samples to select for each query sample.

    Returns:
    Tuple[List[int], List[List[int]]]: Indices of positive samples and negative samples.
    """
    data1 = read_csv(file1)
    data2 = read_csv(file2)

    positive_indices = list(range(len(data1)))
    negative_indices = []

    for i, sample1 in enumerate(data1):
        negative_samples = []
        distances = [(j, euclidean_distance(sample1, sample2)) for j, sample2 in enumerate(data2) if j != i]

        # Filter out samples that are within the distance threshold
        filtered_negatives = [idx for idx, dist in distances if dist > distance_threshold]

        if len(filtered_negatives) < num_negative_samples:
            raise ValueError(
                f"Not enough negative samples found for sample index {i} with the given distance threshold.")

        negative_samples = random.sample(filtered_negatives, num_negative_samples)
        negative_indices.append(negative_samples)

    return positive_indices, negative_indices


if __name__ == '__main__':
    # batch_size = 32  # 假设 batch_size 为 32
    # query = torch.randn(batch_size, 1024)  # 查询样本
    # positive_key = torch.randn(batch_size, 1024)  # 正样本
    # negative_keys = torch.randn(batch_size, 5, 1024)  # 每个查询样本有 5 个负样本
    # model = InfoNCE(temperature=0.1, reduction='mean', negative_mode='paired')
    # loss = model(query, positive_key, negative_keys)
    # print("Loss:", loss.item())
    file1 = '/mnt/raid5_30/luoling/ST/code/STFounda/data/data_6_20/Coor100.csv'
    file2 = '/mnt/raid5_30/luoling/ST/code/STFounda/data/data_6_20/Coor200.csv'
    distance_threshold = 10.0  # 示例阈值
    num_negative_samples = 5  # 每个样本选择的负样本数量
    positive_indices, negative_indices = find_positive_negative_samples(file1, file2, distance_threshold,
                                                                        num_negative_samples)
    print("Negative sample indices:", negative_indices)
    print(positive_indices)
