# @author: Ling LUO
# @date: 9.1.2024

import os
import csv
import itertools
import pandas as pd
import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_gmt(file_path):
    """
    Reads a GMT (Gene Matrix Transposed) file and parses it into a dictionary.

    Parameters:
    file_path (str): The path to the GMT file.

    Returns:
    dict: A dictionary where the keys are gene set names and the values are dictionaries containing
          'url_description' and a list of 'genes'.
    """
    gene_sets = {}
    
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        
        for line in reader:
            gene_set_name = line[0]
            url_description = line[1]  
            genes = line[2:]
            
            gene_sets[gene_set_name] = {
                'url_description': url_description,
                'genes': genes
            }
    
    return gene_sets


def filter_array_by_bool_list(array, bool_list, device='cuda'):
    """
    Filters the input array by removing columns corresponding to False in the boolean list.

    Parameters:
    array (np.ndarray): The input 2D array with shape (2518, 19264).
    bool_list (list of bool): Boolean list with length 19264.
    device (str): The device to run the operation on ('cpu' or 'cuda').

    Returns:
    torch.Tensor: The filtered 2D tensor containing only columns where the boolean list is True.
    """
    bool_tensor = torch.tensor(bool_list, device=device)
    array_tensor = torch.tensor(array, device=device)
    filtered_tensor = array_tensor[:, bool_tensor]
    return filtered_tensor


def generate_gene_matrix(pathway_list: list, gene_list: list, device='cuda') -> torch.Tensor:
    """
    Generates a binary mask matrix indicating the presence of genes in pathways.

    Parameters:
    pathway_list (list of list): A list where each element is a list of genes in a pathway.
    gene_list (list of str): A list of all possible genes.
    device (str): The device to run the operation on ('cpu' or 'cuda').

    Returns:
    torch.Tensor: A 2D boolean tensor where each row corresponds to a pathway and each column corresponds to a gene.
                 A True value indicates the gene is part of the pathway.
    """
    mask_matrix = []
    for pathway in pathway_list:
        mask = np.isin(gene_list, pathway)
        mask_matrix.append(mask)

    mask_matrix = np.array(mask_matrix)
    return torch.tensor(mask_matrix, dtype=torch.bool, device=device)


def multiply_matrices(mask_matrix: torch.Tensor, data_matrix: torch.Tensor) -> torch.Tensor:
    """
    Multiplies a mask matrix with a data matrix, element-wise.

    Parameters:
    mask_matrix (torch.Tensor): The binary mask matrix (e.g., gene presence matrix).
    data_matrix (torch.Tensor): The data matrix containing gene expression data.

    Returns:
    torch.Tensor: The resulting tensor after element-wise multiplication.
    """
    assert mask_matrix.shape[1] == data_matrix.shape[1], "The number of columns must match!"
    result = data_matrix.unsqueeze(1) * mask_matrix.unsqueeze(0)
    return result


def get_pathway_array(array_item_path, bool_list, pathway_mask, device='cuda'):
    """
    Filters gene expression data by a boolean list and multiplies it with a pathway mask.

    Parameters:
    array_item_path (str): Path to the .npz file containing the gene expression array.
    bool_list (list of bool): Boolean list used to filter the array.
    pathway_mask (torch.Tensor): The binary mask matrix indicating pathway-gene associations.
    device (str): The device to run the operation on ('cpu' or 'cuda').

    Saves:
    torch.Tensor: The result after filtering and multiplication, saved to a .npz file.
    """
    array_item = np.load(array_item_path)['arr_0']
    filtered_gene_list = filter_array_by_bool_list(array=array_item, bool_list=bool_list, device=device)
    res = multiply_matrices(pathway_mask, filtered_gene_list)
    output_filename = os.path.join(array_item_path[:-12], 'Pathway100.npz')
    np.savez(output_filename, res.cpu().numpy())
    logging.info(f'{output_filename} Done!')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    gmt_file = '/mnt/public/luoling/FoundaST/code/pathway/pathway2.gmt'
    gene_path = '/mnt/public/luoling/FoundaST/code/pathway/OS_scRNA_gene_index.19264.tsv'
    train_data_path = '/mnt/public/luoling/FoundaST/train_data'

    # Get train data path list
    train_data_path_one_level = [os.path.join(train_data_path, item) for item in os.listdir(train_data_path)]
    train_data_path_two_level = []

    for item in train_data_path_one_level:
        cache = [os.path.join(item, item_item) for item_item in os.listdir(item)]
        train_data_path_two_level.append(cache)

    train_data_path_two_level = list(itertools.chain(*train_data_path_two_level))
    gene_100_path = [os.path.join(item, 'Gene100.npz') for item in train_data_path_two_level]

    gene_row = pd.read_csv(gene_path, delimiter='\t')['gene_name'].to_list()

    gene_sets = read_gmt(gmt_file)
    gene_matrix = []

    for gene_set_name, data in gene_sets.items():
        gene_matrix.append(data['genes'])

    merged_gene_matrix = list(itertools.chain(*gene_matrix))
    gene_need = list(set(merged_gene_matrix))
    gene_bool_list = [item in gene_need for item in gene_row]

    pathway_mask = generate_gene_matrix(gene_matrix, gene_need, device=device)
    
    for gene_100_path_item in gene_100_path:
        get_pathway_array(gene_100_path_item, gene_bool_list, pathway_mask, device=device)
