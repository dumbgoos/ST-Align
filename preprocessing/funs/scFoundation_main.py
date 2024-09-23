import random
import os
import numpy as np
import pandas as pd
import argparse
import torch
from tqdm import tqdm
import scipy.sparse
from scipy.sparse import issparse
import scanpy as sc
from .model.load import *

def initiate_random_seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_mean_of_duplicates(df):
    """
    Calculate the mean of duplicate columns in a DataFrame and remove the duplicates.

    Args:
        df (pd.DataFrame): DataFrame with potential duplicate columns.

    Returns:
        pd.DataFrame: DataFrame with mean values for duplicate columns and duplicates removed.
    """
    duplicated_genes = df.columns[df.columns.duplicated()]
    for gene in duplicated_genes:
        duplicate_columns = df.loc[:, df.columns == gene]
        df[gene] = duplicate_columns.mean(axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
    return df

def filter_gene(X_df, gene_list):
    """
    Filter and process genes in a DataFrame based on a given gene list.

    Args:
        X_df (pd.DataFrame): DataFrame containing gene expression data.
        gene_list (list): List of target genes.

    Returns:
        pd.DataFrame: Filtered and processed DataFrame containing only the target genes.
    """
    shared_genes = list(set(gene_list) & set(X_df.columns))
    if len(shared_genes) == 0:
        raise ValueError('Gene name Error!')

    X_df = X_df[shared_genes]
    X_df = calculate_mean_of_duplicates(X_df)
    return X_df

def main_gene_selection(X_df, gene_list):
    """
    Rebuild the input data to select target genes encoding proteins.

    Args:
        X_df (pd.DataFrame): DataFrame with gene expression data.
        gene_list (list): List of target genes.

    Returns:
        tuple: Processed DataFrame, list of zero-padded genes, and DataFrame with mask information.
    """
    to_fill_columns = list(set(gene_list) - set(X_df.columns))
    padding_df = pd.DataFrame(np.zeros((X_df.shape[0], len(to_fill_columns))),
                              columns=to_fill_columns, index=X_df.index)
    X_df = pd.DataFrame(np.concatenate([df.values for df in [X_df, padding_df]], axis=1),
                        index=X_df.index,
                        columns=list(X_df.columns) + list(padding_df.columns))
    X_df = X_df[gene_list]

    var = pd.DataFrame(index=X_df.columns)
    var['mask'] = [1 if i in to_fill_columns else 0 for i in list(var.index)]
    return X_df, to_fill_columns, var    

def transfer_and_process_h5ad(adata,gene_list,input_type='singlecell',pre_normalized='F'):
    # Load data
    gexpr_feature = adata
    idx = gexpr_feature.obs_names.tolist()
    col = gexpr_feature.var.index.tolist()
    if issparse(gexpr_feature.X):
        gexpr_feature = gexpr_feature.X.toarray()
    gexpr_feature = pd.DataFrame(gexpr_feature, index=idx, columns=col)

    # Filter Genes
    gexpr_feature = filter_gene(gexpr_feature, gene_list)
    if gexpr_feature.shape[1] < 19264:
        print('Convert gene feature into 19264')
        gexpr_feature, _, _ = main_gene_selection(gexpr_feature, gene_list)
        assert gexpr_feature.shape[1] >= 19264

    if (pre_normalized == 'F') and (input_type == 'bulk'):
        adata = sc.AnnData(gexpr_feature)
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        gexpr_feature = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)


    return gexpr_feature

def get_exp_embedding(gene_exp_feature, pretrainmodel, pretrainconfig, device, input_type='singlecell', output_type='cell', pre_normalized='F', tgthighres='t4', pool_type='max'):
    """
    Generate gene expression embeddings using a pre-trained model.

    Args:
        gene_exp_feature (dataframe): Expression matrix with m samples and column 19266 genes
        pretrainmodel (str): Path to the pre-trained model.
        pretrainconfig (dict): Configuration of pre-trained model.
        device (str): Device to process computation.
        input_type (str): Type of input data ('bulk' or 'singlecell').
        output_type (str): Type of output embeddings ('cell').
        pre_normalized (str): Whether the data is pre-normalized ('T', 'F', 'A').
        tgthighres (str): Target high resolution setting.
        pool_type (str): Pooling type for embeddings ('all' or 'max').

    Returns:
        gene expression embedding
    """
    initiate_random_seed()
    pretrainmodel.eval()

    # Initialize
    geneexpemb = []

    # Check the input_matrix
    if type(gene_exp_feature) is type(np.array([])):
        gexpr_feature = pd.DataFrame(gene_exp_feature)
    else:
        gexpr_feature = gene_exp_feature

    # Inference
    for i in tqdm(range(gexpr_feature.shape[0])):
        with torch.no_grad():
            if input_type == 'bulk':
                if pre_normalized == 'T':
                    totalcount = gexpr_feature.iloc[i, :].sum()
                elif pre_normalized == 'F':
                    totalcount = np.log10(gexpr_feature.iloc[i, :].sum())
                else:
                    raise ValueError('pre_normalized must be T or F')
                tmpdata = gexpr_feature.iloc[i, :].tolist()
                pretrain_gene_x = torch.tensor(tmpdata + [totalcount, totalcount]).unsqueeze(0).to(device)
                data_gene_ids = torch.arange(19266, device=device).repeat(pretrain_gene_x.shape[0], 1)
            elif input_type == 'singlecell':
                if pre_normalized == 'F':
                    tmpdata = np.log1p(gexpr_feature.iloc[i, :] / (gexpr_feature.iloc[i, :].sum()) * 1e4).tolist()
                elif pre_normalized == 'T':
                    tmpdata = gexpr_feature.iloc[i, :].tolist()
                elif pre_normalized == 'A':
                    tmpdata = gexpr_feature.iloc[i, :-1].tolist()
                else:
                    raise ValueError('pre_normalized must be T, F, or A')

                if pre_normalized == 'A':
                    totalcount = gexpr_feature.iloc[i, -1]
                else:
                    totalcount = gexpr_feature.iloc[i, :].sum()

                if tgthighres[0] == 'f':
                    pretrain_gene_x = torch.tensor(tmpdata + [np.log10(totalcount * float(tgthighres[1:])), np.log10(totalcount)]).unsqueeze(0).to(device)
                elif tgthighres[0] == 'a':
                    pretrain_gene_x = torch.tensor(tmpdata + [np.log10(totalcount) + float(tgthighres[1:]), np.log10(totalcount)]).unsqueeze(0).to(device)
                elif tgthighres[0] == 't':
                    pretrain_gene_x = torch.tensor(tmpdata + [float(tgthighres[1:]), np.log10(totalcount)]).unsqueeze(0).to(device)
                else:
                    raise ValueError('tgthighres must start with f, a, or t')
                data_gene_ids = torch.arange(19266, device=device).repeat(pretrain_gene_x.shape[0], 1)

            value_labels = pretrain_gene_x > 0
            x, x_padding = gatherData(pretrain_gene_x, value_labels, pretrainconfig['pad_token_id'])

            if output_type == 'cell':
                position_gene_ids, _ = gatherData(data_gene_ids, value_labels, pretrainconfig['pad_token_id'])
                x = pretrainmodel.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
                position_emb = pretrainmodel.pos_emb(position_gene_ids)
                x += position_emb
                geneemb = pretrainmodel.encoder(x, x_padding)

                
                if pool_type == 'all':
                    geneemb1 = geneemb[:, -1, :]
                    geneemb2 = geneemb[:, -2, :]
                    geneemb3, _ = torch.max(geneemb[:, :-2, :], dim=1)
                    geneemb4 = torch.mean(geneemb[:, :-2, :], dim=1)
                    geneembmerge = torch.concat([geneemb1, geneemb2, geneemb3, geneemb4], axis=1)

                elif pool_type == 'max':
                    geneembmerge, _ = torch.max(geneemb, dim=1)
                else:
                    raise ValueError('pool_type must be all or max')
                geneexpemb.append(geneembmerge.detach().cpu().numpy())

    geneexpemb = np.squeeze(np.array(geneexpemb))

    return geneexpemb
