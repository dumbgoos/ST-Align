import os
import scanpy as sc
import pandas as pd
import numpy as np
import torch
import logging

import warnings
warnings.filterwarnings("ignore")

from funs.model.load import *
from funs.scFoundation_main import *
from funs.cropImage import crop_images_by_physical_size

# All path here
savePath = "/mnt/public/luoling/FoundaST/train_data"
if not os.path.exists(savePath):
    os.makedirs(savePath)

dataBasePath = "/mnt/public/luoling/FoundaST/train_data_raw"
geneListPath = "/home/luoling/FoundaST/funs/model/OS_scRNA_gene_index.19264.tsv"
modelPath = "/home/luoling/FoundaST/funs/model/models/models.ckpt"
logfile_path = os.path.join(savePath,'get_crops1.log')

# Set up logging
logging.basicConfig(filename=logfile_path,
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

# Load gene List
gene_list_df = pd.read_csv(geneListPath, header=0, delimiter='\t')
gene_list = list(gene_list_df['gene_name'])
logging.info("Loaded gene list")

# Load RNA Pretrain Model
logging.info("Loaded RNA pretrain model")

# databases = os.listdir(dataBasePath)  # ['10xGenomics', 'CROST', 'SODB', 'STOmics']
databases = ["10xGenomics",'SODB', 'STOmics']
logging.info(f"Databases found: {databases}")

for database in databases:
    logging.info(f"Processing database: {database}")
    databasepath_ = os.path.join(dataBasePath, database)

    # List samples id under database
    samples_ = os.listdir(databasepath_)

    # Iterate on each sample
    for sample_ in samples_:
        try:
            logging.info(f"Processing sample: {sample_}")
            # Load data
            dataPath = os.path.join(databasepath_, sample_)
            adata = sc.read_h5ad(dataPath)

            # gene processing
            expProfile = transfer_and_process_h5ad(adata, gene_list)

            # substitute expression
            adata_new = sc.AnnData(expProfile)

            matched_obs_names = adata.obs_names.intersection(adata_new.obs_names)
            if len(matched_obs_names) == 0:
                raise ValueError('The samples name was not match!')
            else:
                adata_new = adata_new[matched_obs_names, :].copy()
                adata_new.obs = adata.obs.loc[matched_obs_names]
                adata_new.uns = adata.uns
                adata_new.obsm = adata.obsm

            sampleName = sample_.split(".")[0]
            SamplesaveDirPath = os.path.join(savePath, database,sampleName)

            if not os.path.exists(SamplesaveDirPath):
                os.makedirs(SamplesaveDirPath)

            # save .h5ad
            sc.pp.normalize_total(adata_new, target_sum=1e4)
            sc.pp.log1p(adata_new)
            sc.pp.scale(adata_new, max_value=10)

            # adata_new.write_h5ad(os.path.join(SamplesaveDirPath,'anndata.h5ad'))

            # Crop ST into patches
            for d_ in [100, 200]:
                sequence_vector_one, cropped_images_array_one, coor_dataframe = crop_images_by_physical_size(data=adata_new, patch_size_um=d_)

                # Save Crop Result
                np.savez(os.path.join(SamplesaveDirPath, "Gene" + str(d_)), sequence_vector_one)
                np.savez(os.path.join(SamplesaveDirPath, "Image" + str(d_)), cropped_images_array_one)
                coor_dataframe.to_csv(os.path.join(SamplesaveDirPath,"Coor" + str(d_) + ".csv"))

            logging.info(f"Processing of sample {sampleName} done")

            # Manually delete variables and empty GPU cache
            del adata, adata_new, expProfile, sequence_vector_one, cropped_images_array_one, coor_dataframe

        except Exception as e:
            logging.error(f"Error processing sample {sample_}: {e}")

logging.info("All samples processed")
