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
modelPath = "/home/luoling/FoundaST/funs/model/models/models.ckpt"
logfile_path = os.path.join(savePath,'processing_STOmics.log')

# Set up logging
logging.basicConfig(filename=logfile_path,
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

# Load RNA Pretrain Model
key = 'cell'
pretrainmodel, pretrainconfig = load_model_frommmf(modelPath, key)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:6')
pretrainmodel.to(device)
logging.info("Loaded RNA pretrain model")

# databases = os.listdir(dataBasePath)  # ['10xGenomics', 'CROST', 'SODB', 'STOmics']
databases = ["STOmics"]
logging.info(f"Databases found: {databases}")

for database in databases:
    logging.info(f"Processing database: {database}")
    databasepath_ = os.path.join(savePath, database)

    # List samples id under database
    samples_ = os.listdir(databasepath_)

    # Iterate on each sample
    for sample_ in samples_:
        try:
            logging.info(f"Processing sample: {sample_}")
            dataPath = os.path.join(databasepath_, sample_)

            sampleName = sample_.split(".")[0]
            SamplesaveDirPath = os.path.join(savePath, database,sampleName)

            if not os.path.exists(SamplesaveDirPath):
                os.makedirs(SamplesaveDirPath)

            # Crop ST into patches
            for d_ in [100, 200]:
                sequence_vector_one_path = os.path.join(SamplesaveDirPath,f"Gene{d_}.npz")
                sequence_vector_one = np.load(sequence_vector_one_path)['arr_0']

                # Save Emb Result
                gene_emb = get_exp_embedding(sequence_vector_one, pretrainmodel, pretrainconfig, device, pre_normalized='T')
                np.savez(os.path.join(SamplesaveDirPath,"GeneEmb" + str(d_)), gene_emb)

            logging.info(f"Processing of sample {sampleName} done")

            # Manually delete variables and empty GPU cache
            del adata, adata_new, expProfile, sequence_vector_one, cropped_images_array_one, coor_dataframe, gene_emb
            torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"Error processing sample {sample_}: {e}")
            torch.cuda.empty_cache()

logging.info("All samples processed")
