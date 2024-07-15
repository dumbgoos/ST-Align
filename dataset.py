from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from Loss import find_positive_negative_samples

class MultiModalDataset(Dataset):
    def __init__(self, img_data_path, seq_data_path, distance_threshold = 100.0, num_negative_samples = 5):
        self.st_100_data = np.load(os.path.join(seq_data_path, 'Image100.npz'))['arr_0']
        self.st_200_data = np.load(os.path.join(seq_data_path, 'Image200.npz'))['arr_0']
        self.stEmb_100_data = np.load(os.path.join(img_data_path, 'ImageEmb100.npz'))['arr_0']
        self.stEmb_200_data = np.load(os.path.join(img_data_path, 'ImageEmb200.npz'))['arr_0']
        self.gene_100_data = np.load(os.path.join(seq_data_path, 'Gene100.npz'))['arr_0']
        self.gene_200_data = np.load(os.path.join(seq_data_path, 'Gene200.npz'))['arr_0']
        self.geneEmb_100_data = np.load(os.path.join(seq_data_path, 'GeneEmb100.npz'))['arr_0']
        self.geneEmb_200_data = np.load(os.path.join(seq_data_path, 'GeneEmb200.npz'))['arr_0']

        self.csv100 = os.path.join(seq_data_path, 'Coor100.csv')
        self.csv200 = os.path.join(seq_data_path, 'Coor200.csv')
        
        self.positive_indices, self.negative_indices = find_positive_negative_samples(self.csv100, self.csv200, distance_threshold,
                                                                        num_negative_samples)

        self.length = len(self.st_100_data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Fetch data from all modalities
        st100 = self.st_100_data[idx].astype('float32')
        stEmb100 = self.stEmb_100_data[idx].astype('float32')
        gene100 = self.gene_100_data[idx].astype('float32')
        geneEmb100 = self.geneEmb_100_data[idx].astype('float32')

        pos = self.positive_indices[idx]
        neg = self.negative_indices[idx]

        neg_st_1_100 = self.st_100_data[neg[0]].astype('float32')
        neg_stEmb_1_100 = self.stEmb_100_data[neg[0]].astype('float32')
        neg_gene_1_100 = self.gene_100_data[neg[0]].astype('float32')
        neg_geneEmb_1_100 = self.geneEmb_100_data[neg[0]].astype('float32')

        neg_st_2_100 = self.st_100_data[neg[1]].astype('float32')
        neg_stEmb_2_100 = self.stEmb_100_data[neg[1]].astype('float32')
        neg_gene_2_100 = self.gene_100_data[neg[1]].astype('float32')
        neg_geneEmb_2_100 = self.geneEmb_100_data[neg[1]].astype('float32')

        neg_st_3_100 = self.st_100_data[neg[2]].astype('float32')
        neg_stEmb_3_100 = self.stEmb_100_data[neg[2]].astype('float32')
        neg_gene_3_100 = self.gene_100_data[neg[2]].astype('float32')
        neg_geneEmb_3_100 = self.geneEmb_100_data[neg[2]].astype('float32')

        neg_st_4_100 = self.st_100_data[neg[3]].astype('float32')
        neg_stEmb_4_100 = self.stEmb_100_data[neg[3]].astype('float32')
        neg_gene_4_100 = self.gene_100_data[neg[3]].astype('float32')
        neg_geneEmb_4_100 = self.geneEmb_100_data[neg[3]].astype('float32')

        neg_st_5_100 = self.st_100_data[neg[4]].astype('float32')
        neg_stEmb_5_100 = self.stEmb_100_data[neg[4]].astype('float32')
        neg_gene_5_100 = self.gene_100_data[neg[4]].astype('float32')
        neg_geneEmb_5_100 = self.geneEmb_100_data[neg[4]].astype('float32')

        

        pos_st = self.st_200_data[pos].astype('float32')
        pos_stEmb = self.stEmb_200_data[pos].astype('float32')
        pos_gene = self.gene_200_data[pos].astype('float32')
        pos_geneEmb = self.geneEmb_200_data[pos].astype('float32')

        neg_st_1 = self.st_200_data[neg[0]].astype('float32')
        neg_stEmb_1 = self.stEmb_200_data[neg[0]].astype('float32')
        neg_gene_1 = self.gene_200_data[neg[0]].astype('float32')
        neg_geneEmb_1 = self.geneEmb_200_data[neg[0]].astype('float32')

        neg_st_2 = self.st_200_data[neg[1]].astype('float32')
        neg_stEmb_2 = self.stEmb_200_data[neg[1]].astype('float32')
        neg_gene_2 = self.gene_200_data[neg[1]].astype('float32')
        neg_geneEmb_2 = self.geneEmb_200_data[neg[1]].astype('float32')

        neg_st_3 = self.st_200_data[neg[2]].astype('float32')
        neg_stEmb_3 = self.stEmb_200_data[neg[2]].astype('float32')
        neg_gene_3 = self.gene_200_data[neg[2]].astype('float32')
        neg_geneEmb_3 = self.geneEmb_200_data[neg[2]].astype('float32')

        neg_st_4 = self.st_200_data[neg[3]].astype('float32')
        neg_stEmb_4 = self.stEmb_200_data[neg[3]].astype('float32')
        neg_gene_4 = self.gene_200_data[neg[3]].astype('float32')
        neg_geneEmb_4 = self.geneEmb_200_data[neg[3]].astype('float32')

        neg_st_5 = self.st_200_data[neg[4]].astype('float32')
        neg_stEmb_5 = self.stEmb_200_data[neg[4]].astype('float32')
        neg_gene_5 = self.gene_200_data[neg[4]].astype('float32')
        neg_geneEmb_5 = self.geneEmb_200_data[neg[4]].astype('float32')

        return (st100, stEmb100, gene100, geneEmb100,
                
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
                )
