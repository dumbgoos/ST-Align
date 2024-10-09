from torch.utils.data import Dataset, DataLoader
import numpy as np
from find_negtive_index import get_positive_and_negatives

class MultiModalDataset(Dataset):
    def __init__(self, spot_img_data_path, spot_gene_data_path, global_img_data_path, global_gene_data_path, cluster_label):
        self.spot_img_data = np.load(spot_img_data_path)['arr_0']
        self.spot_gene_data = np.load(spot_gene_data_path)['arr_0']
        self.global_img_data = np.load(global_img_data_path)['arr_0']
        self.global_gene_data = np.load(global_gene_data_path)['arr_0']
      
        self.length = self.spot_img_data.shape[0]
        self.labels = cluster_label

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        _, negtive_indices = get_positive_and_negatives(idx, self.labels)

        spot_img = self.spot_img_data[idx].astype('float32')
        spot_gene = self.spot_gene_data[idx].astype('float32')
        global_img = self.global_img_data[idx].astype('float32')
        global_gene = self.global_gene_data[idx].astype('float32')

        neg_spot_img_1 = self.spot_img_data[negtive_indices[0]].astype('float32')
        neg_spot_gene_1 = self.spot_gene_data[negtive_indices[0]].astype('float32')
        neg_global_img_1 = self.global_img_data[negtive_indices[0]].astype('float32')
        neg_global_gene_1 = self.global_gene_data[negtive_indices[0]].astype('float32')

        neg_spot_img_2 = self.spot_img_data[negtive_indices[1]].astype('float32')
        neg_spot_gene_2 = self.spot_gene_data[negtive_indices[1]].astype('float32')
        neg_global_img_2 = self.global_img_data[negtive_indices[1]].astype('float32')
        neg_global_gene_2 = self.global_gene_data[negtive_indices[1]].astype('float32')

        neg_spot_img_3 = self.spot_img_data[negtive_indices[2]].astype('float32')
        neg_spot_gene_3 = self.spot_gene_data[negtive_indices[2]].astype('float32')
        neg_global_img_3 = self.global_img_data[negtive_indices[2]].astype('float32')
        neg_global_gene_3 = self.global_gene_data[negtive_indices[2]].astype('float32')

        neg_spot_img_4 = self.spot_img_data[negtive_indices[3]].astype('float32')
        neg_spot_gene_4 = self.spot_gene_data[negtive_indices[3]].astype('float32')
        neg_global_img_4 = self.global_img_data[negtive_indices[3]].astype('float32')
        neg_global_gene_4 = self.global_gene_data[negtive_indices[3]].astype('float32')

        neg_spot_img_5 = self.spot_img_data[negtive_indices[4]].astype('float32')
        neg_spot_gene_5 = self.spot_gene_data[negtive_indices[4]].astype('float32')
        neg_global_img_5 = self.global_img_data[negtive_indices[4]].astype('float32')
        neg_global_gene_5 = self.global_gene_data[negtive_indices[4]].astype('float32')


        return spot_img, spot_gene, global_img, global_gene,\
               neg_spot_img_1, neg_spot_gene_1, neg_global_img_1, neg_global_gene_1,\
               neg_spot_img_2, neg_spot_gene_2, neg_global_img_2, neg_global_gene_2,\
               neg_spot_img_3, neg_spot_gene_3, neg_global_img_3, neg_global_gene_3,\
               neg_spot_img_4, neg_spot_gene_4, neg_global_img_4, neg_global_gene_4,\
               neg_spot_img_5, neg_spot_gene_5, neg_global_img_5, neg_global_gene_5
               



if __name__ == '__main__':
    import pandas as pd
    cluster_label = pd.read_csv('/mnt/public/luoling/9-23-BEGIN_FROM_HEAD/data/4_init_cluster/GSE144239_GSM4565823.csv')['cluster_label'].tolist()

    dataset = MultiModalDataset('/mnt/public/luoling/9-23-BEGIN_FROM_HEAD/data/2_same_size_spot_img/GSE144239_GSM4565823.npz',
                                '/mnt/public/luoling/9-23-BEGIN_FROM_HEAD/data/5_scgpt_embed_gene_spot/GSE144239_GSM4565823_count.npz',
                                '/mnt/public/luoling/9-23-BEGIN_FROM_HEAD/data/3_same_size_global_img/GSE144239_GSM4565823.npz',
                                '/mnt/public/luoling/9-23-BEGIN_FROM_HEAD/data/7_same_size_global_gene/GSE144239_GSM4565823.npz',
                                cluster_label
                                )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=3, drop_last=True)
    for spot_img, spot_gene, global_img, global_gene,\
               pos_spot_img, pos_spot_gene, pos_global_img, pos_global_gene,\
               neg_spot_img_1, neg_spot_gene_1, neg_global_img_1, neg_global_gene_1,\
               neg_spot_img_2, neg_spot_gene_2, neg_global_img_2, neg_global_gene_2,\
               neg_spot_img_3, neg_spot_gene_3, neg_global_img_3, neg_global_gene_3,\
               neg_spot_img_4, neg_spot_gene_4, neg_global_img_4, neg_global_gene_4,\
               neg_spot_img_5, neg_spot_gene_5, neg_global_img_5, neg_global_gene_5 in loader:
        print(spot_img.shape)
        print(spot_gene.shape)
        print(global_img.shape)
        print(global_gene.shape)
