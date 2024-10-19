from torch.utils.data import Dataset, DataLoader
import numpy as np
from find_negtive_index import get_positive_and_negatives
import os


class MultiModalDataset(Dataset):
    def __init__(self, spot_img_data_path, spot_gene_data_path, global_img_data_path, global_gene_data_path,
                 cluster_label):
        self.spot_img_data_path = spot_img_data_path
        self.spot_gene_data_path = spot_gene_data_path
        self.global_img_data_path = global_img_data_path
        self.global_gene_data_path = global_gene_data_path
        self.all_spot_img_path = [os.path.join(spot_img_data_path, item) for item in os.listdir(spot_img_data_path)]
        self.all_spot_gene_path = [os.path.join(spot_gene_data_path, item) for item in os.listdir(spot_gene_data_path)]
        self.all_global_img_path = [os.path.join(global_img_data_path, item) for item in os.listdir(global_img_data_path)]
        self.all_global_gene_path = [os.path.join(global_gene_data_path, item) for item in os.listdir(global_gene_data_path)]

        self.labels = cluster_label

    def __len__(self):
        return len(self.all_spot_img_path)

    def __getitem__(self, idx):
        idx_base_name_root = os.path.basename(self.all_spot_img_path[idx])[:-4]
        idx_base_name = idx_base_name_root.rsplit('_', 1)[0]
        inner_idx = eval(idx_base_name_root.rsplit('_', 1)[1])

        negtive_indices = get_positive_and_negatives(inner_idx, self.labels[idx_base_name])
        

        neg_spot_img_1 = np.load(os.path.join(self.spot_img_data_path, idx_base_name + '_' +
                                              str(negtive_indices[0]) + '.npz'))['arr_0'].astype('float32')
        neg_spot_gene_1 = np.load(os.path.join(self.spot_gene_data_path, idx_base_name + '_' +
                                               str(negtive_indices[0]) + '.npz'))['arr_0'].astype('float32')
        neg_global_img_1 = np.load(os.path.join(self.global_img_data_path, idx_base_name + '_' +
                                                str(negtive_indices[0]) + '.npz'))['arr_0'].astype('float32')
        neg_global_gene_1 = np.load(os.path.join(self.global_gene_data_path, idx_base_name + '_' +
                                                 str(negtive_indices[0]) + '.npz'))['arr_0'].astype('float32')

        neg_spot_img_2 = np.load(os.path.join(self.spot_img_data_path, idx_base_name + '_' +
                                              str(negtive_indices[1]) + '.npz'))['arr_0'].astype('float32')
        neg_spot_gene_2 = np.load(os.path.join(self.spot_gene_data_path, idx_base_name + '_' +
                                               str(negtive_indices[1]) + '.npz'))['arr_0'].astype('float32')
        neg_global_img_2 = np.load(os.path.join(self.global_img_data_path, idx_base_name + '_' +
                                                str(negtive_indices[1]) + '.npz'))['arr_0'].astype('float32')
        neg_global_gene_2 = np.load(os.path.join(self.global_gene_data_path, idx_base_name + '_' +
                                                 str(negtive_indices[1]) + '.npz'))['arr_0'].astype('float32')

        neg_spot_img_3 = np.load(os.path.join(self.spot_img_data_path, idx_base_name + '_' +
                                              str(negtive_indices[2]) + '.npz'))['arr_0'].astype('float32')
        neg_spot_gene_3 = np.load(os.path.join(self.spot_gene_data_path, idx_base_name + '_' +
                                               str(negtive_indices[2]) + '.npz'))['arr_0'].astype('float32')
        neg_global_img_3 = np.load(os.path.join(self.global_img_data_path, idx_base_name + '_' +
                                                str(negtive_indices[2]) + '.npz'))['arr_0'].astype('float32')
        neg_global_gene_3 = np.load(os.path.join(self.global_gene_data_path, idx_base_name + '_' +
                                                 str(negtive_indices[2]) + '.npz'))['arr_0'].astype('float32')

        neg_spot_img_4 = np.load(os.path.join(self.spot_img_data_path, idx_base_name + '_' +
                                              str(negtive_indices[3]) + '.npz'))['arr_0'].astype('float32')
        neg_spot_gene_4 = np.load(os.path.join(self.spot_gene_data_path, idx_base_name + '_' +
                                               str(negtive_indices[3]) + '.npz'))['arr_0'].astype('float32')
        neg_global_img_4 = np.load(os.path.join(self.global_img_data_path, idx_base_name + '_' +
                                                str(negtive_indices[3]) + '.npz'))['arr_0'].astype('float32')
        neg_global_gene_4 = np.load(os.path.join(self.global_gene_data_path, idx_base_name + '_' +
                                                 str(negtive_indices[3]) + '.npz'))['arr_0'].astype('float32')

        neg_spot_img_5 = np.load(os.path.join(self.spot_img_data_path, idx_base_name + '_' +
                                              str(negtive_indices[4]) + '.npz'))['arr_0'].astype('float32')
        neg_spot_gene_5 = np.load(os.path.join(self.spot_gene_data_path, idx_base_name + '_' +
                                               str(negtive_indices[4]) + '.npz'))['arr_0'].astype('float32')
        neg_global_img_5 = np.load(os.path.join(self.global_img_data_path, idx_base_name + '_' +
                                                str(negtive_indices[4]) + '.npz'))['arr_0'].astype('float32')
        neg_global_gene_5 = np.load(os.path.join(self.global_gene_data_path, idx_base_name + '_' +
                                                 str(negtive_indices[4]) + '.npz'))['arr_0'].astype('float32')

        spot_img = np.load(self.all_spot_img_path[idx])['arr_0'].astype('float32')
        spot_gene = np.load(self.all_spot_gene_path[idx])['arr_0'].astype('float32')
        global_img = np.load(self.all_global_img_path[idx])['arr_0'].astype('float32')
        global_gene = np.load(self.all_global_gene_path[idx])['arr_0'].astype('float32')

        return spot_img, spot_gene, global_img, global_gene,\
               neg_spot_img_1, neg_spot_gene_1, neg_global_img_1, neg_global_gene_1,\
               neg_spot_img_2, neg_spot_gene_2, neg_global_img_2, neg_global_gene_2,\
               neg_spot_img_3, neg_spot_gene_3, neg_global_img_3, neg_global_gene_3,\
               neg_spot_img_4, neg_spot_gene_4, neg_global_img_4, neg_global_gene_4,\
               neg_spot_img_5, neg_spot_gene_5, neg_global_img_5, neg_global_gene_5

# if __name__ == '__main__':
#     import pandas as pd

#     cluster_labels = []
#     dataset_name = pd.read_csv('/mnt/public/luoling/9-23-BEGIN_FROM_HEAD/data/use_dataset.csv')['DATA_BASE_NAME'].tolist()


#     for item in [os.path.join('/mnt/public/luoling/9-23-BEGIN_FROM_HEAD/data/4_init_cluster', item+'.csv') for item in dataset_name]:
#         cluster_labels.append(pd.read_csv(item)['cluster_label'].tolist())

#     dict_cluster = dict(zip(dataset_name, cluster_labels))

#     dataset = MultiModalDataset('/mnt/public/luoling/9-23-BEGIN_FROM_HEAD/data/111_spot_view_img',
#                                 '/mnt/public/luoling/9-23-BEGIN_FROM_HEAD/data/111_spot_view_gene',
#                                 '/mnt/public/luoling/9-23-BEGIN_FROM_HEAD/data/111_global_view_img',
#                                 '/mnt/public/luoling/9-23-BEGIN_FROM_HEAD/data/111_global_view_gene',
#                                 dict_cluster
#                                 )
#     loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=3, drop_last=True)
#     for spot_img, spot_gene, global_img, global_gene,\
#                neg_spot_img_1, neg_spot_gene_1, neg_global_img_1, neg_global_gene_1,\
#                neg_spot_img_2, neg_spot_gene_2, neg_global_img_2, neg_global_gene_2,\
#                neg_spot_img_3, neg_spot_gene_3, neg_global_img_3, neg_global_gene_3,\
#                neg_spot_img_4, neg_spot_gene_4, neg_global_img_4, neg_global_gene_4,\
#                neg_spot_img_5, neg_spot_gene_5, neg_global_img_5, neg_global_gene_5 in loader:
#                print(spot_img.shape)