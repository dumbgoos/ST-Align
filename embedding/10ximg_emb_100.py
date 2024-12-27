import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import timm
import logging

model_file = '/mnt/public/luoling/code_space/UNI_model/'

def uni_100emb(
        data_path,
        model_local_dir,
        save_path,
        device='cuda',
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
):
    data_floder = os.listdir(data_path)
    image100_path = [os.path.join(data_path, _, 'Image100.npz') for _ in data_floder]
    model = timm.create_model(
        "vit_large_patch16_224", 
        img_size=img_size, 
        patch_size=patch_size, 
        init_values=init_values, 
        num_classes=num_classes, 
        dynamic_img_size=dynamic_img_size
        )
    model.load_state_dict(torch.load(os.path.join(model_local_dir, "pytorch_model.bin"), map_location=device), strict=True)
    model.eval()
    for image100_path_item_idx in tqdm(range(len(image100_path))):
        temp_list = []
        image100_orgin = np.load(image100_path[image100_path_item_idx])['arr_0']
        nums = image100_orgin.shape[0]
        img_tensor = torch.tensor(image100_orgin)
        image100_resize = F.interpolate(img_tensor.permute(0, 3, 1, 2), size=(224, 224), mode='bilinear').permute(0, 2, 3, 1)

        with torch.inference_mode():
            for num_item in range(nums):
                reg_img = image100_resize[num_item].unsqueeze(dim=0).permute(0, 3, 2, 1).to(torch.float32)
                feature_emb = model(reg_img).numpy().squeeze(0)
                temp_list.append(feature_emb)
        
        # if not os.path.exists(os.path.join(save_path, data_floder[image100_path_item_idx])):
        #     os.makedirs(os.path.join(save_path, data_floder[image100_path_item_idx]))

        np.savez(os.path.join(save_path, data_floder[image100_path_item_idx], 'ImageEmb100.npz') ,temp_list)

if __name__ == '__main__':
    uni_100emb('/mnt/public/luoling/FoundaST/train_data/10xGenomics/', model_file, '/mnt/public/luoling/FoundaST/train_data/10xGenomics/')
    logging.info('10xGenomics 100 Done!')
