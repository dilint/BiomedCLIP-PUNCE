import torch
import numpy as np
import os
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path

class TCTWSIDataset(Dataset):

    def __init__(self, dataroot, label_file, num_classes, transforms=None):
        
        super(TCTWSIDataset, self).__init__()

        assert os.path.exists(dataroot), f'Error, {dataroot} is not exist!'
        assert os.path.exists(label_file), f'Error, {label_file} is not exist!'

        self.dataroot = dataroot
        self.label_file = label_file
        self._transforms = transforms
        self.wsi_data = pd.read_csv(label_file)
        self.num_classes = num_classes
    def __len__(self):

        return len(self.wsi_data)
    
    def __getitem__(self, idx):
        
        wsi_name = self.wsi_data.iloc[idx]['wsi_name']
        assert '.pt' in wsi_name, f'Error, wrong format for {wsi_name}!'

        wsi_label = torch.tensor(self.wsi_data.iloc[idx]['wsi_label'], dtype=torch.float32)  
        wsi_label = F.one_hot(wsi_label, num_classes=self.num_classes)

        wsi = torch.load(os.path.join(self.dataroot, wsi_name if '.pt' in wsi_name else f'{wsi_name}.pt'))

        if self._transforms is not None:
            wsi = self._transforms(wsi)

        return wsi, wsi_label


def collate_fn_wsi(batch):
    wsis = [item[0] for item in batch] #each wsi is a list, contain N patches, each patch is a ndarray, with shape [M, 256]
    max_N = min(max([len(wsi) for wsi in wsis]), 800)
    max_M = max([max([patch.shape[0] for patch in wsi]) for wsi in wsis])

    padded_patches_lists = []
    masks_lists = []
    for wsi in wsis:
        if len(wsi) > max_N:
            wsi = wsi[:max_N] #大于max_N的进行截断
        padded_patches = []
        masks = []
        for patch in wsi:
            # 计算需要 padding 的数量
            padding_size = max_M - patch.shape[0]
            # 对 patch 进行 padding
            padded_patch = F.pad(torch.tensor(patch, dtype=torch.float32), (0, padding_size), mode='constant', value=0)
            padded_patches.append(padded_patch)
            # 创建 mask，真实数据部分为 1，padding 部分为 0
            mask = torch.ones(patch.shape[0], dtype=torch.float32)
            if padding_size > 0:
                mask = torch.cat([mask, torch.zeros(padding_size, dtype=torch.float32)], dim=0)
            masks.append(mask)
        # 将 padded patches 和 masks 转换为张量
        padded_patches_lists.append(torch.stack(padded_patches))
        masks_lists.append(torch.stack(masks))
        # padded_patches = [torch.nn.functional.pad(torch.tensor(patch, dtype=torch.float32), (0, 0, 0, max_M - patch.shape[0]), "constant", 0) for patch in wsi]
        # padded_patches_lists.append(padded_patches)

    padded_patches_tensor = pad_sequence(padded_patches_lists, batch_first=True, padding_value=0) # shape [B, max_N, max_M, 256]
    masks_tensor = pad_sequence(masks_lists, batch_first=True, padding_value=0) # shape [B, max_N, max_M]
    
    wsi_labels = torch.stack([item[1] for item in batch])

    return padded_patches_tensor, masks_tensor, wsi_labels

def build_dataset(image_set, args):
    root = Path(args.root)
    assert root.exists(), f'provided path {root} does not exist'
    PATHS = {
        'train': [root / 'all_data_train.npz', ''],
        'val': [root / 'all_data_test.npz', '']
    }

    data_path = PATHS[image_set]
    dataset = TCTWSIDataset(data_path[0], data_path[1])

    return dataset