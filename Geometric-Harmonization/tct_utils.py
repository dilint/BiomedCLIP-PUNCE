import torch
import torch.nn as nn

class PatchFeatureAugmenter:
    def __init__(self, 
                 augment_type='kmeans',  # 'kmeans', 'random', 'none'
                 kmeans_k=4, 
                 kmeans_ratio=0.2, 
                 kmeans_min=20,
                 random_drop_ratio=0.3,
                 target_pad_size=1000,):
        """
        Args:
            augment_type: 增强类型 ('kmeans', 'random', 'none')
            kmeans_k: K-Means聚类数 (int)
            kmeans_ratio: 每类丢弃比例 (float 0-1)
            kmeans_min: 每类最少保留数 (int)
            random_drop_ratio: 随机丢弃比例 (float 0-1)
            target_pad_size: 目标填充尺寸 (int)
            device: 设备 ('cuda' or 'cpu')
        """
        self.augment_type = augment_type
        self.kmeans_k = kmeans_k
        self.kmeans_ratio = kmeans_ratio
        self.kmeans_min = kmeans_min
        self.random_drop_ratio = random_drop_ratio
        self.target_pad_size = target_pad_size

    def __call__(self, patch_features, cluster_labels):
        """ 输入: [N, D] 输出: [target_pad_size, D] """
        if self.augment_type == 'none':
            return self._pad_features(patch_features)
        
        elif self.augment_type == 'kmeans':
            return self._kmeans_augment(patch_features, cluster_labels)
        
        elif self.augment_type == 'random':
            return self._random_drop(patch_features)
        
        else:
            raise ValueError(f"Unknown augment_type: {self.augment_type}")

    def _kmeans_augment(self, patch_features, cluster_labels):
        """ K-Means聚类增强 """
        if self.kmeans_k == 0:
            return self._pad_features(patch_features)

        # 向量化替代循环
        masks = [cluster_labels == i for i in range(self.kmeans_k)]
        keep_counts = [
            max(self.kmeans_min, int(m.sum() * (1 - self.kmeans_ratio)))
            for m in masks
        ]

        # 并行化处理每个聚类
        kept_patches = []
        for mask, keep_num in zip(masks, keep_counts):
            cluster_patches = patch_features[mask]
            if keep_num < cluster_patches.size(0):
                idx = torch.randperm(cluster_patches.size(0), device=patch_features.device)[:keep_num]
                cluster_patches = cluster_patches[idx]
            kept_patches.append(cluster_patches)

        new_features = torch.cat(kept_patches, dim=0)
        return self._pad_features(new_features)
    
    def _random_drop(self, patch_features):
        """ 随机丢弃增强 """
        N = patch_features.size(0)
        keep_num = max(1, int(N * (1 - self.random_drop_ratio)))
        keep_idx = torch.randperm(N, device=patch_features.device)[:keep_num]
        return self._pad_features(patch_features[keep_idx])

    def _pad_features(self, features):
        """ 填充到目标尺寸 """
        N, dims = features.shape
        if N >= self.target_pad_size:
            return features[:self.target_pad_size]
        
        # 预分配内存并直接填充
        padded = torch.empty((self.target_pad_size, dims), 
                           device=features.device, 
                           dtype=features.dtype)
        padded[:N] = features
        padded[N:] = 0  # 显式填充0
        return padded
    
    
class TwoViewAugDataset_index(torch.utils.data.Dataset):
    r"""Returns two augmentation of each image and the image label."""

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        feature, label = self.dataset[index]
        cluster_labels = self.dataset.cluster_labels[index]
        device = cluster_labels.device
        return self.transform(feature.to(device), cluster_labels), self.transform(feature.to(device), cluster_labels), label, index

    def __len__(self):
        return len(self.dataset)