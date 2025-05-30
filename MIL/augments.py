import torch
import cupy as cp
from cuml.cluster import KMeans


def augment_patch_features(patch_features, args):
    """
    Args:
        patch_features: torch.Tensor (on GPU), shape [N, D] (N patches, D-dimensional features)
        K: Number of clusters (int)
        ratio: Fraction of patches to drop per cluster (float, 0-1)
        min_class_patches: Minimum patches to keep in each cluster (int)
    Returns:
        K=5, ratio=0.3, min_class_patches=20
        new_patch_features: torch.Tensor (on GPU), shape [M, D] (M <= N after dropping)
    """
    K = args.kmeans_k
    ratio = args.kmeans_ratio
    min_class_patches = args.kmeans_min
    # 检查输入合法性
    if patch_features.size(0) == 0:
        return patch_features
    patch_features = patch_features.squeeze()
    # 将PyTorch Tensor转换为CuPy数组（KMeans需要）
    patch_features_cp = cp.asarray(patch_features)  # 直接转CuPy，无需CPU

    # GPU加速的K-Means聚类
    kmeans = KMeans(n_clusters=K, random_state=42)
    cluster_labels = kmeans.fit_predict(patch_features_cp)  # 返回CuPy数组
    cluster_labels = torch.as_tensor(cluster_labels, device=patch_features.device)  # 转回PyTorch Tensor

    # 按类别处理
    new_patch_features = []
    for cluster_id in range(K):
        # 获取当前类别的patch索引
        cluster_mask = (cluster_labels == cluster_id)
        cluster_patches = patch_features[cluster_mask]  # 直接索引PyTorch Tensor
        
        # 计算需要保留的patch数量（移除ratio部分）
        N_cluster_patches = cluster_patches.size(0)
        N_keep = max(min_class_patches, int(N_cluster_patches * (1 - ratio)))
        
        # 随机选择保留的patch
        if N_keep < N_cluster_patches:
            keep_indices = torch.randperm(N_cluster_patches, device=patch_features.device)[:N_keep]
            cluster_patches = cluster_patches[keep_indices]
        
        new_patch_features.append(cluster_patches)

    # 合并所有保留的patch
    new_patch_features = torch.cat(new_patch_features, dim=0)
    return new_patch_features


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
        device = patch_features.device
        psize, dims = patch_features.shape
        if self.kmeans_k == 0:
            return self._pad_features(patch_features)

        # GPU加速的K-Means
        # patch_features_cp = cp.asarray(patch_features)
        # # patch_features_cp = patch_features
        # kmeans = KMeans(n_clusters=self.kmeans_k, random_state=42)
        # cluster_labels = kmeans.fit_predict(patch_features_cp)
        
        cluster_labels = torch.as_tensor(cluster_labels, device=device)

        # 按类处理
        new_features = []
        for cluster_id in range(self.kmeans_k):
            cluster_mask = (cluster_labels == cluster_id)
            cluster_patches = patch_features[cluster_mask]
            
            N_keep = max(
                self.kmeans_min, 
                int(cluster_patches.size(0) * (1 - self.kmeans_ratio))
            )
            
            if N_keep < cluster_patches.size(0):
                keep_idx = torch.randperm(cluster_patches.size(0), device=device)[:N_keep]
                cluster_patches = cluster_patches[keep_idx]
            
            new_features.append(cluster_patches)

        new_features = torch.cat(new_features, dim=0)
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
            padded = features[:self.target_pad_size]
        else:
            padded = torch.zeros((self.target_pad_size, dims), device=features.device)
            padded[:N] = features
        return padded