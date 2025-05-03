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