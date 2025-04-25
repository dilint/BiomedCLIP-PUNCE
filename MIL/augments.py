import torch
import numpy as np
from sklearn.cluster import KMeans

def augment_patch_features(patch_features, ratio=0.05, K=20, min_class_patches=0):
    """
    对所有 cell 进行聚类和分层 dropout，但返回的结果按 patch 分开。
    
    参数:
        patch_features: tensor, [N_patch, 1536], N_patch为不定长
        ratio: 控制 dropout 的比例。 去除数量: N_cluster_cells // ratio
        K: 聚类的类别数。
        min_class_patches: 每个类别最少保留的 cell 数量。
    
    返回:
        处理后的 cell_features，按 patch 分开，每个 patch 的 cell 数量可能减少。
    """
    kmeans = KMeans(n_init='auto', n_clusters=K, random_state=42)
    # if len(patch_features.shape) == 3:
    patch_features = patch_features.squeeze()
    try:
        cluster_labels = kmeans.fit_predict(patch_features.cpu())  # [N_total_cells,]
    except Exception as e:
        print(f"WSI name: {e}")
    # print(cluster_labels)
    new_patch_features = []
    
    for cluster_id in range(K):
        # 找到当前类别的 cell 索引
        cluster_mask = (cluster_labels == cluster_id)
        cluster_patches = patch_features[cluster_mask]  

        # 计算需要 drop 的 cell 数量
        N_cluster_patches = cluster_patches.shape[0]
        if N_cluster_patches <= min_class_patches:
            N_drop = 0
        else:
            N_drop = int(N_cluster_patches * ratio)  # 每个类别最多 drop N_cluster_cells // ratio 个 cell

        # 随机选择要 drop 的 cell
        if N_drop > 0:
            drop_indices = torch.randperm(N_cluster_patches)[:N_drop]
            keep_mask = torch.ones(N_cluster_patches, dtype=torch.bool)
            keep_mask[drop_indices] = False
            cluster_patches = cluster_patches[keep_mask]  # 保留的 cell
        new_patch_features.append(cluster_patches)
    new_patch_features = torch.cat(new_patch_features, dim=0)
    return new_patch_features