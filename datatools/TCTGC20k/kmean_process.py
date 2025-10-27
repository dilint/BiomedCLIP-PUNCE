import os
import pandas as pd
import torch
from tqdm import tqdm
from cuml.cluster import KMeans
import cupy as cp
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# 关键：在任何CUDA操作（包括import cupy/cuml）之前设置CUDA_VISIBLE_DEVICES
# 这将确保所有 spawned 的子进程都继承这个环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# --- 工作函数 ---
# 将处理单个WSI的逻辑封装成一个函数
# 这个函数将在单独的进程中运行
def process_wsi(wsi_name, input_path, n_clusters):
    """
    为单个WSI加载特征、运行K-Means并返回聚类标签字符串。
    """
    pt_path = os.path.join(input_path, f"{wsi_name}.pt")
    
    try:
        # 1. 加载特征 (I/O)
        features = torch.load(pt_path, weights_only=False)
        
        # 2. 转移到GPU (CPU -> GPU)
        # 每个子进程会创建自己的CUDA上下文
        patch_features_cp = cp.asarray(features)
        
        # 3. 在GPU上运行K-Means (GPU Compute)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels_cp = kmeans.fit_predict(patch_features_cp)
        
        # 4. 将结果传回CPU
        cluster_labels = cp.asnumpy(cluster_labels_cp)
        
        label_str = " ".join(map(str, cluster_labels))
        
        # 返回 (wsi_name, 成功的结果, 无错误)
        return wsi_name, label_str, None
        
    except Exception as e:
        error_msg = f"Error processing {wsi_name}: {str(e)}"
        # 返回 (wsi_name, 无结果, 错误信息)
        return wsi_name, None, error_msg

# --- 主函数 ---
def main():
    # --- 配置 ---
    input_path = '/data/wsi/TCTGC50k-features/gigapath-coarse/pt'
    input_label = '/data/wsi/TCTGC10k-labels/6_labels/TCTGC20k-v15-train.csv'
    labels = pd.read_csv(input_label)
    n_clusters = 5
    output_path = './cluster'
    os.makedirs(output_path, exist_ok=True)
    
    # --- !! 关键调优参数 !! ---
    # 设置进程数。这取决于你的CPU核心数和GPU显存。
    # 每个进程都会在GPU上加载数据并运行KMeans。
    # 如果设置得太高，GPU显存会耗尽 (OOM)。
    # 建议从一个较小的值（例如 4 或 8）开始，并使用 `nvidia-smi` 监控显存。
    N_PROCESSES = 4

    # --- 准备工作 ---
    df = labels[['wsi_name']].copy()
    # 使用字典来快速存储和映射结果
    results_map = {}

    print(f"Starting WSI processing with {N_PROCESSES} processes...")

    # --- 多进程执行 ---
    with ProcessPoolExecutor(max_workers=N_PROCESSES) as executor:
        
        # 提交所有任务
        futures = {
            executor.submit(process_wsi, row['wsi_name'], input_path, n_clusters): row['wsi_name']
            for _, row in df.iterrows()
        }
        
        # 使用tqdm显示进度
        pbar = tqdm(total=len(futures), desc="Processing WSIs")
        for future in as_completed(futures):
            wsi_name, label_str, error_msg = future.result()
            
            if error_msg:
                print(error_msg)
                results_map[wsi_name] = None
            else:
                results_map[wsi_name] = label_str
            
            pbar.update(1)
        pbar.close()

    # --- 收集并保存结果 ---
    print("Mapping results back to DataFrame...")
    # 使用 .map() 一次性将所有结果映射回DataFrame，效率很高
    df['cluster_label'] = df['wsi_name'].map(results_map)

    # 保存最终的CSV
    output_df_path = os.path.join(output_path, f"kmeans_{n_clusters}.csv")
    df.to_csv(output_df_path, index=False)
    print(f"Saved clustered labels to {output_df_path}")


# --- 程序入口 ---
if __name__ == "__main__":
    # 关键：设置多进程启动方式为 'spawn'
    # 这对于 CUDA (cuml, cupy, torch) 在多进程中安全运行至关重要。
    # 'fork'（Linux默认）会引发CUDA初始化错误。
    multiprocessing.set_start_method('spawn', force=True)
    main()