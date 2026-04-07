import os
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. 路径与参数配置 (保持不变)
# ==========================================
SOURCE_CSV = '/data/wsi/TCTGC10k-labels/6_labels/TCTGC20k-v15-train.csv'
TARGET_BASE_DIR = '/home1/wsi/gc-filter/filter-images/gc20k-gigapath-mean'
TARGET_CSV = os.path.join(TARGET_BASE_DIR, 'train_label.csv')
TARGET_IMG_DIR = os.path.join(TARGET_BASE_DIR, 'image')

# 特征库路径
FEATURE_DIR = '/data/wsi/TCTGC50k-features/gigapath-coarse/pt' 

# 预训练模型权重路径
MODEL_CKPT_PATH = '/home/huangjialong/projects/mil_baseline/logs/gc20k-2/MEAN_MIL/time_2026-01-19-18-48_gc20k-2_MEAN_MIL{}_k{}_alpha{}_ratio{}/seed_2024/Best_EPOCH_13.pth'

FEATURE_DIM = 1536
NUM_CLASSES = 2
TOP_K = 10
RESIZE_SHAPE = (224, 224)
MAX_WORKERS = 40  # 新增: 线程池数量，建议设为 CPU 核心数的 1~2 倍

# ==========================================
# 2. 模型定义 (保持不变)
# ==========================================
def initialize_weights(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class MEAN_MIL(nn.Module):
    def __init__(self,num_classes=2,dropout=True,act=nn.ReLU() ,in_dim = 512):
        super(MEAN_MIL, self).__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.act = act
        self.in_dim = in_dim
        head = [nn.Linear(self.in_dim,512)]
        head+=[act,]

        if dropout:
            head += [nn.Dropout(self.dropout)]
            
        self.classifier = nn.Linear(512,self.num_classes)
        self.head = nn.Sequential(*head)
        self.apply(initialize_weights)

    def forward(self,x,return_WSI_attn = False, return_WSI_feature = False):
        forward_return = {}
        all_feartures = self.head(x)
        features_cls = self.classifier(all_feartures)
        logits = features_cls.mean(axis=1)
        forward_return['logits'] = logits
        if return_WSI_feature:
            forward_return['WSI_feature'] = all_feartures.mean(axis=1)
        if return_WSI_attn:
            WSI_attn = features_cls.mean(axis=2).transpose(0,1)
            forward_return['WSI_attn'] = WSI_attn
        return forward_return

# ==========================================
# 3. 数据过滤与新 CSV 保存 (保持不变)
# ==========================================
def filter_and_save_csv():
    print(f"正在读取原始 CSV: {SOURCE_CSV}")
    df = pd.read_csv(SOURCE_CSV)
    
    condition = df['wsi_path'].str.contains('volume4|volume5', na=False)
    filtered_df = df[condition].copy()
    
    print(f"原始数据共 {len(df)} 条，筛选出 volume4/5 的数据共 {len(filtered_df)} 条。")
    os.makedirs(TARGET_BASE_DIR, exist_ok=True)
    filtered_df.to_csv(TARGET_CSV, index=False)
    print(f"过滤后的 CSV 已保存至: {TARGET_CSV}")
    
    return filtered_df

# ==========================================
# 4. 多线程核心处理逻辑
# ==========================================
def process_single_wsi(row, model, device):
    """
    单个 WSI 的处理逻辑：被提取出来以便多线程调用
    """
    wsi_name = row['wsi_name']
    wsi_label = row['wsi_label'] 
    patch_folder = row['wsi_path']
    
    # --- 4.1 加载 WSI 特征 ---
    feature_path = os.path.join(FEATURE_DIR, f"{wsi_name}.pt")
    if not os.path.exists(feature_path):
        return f"[{wsi_name}] 跳过: 未找到特征文件"
        
    features = torch.load(feature_path, map_location=device)
    features = features.unsqueeze(0) 
    
    # --- 4.2 模型推理 (单线程内独占推断安全) ---
    with torch.no_grad():
        outputs = model(features, return_WSI_attn=True)
        patch_scores = outputs['WSI_attn'].squeeze() 
    
    # --- 4.3 获取 Top-K 索引 ---
    actual_k = min(TOP_K, patch_scores.shape[0])
    topk_scores, topk_indices = torch.topk(patch_scores, actual_k)
    topk_indices = topk_indices.cpu().numpy()
    
    # --- 4.4 读取图像、Resize 并保存 ---
    if not os.path.exists(patch_folder) or not os.path.isdir(patch_folder):
        return f"[{wsi_name}] 跳过: 未找到原始 Patch 目录"
        
    # 【优化点】: 使用 os.listdir 替代 glob，在大文件夹下速度大幅提升
    all_patch_files = [os.path.join(patch_folder, f) for f in os.listdir(patch_folder) if f.endswith('.jpg')]
    
    try:
        all_patch_files = sorted(
            all_patch_files, 
            key=lambda x: (
                int(os.path.basename(x).split(".")[0].split("_")[0]), 
                int(os.path.basename(x).split(".")[0].split("_")[1])
            )
        )
    except Exception as e:
        return f"[{wsi_name}] 排序失败: {e}"
        
    wsi_save_dir = os.path.join(TARGET_IMG_DIR, wsi_name)
    os.makedirs(wsi_save_dir, exist_ok=True)
    
    success_count = 0
    for rank, p_idx in enumerate(topk_indices):
        if p_idx >= len(all_patch_files):
            continue
            
        img_path = all_patch_files[p_idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
            img_resized = img.resize(RESIZE_SHAPE, Image.Resampling.LANCZOS)
            save_name = f"{wsi_name}_{wsi_label}_top{rank+1}_idx{p_idx}.jpg"
            save_path = os.path.join(wsi_save_dir, save_name)
            img_resized.save(save_path)
            success_count += 1
        except Exception as e:
            print(f"处理图片失败 {img_path}: {e}")
            
    return f"[{wsi_name}] 完成, 保存了 {success_count} 张"

def extract_topk_patches(filtered_df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MEAN_MIL(num_classes=NUM_CLASSES, in_dim=FEATURE_DIM).to(device)
    print(f"正在加载模型权重: {MODEL_CKPT_PATH}")
    model.load_state_dict(torch.load(MODEL_CKPT_PATH, map_location=device))
    model.eval()
    
    print(f"模型准备就绪，开启 {MAX_WORKERS} 线程池开始处理...")

    # 准备行数据
    rows_to_process = [row for _, row in filtered_df.iterrows()]

    # 【优化点】: 使用多线程并发处理每个 WSI
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        futures = {executor.submit(process_single_wsi, row, model, device): row for row in rows_to_process}
        
        # 配合 tqdm 实时更新进度条
        for future in tqdm(as_completed(futures), total=len(futures), desc="并发处理 WSI"):
            try:
                # 获取返回值 (如果想看日志，可以解除下方的注释)
                result_msg = future.result()
                # print(result_msg) 
            except Exception as exc:
                print(f"线程执行出现异常: {exc}")

if __name__ == "__main__":
    df_filtered = filter_and_save_csv()
    extract_topk_patches(df_filtered)
    print("✨ 所有任务处理完成！")