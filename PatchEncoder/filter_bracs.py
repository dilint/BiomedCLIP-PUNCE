import os
import glob
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import h5py
import openslide
import queue # 新增：用于管理多卡 GPU 队列

# ==========================================
# 1. Configuration / Path Setup
# ==========================================

# --- Input Files & Directories ---
SPLIT_DIR = '/home/huangjialong/projects/D2VFM-lgj/MIL_BASELINE/datasets/split/bracs-3'
COORD_H5_DIR = '/data/wsi/BRACS-process/patch-results/patches'
WSI_DIR = '/data/wsi/BRACS-process/wsi-soft-link'

# --- Output Settings ---
TARGET_BASE_DIR = '/data/wsi/BRACS-process/bracs-gigapath-mean'
TARGET_IMG_DIR = os.path.join(TARGET_BASE_DIR, 'image') 
TRAIN_LIST_OUTPUT_DIR = os.path.join(TARGET_BASE_DIR, 'train_lists') 

# --- Pre-trained Model & Parameters ---
MODEL_CKPT_PATH = '/home/huangjialong/projects/mil_baseline/logs/bracs-3/MEAN_MIL/time_2025-11-15-11-16_bracs-3_MEAN_MIL_seed_2024/fold_1/Best_EPOCH_9.pth'

FEATURE_DIM = 1536 
NUM_CLASSES = 3
SLIDE_EXT = '.svs' 
PATCH_SIZE = 224 

# --- Multi-GPU & Processing Settings ---
GPU_IDS = [0, 1, 2, 3]  # 指定使用的 4 张显卡 ID
TOP_K = 50
RESIZE_SHAPE = (224, 224) 
MAX_WORKERS = 40        # CPU 线程数，可放心开到 40，因为 GPU 已经被队列保护起来了

# --- Label Mapping ---
LABEL_MAP = {
    0: 'nilm',  
    1: 'class_1', 
    2: 'class_2'
}

# ==========================================
# 2. Model Definition
# ==========================================
def initialize_weights(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            if m.weight is not None: 
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
            head += [nn.Dropout(0.25)]
            
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
            WSI_attn = features_cls.mean(axis=2).squeeze(0) 
            forward_return['WSI_attn'] = WSI_attn
        return forward_return

# ==========================================
# 3. Data Parsing Logic
# ==========================================
def process_folds_and_generate_lists():
    os.makedirs(TRAIN_LIST_OUTPUT_DIR, exist_ok=True)
    csv_files = glob.glob(os.path.join(SPLIT_DIR, '*fold*.csv'))
    
    unique_wsi_tasks = {} 
    print(f"找到 {len(csv_files)} 个 CSV 划分文件。开始解析...")
    
    for csv_path in csv_files:
        fold_name = os.path.basename(csv_path).replace('.csv', '')
        df = pd.read_csv(csv_path)
        
        train_list_data = [] 
        
        splits = ['train', 'val', 'test']
        for split in splits:
            path_col = f'{split}_slide_path'
            label_col = f'{split}_label'
            
            if path_col in df.columns and label_col in df.columns:
                valid_rows = df[df[path_col].notna() & (df[path_col] != '')]
                
                for _, row in valid_rows.iterrows():
                    feature_path = str(row[path_col]).strip()
                    slide_id = os.path.splitext(os.path.basename(feature_path))[0]
                    
                    label_int = int(float(row[label_col]))
                    label_str = LABEL_MAP.get(label_int, str(label_int))
                    
                    if slide_id not in unique_wsi_tasks:
                        unique_wsi_tasks[slide_id] = {
                            'slide_id': slide_id,
                            'feature_path': feature_path,
                            'label_str': label_str
                        }
                    
                    if split == 'train':
                        train_list_data.append(f"{slide_id},{label_str}")
        
        # 修改点：保存为 .csv 文件格式
        train_output_path = os.path.join(TRAIN_LIST_OUTPUT_DIR, f"{fold_name}_train_list.csv")
        with open(train_output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(train_list_data))
        print(f"✅ 生成 Train 列表 -> {train_output_path} (共 {len(train_list_data)} 条)")
        
    print(f"\n解析完成！跨 fold 汇总去重后，共有 {len(unique_wsi_tasks)} 个独立的 WSI 需要提取 Patch。")
    return list(unique_wsi_tasks.values())

# ==========================================
# 4. Multi-threaded & Multi-GPU Processing
# ==========================================
def process_single_wsi(task, gpu_queue):
    slide_id = task['slide_id']
    feature_path = task['feature_path']
    label_str = task['label_str'] 
    
    coord_h5_path = os.path.join(COORD_H5_DIR, f"{slide_id}.h5")
    wsi_path = os.path.join(WSI_DIR, f"{slide_id}{SLIDE_EXT}")
    wsi_save_dir = os.path.join(TARGET_IMG_DIR, slide_id)
    
    if not os.path.exists(feature_path): return f"[{slide_id}] 跳过: 缺特征文件"
    if not os.path.exists(coord_h5_path): return f"[{slide_id}] 跳过: 缺坐标文件"
    if not os.path.exists(wsi_path): return f"[{slide_id}] 跳过: 缺WSI文件"
    
    # ---------------- 多卡推理核心逻辑 ----------------
    # 1. 从队列中获取一个空闲的 (模型, 设备)
    model, device = gpu_queue.get() 
    
    try:
        # 先加载到 CPU，避免直接塞满显存
        features = torch.load(feature_path, map_location='cpu')
        features = features.unsqueeze(0).to(device) # 只在这一刻进入特定 GPU
        
        with torch.no_grad():
            outputs = model(features, return_WSI_attn=True)
            patch_scores = outputs['WSI_attn'].cpu() # 算完立刻移回 CPU
            
        del features # 清理显存里的特征
        
    except Exception as e:
        return f"[{slide_id}] 模型推理失败: {e}"
    finally:
        # 2. 无论推理成功还是异常，都必须把 GPU 还给队列，供下一个线程使用！
        gpu_queue.put((model, device)) 
    # ------------------------------------------------

    # --- 后续完全在 CPU 上运行，不占用 GPU ---
    try:
        with h5py.File(coord_h5_path, "r") as f:
            coords = f['coords'][:]
    except Exception as e:
        return f"[{slide_id}] 读取坐标失败: {e}"

    if patch_scores.shape[0] != coords.shape[0]:
        return f"[{slide_id}] 跳过: 坐标数量 ({coords.shape[0]}) 与分数 ({patch_scores.shape[0]}) 不匹配"

    actual_k = min(TOP_K, patch_scores.shape[0])
    topk_scores, topk_indices = torch.topk(patch_scores, actual_k)
    topk_indices = topk_indices.cpu().numpy()
    
    os.makedirs(wsi_save_dir, exist_ok=True)
    success_count = 0
    
    try:
        with openslide.open_slide(wsi_path) as wsi:
            for rank, p_idx in enumerate(topk_indices):
                if p_idx >= coords.shape[0]: continue
                
                coord = coords[p_idx]
                y, x = int(coord[0]), int(coord[1])
                
                save_name = f"{slide_id}_{label_str}_top{rank+1}_idx{p_idx}.jpg"
                save_path = os.path.join(wsi_save_dir, save_name)
                
                if os.path.exists(save_path):
                    success_count += 1
                    continue
                
                try:
                    patch = wsi.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE)).convert('RGB')
                    img_resized = patch.resize(RESIZE_SHAPE, Image.Resampling.LANCZOS)
                    img_resized.save(save_path)
                    success_count += 1
                except Exception as e:
                    pass # 静默忽略单张 Patch 的切图错误，继续切下一张
                    
    except Exception as e:
        return f"[{slide_id}] 打开WSI文件失败: {e}"
            
    return f"[{slide_id}] 完成, 保存了 {success_count} 张"


def extract_topk_patches(tasks):
    print("\n初始化 GPU 资源池...")
    gpu_queue = queue.Queue()
    
    # 在每张卡上实例化一个模型，并放入队列
    for gpu_id in GPU_IDS:
        device = torch.device(f"cuda:{gpu_id}")
        model = MEAN_MIL(num_classes=NUM_CLASSES, in_dim=FEATURE_DIM).to(device)
        # 注意：这里可能需要加上 weights_only=True 如果 PyTorch 警告
        model.load_state_dict(torch.load(MODEL_CKPT_PATH, map_location=device))
        model.eval()
        gpu_queue.put((model, device))
        print(f" -> 模型已加载至 {device}")
        
    print(f"\n模型准备就绪，开启 {MAX_WORKERS} 线程池并发处理...")

    # 打开一个日志文件，记录所有结果
    with open("process_debug.log", "w", encoding="utf-8") as log_f:
        log_f.write("=== 开始处理日志 ===\n")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 现在我们将 gpu_queue 传递给每个任务
            futures = {executor.submit(process_single_wsi, task, gpu_queue): task for task in tasks}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="提取进度"):
                try:
                    result_msg = future.result()
                    log_f.write(result_msg + "\n")
                    log_f.flush() 
                    
                    if "跳过" in result_msg or "失败" in result_msg:
                        tqdm.write(result_msg)
                        
                except Exception as exc:
                    task = futures[future]
                    error_msg = f"线程致命异常 {task['slide_id']}: {exc}"
                    log_f.write(error_msg + "\n")
                    tqdm.write(error_msg)

if __name__ == "__main__":
    unique_tasks = process_folds_and_generate_lists()
    
    if len(unique_tasks) > 0:
        extract_topk_patches(unique_tasks)
        print("\n✨ 所有任务处理完成！请查看 process_debug.log 获取详细日志。")
    else:
        print("⚠️ 未发现需要处理的数据。")