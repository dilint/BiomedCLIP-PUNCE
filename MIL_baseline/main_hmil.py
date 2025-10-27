import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, default_collate
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import pandas as pd
import argparse
import os
import random
from contextlib import suppress

# 匯入自定義模組
from dataloader import C16DatasetSharedMemory, ClassBalancedDataset
from models import MIL
from augments import PatchFeatureAugmenter
from loss import BuildClsLoss
from utils import (
    print_and_log, 
    seed_torch, 
    evaluation_cancer_sigmoid,
    evaluation_cancer_sigmoid_cascade,
    evaluation_cancer_sigmoid_cascade_binary,
    evaluation_cancer_softmax,
    save_metrics_to_excel, 
    save_logits
)
from timm.utils import AverageMeter, dispatch_clip_grad
from trainer.hmil_trainer import hmil_training_loop, hmil_validation_loop



ID_TO_LABEL = {
    0: 'nilm', 1: 'ascus', 2: 'asch', 3: 'lsil', 
    4: 'hsil', 5: 'agc', 6: 't', 7: 'm', 8: 'bv'
}
LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}

# --- 分散式訓練設定 ---

def setup_ddp(rank, world_size):
    """初始化分散式訓練環境 (DDP)"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    """清理分散式訓練環境"""
    dist.destroy_process_group()

def broadcast_state_dict(state_dict, device, rank):
    """從主進程 (rank 0) 廣播狀態字典到所有其他進程"""
    if rank == 0:
        objects = [state_dict]
        dist.broadcast_object_list(objects, src=0)
        return state_dict
    else:
        objects = [None]
        dist.broadcast_object_list(objects, src=0)
        return objects[0]

# --- 資料準備 ---

import pandas as pd
import numpy as np
from utils import print_and_log # 假設 print_and_log 從 utils 匯入
import os 
from typing import Any # 為了 gen_mapping_dict

ID_TO_LABEL = {
    0: 'nilm', 1: 'ascus', 2: 'asch', 3: 'lsil', 
    4: 'hsil', 5: 'agc', 6: 't', 7: 'm', 8: 'bv'
}
LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}

def custom_collate_fn(batch):
    cell_images, patient_labels, third_elements = zip(*batch)
    return list(cell_images), list(patient_labels), list(third_elements)

def prepare_metadata(args):
    """
    從 CSV 讀取元數據。
    使用硬編碼規則 (label > 0) 創建 [N, 2] 分層標籤。
    同時動態生成 args.fine_to_coarse_map 和 args.n_class 供後續使用。
    """
    if 'gc' not in args.datasets.lower():
        raise ValueError(f"不支援的資料集: {args.datasets}")

    df_train = pd.read_csv(args.train_label_path)
    
    if args.train_cluster_path:
        df_cluster = pd.read_csv(args.train_cluster_path)
        df_train = df_train.merge(
            df_cluster[['wsi_name', 'cluster_label']], 
            on='wsi_name', 
            how='left'
        )
        train_cluster_labels = df_train['cluster_label'].apply(
            lambda x: [int(i) for i in x.split()]
        ).values
    else:
        train_cluster_labels = None
    
    train_wsi_names = df_train['wsi_name'].values
    train_wsi_labels_fine = df_train['wsi_label'].map(LABEL_TO_ID).values
    
    df_test = pd.read_csv(args.test_label_path)
    test_wsi_names = df_test['wsi_name'].values
    test_wsi_labels_fine = df_test['wsi_label'].map(LABEL_TO_ID).values
    train_wsi_labels_coarse = (train_wsi_labels_fine > 0).astype(int)
    test_wsi_labels_coarse = (test_wsi_labels_fine > 0).astype(int)
    train_wsi_labels = np.column_stack((train_wsi_labels_coarse, train_wsi_labels_fine))
    test_wsi_labels = np.column_stack((test_wsi_labels_coarse, test_wsi_labels_fine))
    
    print_and_log("Info: Labels converted to hierarchical [N, 2] format (Coarse, Fine).", args.log_file, args.no_log)
    return train_wsi_names, train_wsi_labels, train_cluster_labels, test_wsi_names, test_wsi_labels
# --- 核心訓練與評估流程 ---

def run_training_process(rank, world_size, args, 
                         train_wsi_names, train_wsi_labels, train_cluster_labels, train_data_in_shared_memory,
                         test_wsi_names, test_wsi_labels, test_data_in_shared_memory):
    """
    單個 GPU 進程執行的主訓練函數。
    """
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    if world_size > 1:
        setup_ddp(rank, world_size)
    
    seed_torch(args.seed + rank)
    
    args.rank = rank
    
    if rank != 0:
        args.no_log = True

    print_and_log(f'Process {rank}: Dataset: {args.datasets}', args.log_file, args.no_log)

    # ---> 初始化設定
    collate_fn = custom_collate_fn
    amp_autocast = torch.cuda.amp.autocast if args.amp else suppress
    
    train_cluster_labels_tensor = [torch.tensor(labels) for labels in train_cluster_labels] if train_cluster_labels is not None else None

    # ---> 資料載入 (DataLoader)
    # 設定資料增強
    train_transform = test_transform = None
    if args.patch_drop:
        train_transform = PatchFeatureAugmenter(kmeans_k=args.kmeans_k, kmeans_ratio=args.kmeans_ratio, kmeans_min=args.kmeans_min)
    elif args.patch_pad:
        train_transform = PatchFeatureAugmenter(augment_type='none')
    if args.patch_pad:
        test_transform = PatchFeatureAugmenter(augment_type='none')
    
    # 從共享記憶體建立 Dataset
    train_set = C16DatasetSharedMemory(
        train_wsi_names, train_wsi_labels, root=args.dataset_root, 
        cluster_labels=train_cluster_labels_tensor, transform=train_transform,
        shared_data=train_data_in_shared_memory
    )
    test_set = C16DatasetSharedMemory(
        test_wsi_names, test_wsi_labels, root=args.dataset_root, 
        cluster_labels=None, transform=test_transform,
        shared_data=test_data_in_shared_memory
    )
    
    # 設定 Sampler
    if world_size > 1:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    else:
        train_sampler = RandomSampler(train_set)

    if args.balanced_sampling:
        train_set = ClassBalancedDataset(dataset=train_set,oversample_thr=0.5)

    # 修正 DataLoader 的隨機種子
    generator = torch.Generator()
    if args.fix_loader_random:
        generator.manual_seed(7784414403328510413)
    
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), pin_memory=True, num_workers=args.num_workers,
        persistent_workers=True, prefetch_factor=2, generator=generator, collate_fn=collate_fn
    )
    
    # 只有主進程需要載入測試集進行評估
    test_loader = None
    if rank == 0:
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    
    model = MIL(
        n_classes=args.num_classes, 
        mil=args.mil_method
    ).to(device)
    
    if world_size > 1:
        # HMIL 可能需要 find_unused_parameters=True，取決於模型實現
        model = DDP(model, device_ids=[rank], find_unused_parameters=True) 

    # 載入預訓練權重
    if args.pretrain:
        state_dict = None
        if rank == 0:
            state_dict = torch.load(args.pretrain_model_path, map_location='cpu', weights_only=False)
            state_dict.pop('predictor.weight', None)
            state_dict.pop('predictor.bias', None)
        
        if world_size > 1:
            state_dict = broadcast_state_dict(state_dict, device, rank)
        
        model_to_load = model.module if world_size > 1 else model
        missing_keys, unexpected_keys = model_to_load.load_state_dict(state_dict, strict=False)
        print_and_log(f"Process {rank}: Missing keys: {missing_keys}", args.log_file, args.no_log)
        print_and_log(f"Process {rank}: Unexpected keys: {unexpected_keys}", args.log_file, args.no_log)
    
    # 凍結部分權重
    if args.frozen:
        for name, param in model.named_parameters():
            if "predictor" not in name:
                param.requires_grad_(False)
    
    criterions = {
        'cls': BuildClsLoss(args),
        # 'com': CompactnessLoss() # 假設 CompactnessLoss 不需要特殊參數
    }
    
    # 'validation_loop' 也不使用 'loss_weights'，但為保持一致性，我們保留
    loss_weights = {
        'cls': 1.0, # 主要的分類損失權重通常為 1
        # 'com': args.com_loss_weight
    }
    
    optimizer_params = filter(lambda p: p.requires_grad, model.parameters())
    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(optimizer_params, lr=args.lr, weight_decay=args.weight_decay)

    # 設定學習率排程
    scheduler = None
    if args.lr_sche == 'cosine':
        steps = args.num_epoch * len(train_loader) if args.lr_supi else args.num_epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, 0)
    elif args.lr_sche == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.num_epoch // 2, 0.2)
    elif args.lr_sche == 'cycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.num_epoch, steps_per_epoch=len(train_loader))

    # ---> 僅評估模式
    if args.eval_only:
        if rank == 0:
            checkpoint = torch.load(os.path.join(args.model_path, 'ckp.pt'), weights_only=False)
            model_to_load = model.module if world_size > 1 else model
            model_to_load.load_state_dict(checkpoint['model'])
            # **** 警告: validation_loop 可能與 HMIL 模型輸出不相容 ****
            hmil_validation_loop(args, model, test_loader, device, criterions, loss_weights, rank)
            # validation_loop(args, model, train_loader, device, criterions, loss_weights, rank, if_train_data=True)
        return

    # ---> 訓練迴圈
    train_time_meter = AverageMeter()
    for epoch in range(args.num_epoch):
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
        
        # === MODIFICATION START: 'criterions' 和 'loss_weights' 不再傳遞 ===
        # 因為 'training_loop' 內部將實作 HMIL 專用的損失計算
        # 'training_loop' 現在將返回一個包含多個損失的字典
        train_loss, start_time, end_time = hmil_training_loop(
            args, model, train_loader, optimizer, device, 
            amp_autocast, scheduler, epoch, rank
        )
        # === MODIFICATION END ===
        
        train_time_meter.update(end_time - start_time)
        
        # 'train_loss' 是一個包含多個損失的字典
        total_loss = train_loss.get('total', 0.0) # 從字典中獲取 'total'
        
        print_and_log(f'\rEpoch [{epoch+1}/{args.num_epoch}] train loss: {total_loss:.1E}, time: {train_time_meter.val:.3f}({train_time_meter.avg:.3f})', 
                      args.log_file, args.no_log)
        # print(f'\rEpoch [{epoch+1}/{args.num_epoch}] train loss: {total_loss:.1E}, time: {train_time_meter.val:.3f}({train_time_meter.avg:.3f})')
                      
        # 只有主進程儲存模型
        if rank == 0:
            model_state_dict = model.module.state_dict() if world_size > 1 else model.state_dict()
            checkpoint = {
                'model': model_state_dict,
                'lr_sche': scheduler.state_dict() if scheduler else None,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
            }
            # if (epoch + 1) % args.save_epoch == 0 or (epoch + 1) == args.num_epoch:
            #     torch.save(checkpoint, os.path.join(args.model_path, f'epoch_{epoch+1}_model.pt'))
            torch.save(checkpoint, os.path.join(args.model_path, 'ckp.pt'))

        
    # ---> 訓練結束後的最終儲存與評估
    if rank == 0:
        torch.save(checkpoint, os.path.join(args.model_path, 'ckp.pt'))
        
        # 載入最佳模型進行最終評估
        best_checkpoint = torch.load(os.path.join(args.model_path, 'ckp.pt'), weights_only=False)
        model_to_load = model.module if world_size > 1 else model
        info = model_to_load.load_state_dict(best_checkpoint['model'])
        print_and_log(f'Loaded model info: {info}', args.log_file, args.no_log)
        
        print_and_log('Info: Final evaluation for the test set', args.log_file, args.no_log)
        # **** 警告: validation_loop 可能與 HMIL 模型輸出不相容 ****
        hmil_validation_loop(args, model, test_loader, device, criterions, loss_weights, rank)
        # validation_loop(args, model, train_loader, device, criterions, loss_weights, rank, if_train_data=True)
    
    if world_size > 1:
        cleanup_ddp()
        


# --- 主程式進入點 ---

def main():
    """主函數：解析參數、準備資料並啟動訓練"""
    args = parse_args()
    print_and_log(args, args.log_file, args.no_log)
    
    world_size = args.world_size

    # 在主進程中準備資料元數據
    # *注意*: prepare_metadata 現在會修改 args.n_class = [coarse_num, fine_num]
    train_wsi_names, train_wsi_labels, train_cluster_labels, \
    test_wsi_names, test_wsi_labels = prepare_metadata(args)
    
    # *** 關鍵: 覆蓋 prepare_metadata 設置的 n_class ***
    # parse_args 根據 mil_method == 'hmil' 將 num_classes 設置為 [2, 6]
    # 這將傳遞給模型 (MIL(n_classes=args.num_classes...))
    # prepare_metadata 設置的 args.n_class 用於 HMIL 損失函數
    # (compute_hierarchical_loss)
    # 確保 parse_args 中的 'hmil' 邏輯是您想要的
    if args.mil_method == 'hmil':
        # 確保 n_class (用於損失) 和 num_classes (用於模型) 一致
        args.n_class = args.num_classes 
        print_and_log(f"Info: HMIL active. Overriding n_class for model AND loss to {args.num_classes}", args.log_file, args.no_log)


    # 將資料預載入到共享記憶體
    print_and_log("Loading datasets into shared memory...", args.log_file, args.no_log)
    train_data_in_shared_memory = C16DatasetSharedMemory.preload_to_shared_memory(
        train_wsi_names, 
        root_dir=args.dataset_root, 
        memory_limit_ratio=args.memory_limit_ratio,
    )
    # 測試集通常較小，這裡可以選擇不預載入或少量載入
    test_data_in_shared_memory = {} 
    print_and_log("Datasets loaded into shared memory.", args.log_file, args.no_log)
    
    # 組合傳遞給子進程的參數
    spawn_args = (
        world_size, args, 
        train_wsi_names, train_wsi_labels, train_cluster_labels, train_data_in_shared_memory,
        test_wsi_names, test_wsi_labels, test_data_in_shared_memory
    )

    if world_size > 1:
        mp.spawn(run_training_process, args=spawn_args, nprocs=world_size, join=True)
    else:
        # 單 GPU 模式
        run_training_process(0, *spawn_args)

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='MIL Training Script')

    # Dataset 
    parser.add_argument('--datasets', default='gc_v15', type=str, help='[gc_v15, gc_10k, gc_2625]')
    parser.add_argument('--dataset_root', default='/data/wsi/TCTGC50k-features/gigapath-coarse', type=str)
    parser.add_argument('--fix_loader_random', action='store_true', help='Fix dataloader random seed')
    parser.add_argument('--balanced_sampling', default=0, type=int)
    
    # *** 'num_classes' 預設值被修改。HMIL 邏輯將在下面覆蓋它 ***
    parser.add_argument('--num_classes', default=2, type=int, 
                        help='Number of classes. For HMIL, this will be overridden by a list [coarse, fine].')
    
    
    # Augment
    parser.add_argument('--patch_drop', default=1.0, type=float)
    parser.add_argument('--patch_pad', default=1.0, type=float)
    
    # Train
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    
    # Loss
    parser.add_argument('--loss', default='bce', type=str, help='[ce, bce, asl, softbce, ranking, aploss, focal]')
    parser.add_argument('--loss_neg', default=1, type=int, help='if use negative loss')
    parser.add_argument('--loss_drop_weight', default=1., type=float)
    parser.add_argument('--gamma_neg', default=4.0, type=float)
    parser.add_argument('--gamma_pos', default=1.0, type=float)
    parser.add_argument('--alpha', default=0.9, type=float)
    parser.add_argument('--gamma', default=2, type=float)
    parser.add_argument('--neg_weight', default=0.0, type=float, help='Weight for positive sample in SoftBCE')
    parser.add_argument('--neg_margin', default=0, type=float, help='if use neg_margin in ranking loss')
    parser.add_argument('--multi_label', default=1, type=int, help='if use multi label by ranking')
    
    # Optimizer & Scheduler
    parser.add_argument('--opt', default='adam', type=str, help='[adam, adamw]')
    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--lr', default=2e-3, type=float)
    parser.add_argument('--lr_sche', default='cosine', type=str, help='[cosine, step, const, cycle]')
    parser.add_argument('--lr_supi', action='store_true', help='LR scheduler update per iteration')
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--accumulation_steps', default=1, type=int)
    parser.add_argument('--clip_grad', default=0.0, type=float)
    
    # Model
    # *** 關鍵：確保 'hmil' 是支援的選項 ***
    parser.add_argument('--mil_method', default='abmil', type=str, help='[abmil, transmil, dsmil, clam, hmil]')
    parser.add_argument('--input_dim', default=1536, type=int)
    parser.add_argument('--dropout', default=0.25, type=float)
    
    # Misc
    parser.add_argument('--output_path', type=str, default='./output-model')
    parser.add_argument('--project', default='gcv15', type=str)
    parser.add_argument('--title', default='gigapath-abmil-refactored', type=str)
    parser.add_argument('--log_iter', default=100, type=int)
    parser.add_argument('--amp', action='store_true', help='Automatic Mixed Precision')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--save_epoch', default=25, type=int)
    parser.add_argument('--save_logits', default=1, type=int)
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--keep_psize_collate', default=0, type=int)

    # K-means Augmentation
    parser.add_argument('--kmeans-k', default=5, type=int)
    parser.add_argument('--kmeans-ratio', default=0.3, type=float)
    parser.add_argument('--kmeans-min', default=20, type=int)
    
    # Pretrain & Frozen
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--frozen', action='store_true')
    parser.add_argument('--pretrain_model_path', default='', type=str)
    
    parser.add_argument('--world_size', type=int, default=1, help='World Size for distributed training')
    
    parser.add_argument('--contrastive_temp', default=0.1, type=float, help='Temperature for contrastive loss')
    parser.add_argument('--contrastive_temp_epoch', default=10, type=int, help='Epoch to change contrastive temperature')
    # === MODIFICATION END ===
    
    args = parser.parse_args()
    
    # 根據資料集設定路徑
    if args.datasets == 'gc_v15':
        args.train_label_path = '/data/wsi/TCTGC50k-labels/6_labels/TCTGC50k-v15-train.csv'
        args.test_label_path = '/data/wsi/TCTGC50k-labels/6_labels/TCTGC50k-v15-test.csv'
        args.train_cluster_path = f'../datatools/TCTGC50k/cluster/kmeans_{args.kmeans_k}.csv'
        args.memory_limit_ratio = 0
    elif args.datasets[:3] == 'gc_' and args.datasets[-1] == 'k':
        num = int((args.datasets.split('_')[1].split('k')[0]))
        args.train_label_path = f'/data/wsi/TCTGC10k-labels/6_labels/TCTGC{num}k-v15-train.csv'
        args.test_label_path = f'/data/wsi/TCTGC10k-labels/6_labels/TCTGC{num}k-v15-test.csv'
        args.train_cluster_path = f'../datatools/TCTGC{num}k/cluster/kmeans_{args.kmeans_k}.csv'
        args.memory_limit_ratio = 0
    elif args.datasets == 'gc_2625':
        args.train_label_path = '/data/wsi/TCTGC2625-labels/6_labels/TCTGC-2625-train.csv'
        args.test_label_path = '/data/wsi/TCTGC2625-labels/6_labels/TCTGC-2625-test.csv'
        args.dataset_root = '/home1/wsi/gc-all-features/frozen/gigapath1'
        # TODO
        args.train_cluster_path = None
    
    # 'class_labels' 用於 'validation_loop'。
    # 這 *必須* 根據 HMIL 的評估目標（coarse 或 fine）進行調整。
    
    # *** 關鍵：HMIL 邏輯 ***
    if args.mil_method == 'hmil':
        args.num_classes = [2, 6] 
        args.mapping = "0:0, 1:1, 2:1, 3:1, 4:1, 5:1"
        args.class_labels = ['nilm', 'ascus', 'asch', 'lsil', 'hsil', 'agc']
        args.memory_limit_ratio = 0
        
    else:
        # 原始框架的邏輯
        if args.num_classes == 2:
            args.class_labels = ['Normal', 'Abnormal']
        else:
            args.class_labels = ['nilm', 'ascus', 'asch', 'lsil', 'hsil', 'agc']
            args.memory_limit_ratio = 0
    
    # 設定輸出路徑
    args.model_path = os.path.join(args.output_path, args.project, args.title)
    args.log_file = os.path.join(args.model_path, 'log.txt')
    os.makedirs(args.model_path, exist_ok=True)
    
    return args

if __name__ == '__main__':
    main()