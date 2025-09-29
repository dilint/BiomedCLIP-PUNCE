import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, default_collate, Dataset ### MODIFIED ###
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.functional import one_hot
import torch.nn.functional as F ### ADDED ###

import numpy as np
import pandas as pd
import argparse
import os
import random
from contextlib import suppress

# 匯入自定義模組
from dataloader import C16DatasetSharedMemory, C16DatasetTwoView
from model import MIL
from loss import BuildClsLoss
from augments import PatchFeatureAugmenter
from utils import (
    print_and_log, 
    seed_torch, 
    multi_class_scores_mtl, 
    multi_class_scores_mtl_ce, 
    save_metrics_to_excel, 
    save_logits
)
from timm.utils import AverageMeter, dispatch_clip_grad
from timm.models import model_parameters

# === 全域設定 ===
# 定義標籤與ID的對應關係
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

def prepare_metadata(args):
    """
    從 CSV 檔案中讀取並準備訓練和測試集的元數據。
    """
    if 'gc' not in args.datasets.lower():
        raise ValueError(f"不支援的資料集: {args.datasets}")

    df_train = pd.read_csv(args.train_label_path)
    df_cluster = pd.read_csv(args.train_cluster_path)
    
    # 將聚類標籤合併到訓練 DataFrame
    df_train = df_train.merge(
        df_cluster[['wsi_name', 'cluster_label']], 
        on='wsi_name', 
        how='left'
    )
    
    train_wsi_names = df_train['wsi_name'].values
    train_wsi_labels = df_train['wsi_label'].map(LABEL_TO_ID).values
    train_cluster_labels = df_train['cluster_label'].apply(
        lambda x: [int(i) for i in x.split()]
    ).values
    
    df_test = pd.read_csv(args.test_label_path)
    test_wsi_names = df_test['wsi_name'].values
    test_wsi_labels = df_test['wsi_label'].map(LABEL_TO_ID).values
    
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
    collate_fn = default_collate
    amp_autocast = torch.cuda.amp.autocast if args.amp else suppress
    
    train_cluster_labels_tensor = [torch.tensor(labels) for labels in train_cluster_labels]

    # ---> 資料載入 (DataLoader)
    # ### MODIFIED ###: 設定兩種不同的資料增強
    train_transform_1 = PatchFeatureAugmenter(
        kmeans_k=args.kmeans_k, 
        kmeans_ratio=args.kmeans_ratio, 
        kmeans_min=args.kmeans_min
    )
    
        # 第二個分支使用不同的增強，例如 patch_pad 或不同的 drop 參數
    train_transform_2 = PatchFeatureAugmenter(augment_type='none') # 使用 'pad' 作為第二種增強
    test_transform = PatchFeatureAugmenter(augment_type='none')

    # ### MODIFIED ###: 使用 TwoViewDataset 包裝訓練集
    # 先建立一個不含增強的基礎訓練集
    train_set = C16DatasetTwoView(
        train_wsi_names, train_wsi_labels, root=args.dataset_root, 
        cluster_labels=train_cluster_labels_tensor, transform1=train_transform_1, transform2=train_transform_2, # 增強由 TwoViewDataset 處理
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

    # ---> 模型、損失函數、優化器設定
    model = MIL(
        input_dim=args.input_dim, mlp_dim=512, n_classes=args.num_classes,
        mil=args.mil_method, dropout=args.dropout
    ).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

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
    
    # ### MODIFIED ###: 加入一致性損失
    criterions = {
        'cls': BuildClsLoss(args),
        'con': nn.MSELoss() # 使用均方誤差作為一致性損失
    }
    
    loss_weights = {
        'cls': 1.0,
        'con': args.consistency_weight
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
            validation_loop(args, model, test_loader, device, criterions, loss_weights, rank)
        return

    # ---> 訓練迴圈
    train_time_meter = AverageMeter()
    for epoch in range(args.num_epoch):
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
        
        train_loss, start_time, end_time = training_loop(
            args, model, train_loader, optimizer, device, 
            amp_autocast, criterions, loss_weights, scheduler, epoch, rank
        )
        train_time_meter.update(end_time - start_time)
        
        total_loss = train_loss['total']
        print_and_log(f'\rEpoch [{epoch+1}/{args.num_epoch}] train loss: {total_loss:.1E}, time: {train_time_meter.val:.3f}({train_time_meter.avg:.3f})', 
                      args.log_file, args.no_log)
        
        # 只有主進程儲存模型
        if rank == 0:
            model_state_dict = model.module.state_dict() if world_size > 1 else model.state_dict()
            checkpoint = {
                'model': model_state_dict,
                'lr_sche': scheduler.state_dict() if scheduler else None,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
            }
            if (epoch + 1) % args.save_epoch == 0 or (epoch + 1) == args.num_epoch:
                torch.save(checkpoint, os.path.join(args.model_path, f'epoch_{epoch+1}_model.pt'))

    # ---> 訓練結束後的最終儲存與評估
    if rank == 0:
        torch.save(checkpoint, os.path.join(args.model_path, 'ckp.pt'))
        
        # 載入最佳模型進行最終評估
        best_checkpoint = torch.load(os.path.join(args.model_path, 'ckp.pt'), weights_only=False)
        model_to_load = model.module if world_size > 1 else model
        info = model_to_load.load_state_dict(best_checkpoint['model'])
        print_and_log(f'Loaded model info: {info}', args.log_file, args.no_log)
        
        print_and_log('Info: Final evaluation for the test set', args.log_file, args.no_log)
        validation_loop(args, model, test_loader, device, criterions, loss_weights, rank)
    
    if world_size > 1:
        cleanup_ddp()

def training_loop(args, model, loader, optimizer, device, amp_autocast, criterions, loss_weights, scheduler, epoch, rank):
    """### MODIFIED ###: 單個 epoch 的訓練迴圈，支援雙分支和多個損失函數"""
    start_time = time.time()
    
    loss_meters = {name: AverageMeter() for name in list(criterions.keys()) + ['total']}
    
    model.train()
    
    if epoch == 0 and rank == 0:
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in model.parameters())
        print_and_log(f'Total parameters: {num_total_param}, Tunable parameters: {n_parameters}', args.log_file)
    
    # ### MODIFIED ###: 解包兩個視圖
    for i, (bags, label, file_path) in enumerate(loader):
        bag1, bag2 = bags
        bag1, bag2, label = bag1.to(device), bag2.to(device), label.to(device)
        
        if epoch < 3 and i < 3 and rank == 0:
            print_and_log(list(map(lambda x: os.path.basename(x), file_path)), args.log_file, args.no_log)
            print_and_log(f"Bag1 shape: {bag1.shape}, Bag2 shape: {bag2.shape}", args.log_file, args.no_log)
            
        with amp_autocast():
            # ### MODIFIED ###: 兩個分支的前向傳播
            logits1 = model(bag1)
            logits2 = model(bag2)
            
            losses = {}
            # 分類損失 (兩個分支的平均)
            loss_cls1 = criterions['cls'](logits1, label)
            loss_cls2 = criterions['cls'](logits2, label)
            losses['cls'] = (loss_cls1 + loss_cls2) / 2.0
            
            # 一致性損失
            losses['con'] = criterions['con'](logits1, logits2)
            
            # 根據權重加總所有損失
            total_loss = sum(losses[name] * loss_weights[name] for name in losses)
            
        # --- 反向傳播 ---
        total_loss_scaled = total_loss / args.accumulation_steps
        total_loss_scaled.backward()
        
        if (i + 1) % args.accumulation_steps == 0:
            if args.clip_grad > 0.:
                dispatch_clip_grad(model_parameters(model), value=args.clip_grad, mode='norm')
            optimizer.step()
            optimizer.zero_grad()
            if args.lr_supi and scheduler:
                scheduler.step()

        # --- 記錄與日誌 ---
        for name, loss in losses.items():
            loss_meters[name].update(loss.item(), bag1.size(0))
        loss_meters['total'].update(total_loss.item(), bag1.size(0))

        if (i % args.log_iter == 0 or i == len(loader) - 1) and rank == 0:
            lr = optimizer.param_groups[0]['lr']
            loss_str = ', '.join([f'{name}: {meter.avg:.4f}' for name, meter in loss_meters.items()])
            print_and_log(f'[{i}/{len(loader)-1}] {loss_str}, lr: {lr:.5f}', args.log_file, args.no_log)

    end_time = time.time()
    
    avg_losses_summary = {}
    for name, meter in loss_meters.items():
        avg_loss = torch.tensor(meter.avg, device=device)
        if args.world_size > 1:
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
        avg_losses_summary[name] = avg_loss.item()
    
    if not args.lr_supi and scheduler:
        scheduler.step()
        
    return avg_losses_summary, start_time, end_time


def validation_loop(args, model, loader, device, criterions, loss_weights, rank):
    """評估迴圈，只在主進程 (rank 0) 執行"""
    if rank != 0:
        return
    
    model.eval()
    loss_meter = AverageMeter()
    
    all_logits, all_labels, all_onehot_labels, all_wsi_names = [], [], [], []
    
    with torch.no_grad():
        for bag, label, file_path in loader:
            bag, label = bag.to(device), label.to(device)
            batch_size = bag.size(0)
            
            logits = model(bag)
            loss = criterions['cls'](logits, label)
            
            loss_meter.update(loss.item(), batch_size)
            
            label_onehot = one_hot(label, num_classes=args.num_classes).cpu()
            all_labels.extend(label.cpu().numpy())
            all_onehot_labels.extend(label_onehot)
            all_wsi_names.extend([os.path.basename(p) for p in file_path])
            
            if args.loss in ['ce']:
                probabilities = torch.softmax(logits, dim=-1)
            else: # bce, asl, etc.
                probabilities = torch.sigmoid(logits)

            if args.num_classes == 2:
                all_logits.extend(probabilities[:, 1].cpu().numpy())
            else:
                all_logits.extend(probabilities.cpu().numpy())

    # 計算並儲存評估指標
    eval_func = multi_class_scores_mtl_ce if (args.loss in ['ce'] and not args.multi_label) else multi_class_scores_mtl
    
    results = eval_func(all_onehot_labels, all_logits, args.class_labels, all_wsi_names, threshold=args.threshold)
    roc_auc, accuracies, recalls, precisions, fscores, thresholds, cancer_matrix, microbial_matrix = results

    output_excel_path = os.path.join(args.model_path, 'metrics.xlsx')
    save_metrics_to_excel(roc_auc, accuracies, recalls, precisions, fscores, thresholds, cancer_matrix, microbial_matrix, args.class_labels, output_excel_path)
    
    if args.save_logits:
        output_logits_path = os.path.join(args.model_path, 'logits.csv')
        save_logits(all_onehot_labels, all_logits, args.class_labels, all_wsi_names, output_logits_path)

    print_and_log(f'Validation Loss: {loss_meter.avg:.4f}', args.log_file, args.no_log)


# --- 主程式進入點 ---

def main():
    """主函數：解析參數、準備資料並啟動訓練"""
    args = parse_args()
    print_and_log(args, args.log_file, args.no_log)
    
    world_size = args.world_size

    # 在主進程中準備資料元數據
    train_wsi_names, train_wsi_labels, train_cluster_labels, \
    test_wsi_names, test_wsi_labels = prepare_metadata(args)
    
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
    parser.add_argument('--datasets', default='gc_v15', type=str, help='[gc_v15, gc_10k]')
    parser.add_argument('--dataset_root', default='/data/wsi/TCTGC50k-features/gigapath-coarse', type=str)
    parser.add_argument('--fix_loader_random', action='store_true', help='Fix dataloader random seed')
    parser.add_argument('--num_classes', default=6, type=int)
    
    # Augment
    parser.add_argument('--patch_drop', default=1.0, type=float)
    parser.add_argument('--patch_pad', default=1.0, type=float)
    
    # Train
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    
    # Loss
    parser.add_argument('--loss', default='bce', type=str, help='[ce, bce, asl, softbce, ranking, aploss, focal]')
    parser.add_argument('--consistency_weight', default=1.0, type=float, help='Weight for consistency loss') ### ADDED ###
    parser.add_argument('--loss_drop_weight', default=1., type=float)
    parser.add_argument('--gamma_neg', default=4.0, type=float)
    parser.add_argument('--gamma_pos', default=1.0, type=float)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.25, type=float)
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
    parser.add_argument('--mil_method', default='abmil', type=str, help='[abmil, transmil, dsmil, clam]')
    parser.add_argument('--input_dim', default=1536, type=int)
    parser.add_argument('--dropout', default=0.25, type=float)
    
    # Misc
    parser.add_argument('--output_path', type=str, default='./output-model')
    parser.add_argument('--project', default='gcv15', type=str)
    parser.add_argument('--title', default='gigapath-abmil-dual-branch', type=str) ### MODIFIED ###
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
    
    args = parser.parse_args()
    
    # 根據資料集設定路徑
    if args.datasets == 'gc_v15':
        args.train_label_path = '/data/wsi/TCTGC50k-labels/6_labels/TCTGC50k-v15-train.csv'
        args.test_label_path = '/data/wsi/TCTGC50k-labels/6_labels/TCTGC50k-v15-test.csv'
        args.train_cluster_path = f'../datatools/TCTGC50k/cluster/kmeans_{args.kmeans_k}.csv'
        args.class_labels = ['nilm', 'ascus', 'asch', 'lsil', 'hsil', 'agc']
        args.memory_limit_ratio = 0.4
    elif args.datasets == 'gc_10k':
        args.train_label_path = '/data/wsi/TCTGC10k-labels/6_labels/TCTGC10k-v15-train.csv'
        args.test_label_path = '/data/wsi/TCTGC10k-labels/6_labels/TCTGC10k-v15-test.csv'
        args.train_cluster_path = f'../datatools/TCTGC10k/cluster/kmeans_{args.kmeans_k}.csv'
        args.class_labels = ['nilm', 'ascus', 'asch', 'lsil', 'hsil', 'agc']
        args.memory_limit_ratio = 0
    
    # 設定輸出路徑
    args.model_path = os.path.join(args.output_path, args.project, args.title)
    args.log_file = os.path.join(args.model_path, 'log.txt')
    os.makedirs(args.model_path, exist_ok=True)
    
    return args

if __name__ == '__main__':
    main()