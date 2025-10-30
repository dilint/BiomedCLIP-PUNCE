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

# 导入自定义模块
from dataloader import C16DatasetSharedMemory, ClassBalancedDataset
from models import MIL, Valina_MIL
from augments import PatchFeatureAugmenter
from loss import BuildClsLoss
from utils import print_and_log, seed_torch
from timm.utils import AverageMeter, dispatch_clip_grad
from trainer.hmil_trainer import hmil_training_loop, hmil_validation_loop
from trainer.abmil_trainer import abmil_training_loop, abmil_validation_loop

ID_TO_LABEL = {
    0: 'nilm', 1: 'ascus', 2: 'asch', 3: 'lsil', 
    4: 'hsil', 5: 'agc', 6: 't', 7: 'm', 8: 'bv'
}
LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}

# --- 分布式训练设定 ---

def setup_ddp(rank, world_size):
    """初始化分布式训练环境 (DDP)"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def broadcast_state_dict(state_dict, device, rank):
    """从主进程 (rank 0) 广播状态字典到所有其他进程"""
    if rank == 0:
        objects = [state_dict]
        dist.broadcast_object_list(objects, src=0)
        return state_dict
    else:
        objects = [None]
        dist.broadcast_object_list(objects, src=0)
        return objects[0]

import pandas as pd
import numpy as np
from utils import print_and_log 
import os 

def custom_collate_fn(batch):
    cell_images, patient_labels, third_elements = zip(*batch)
    return list(cell_images), list(patient_labels), list(third_elements)

def prepare_metadata(args):
    """
    从 CSV 读取元数据。
    使用硬编码规则 (label > 0) 创建 [N, 2] 分层标签。
    同时动态生成 args.fine_to_coarse_map 和 args.n_class 供后续使用。
    """
    if 'gc' not in args.datasets.lower():
        raise ValueError(f"不支持的数据集: {args.datasets}")

    # --- 训练集 ---
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
    
    # ---验证集 ---
    df_val = pd.read_csv(args.val_label_path)
    val_wsi_names = df_val['wsi_name'].values
    val_wsi_labels_fine = df_val['wsi_label'].map(LABEL_TO_ID).values
    
    # --- 测试集 ---
    df_test = pd.read_csv(args.test_label_path)
    test_wsi_names = df_test['wsi_name'].values
    test_wsi_labels_fine = df_test['wsi_label'].map(LABEL_TO_ID).values
    
    # --- 组合标签 ---
    train_wsi_labels_coarse = (train_wsi_labels_fine > 0).astype(int)
    val_wsi_labels_coarse = (val_wsi_labels_fine > 0).astype(int) 
    test_wsi_labels_coarse = (test_wsi_labels_fine > 0).astype(int)
    
    train_wsi_labels = np.column_stack((train_wsi_labels_coarse, train_wsi_labels_fine))
    val_wsi_labels = np.column_stack((val_wsi_labels_coarse, val_wsi_labels_fine))
    test_wsi_labels = np.column_stack((test_wsi_labels_coarse, test_wsi_labels_fine))
    
    print_and_log("Info: 标签已转换为 [N, 2] 层次结构格式 (Coarse, Fine).", args.log_file, args.no_log)
    
    return train_wsi_names, train_wsi_labels, train_cluster_labels, \
           val_wsi_names, val_wsi_labels, \
           test_wsi_names, test_wsi_labels

# --- 核心训练与评估流程 ---
def run_training_process(rank, world_size, args, 
                         train_wsi_names, train_wsi_labels, train_cluster_labels, train_data_in_shared_memory,
                         val_wsi_names, val_wsi_labels, val_data_in_shared_memory,
                         test_wsi_names, test_wsi_labels, test_data_in_shared_memory):
    """
    单个 GPU 进程执行的主训练函数。
    """
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    if world_size > 1:
        setup_ddp(rank, world_size)
    
    seed_torch(args.seed + rank)
    
    args.rank = rank
    
    if rank != 0:
        args.no_log = True

    print_and_log(f'进程 {rank}: 数据集: {args.datasets}', args.log_file, args.no_log)

    # ---> 初始化设定
    amp_autocast = torch.cuda.amp.autocast if args.amp else suppress
    
    train_cluster_labels_tensor = [torch.tensor(labels) for labels in train_cluster_labels] if train_cluster_labels is not None else None

    # ---> 资料加载 (DataLoader)
    # 设定资料增强
    train_transform = test_transform = None
    if args.patch_drop:
        train_transform = PatchFeatureAugmenter(kmeans_k=args.kmeans_k, kmeans_ratio=args.kmeans_ratio, kmeans_min=args.kmeans_min)
    elif args.patch_pad:
        train_transform = PatchFeatureAugmenter(augment_type='none')
    if args.patch_pad:
        test_transform = PatchFeatureAugmenter(augment_type='none')
    
    # 从共享内存建立 Dataset
    train_set = C16DatasetSharedMemory(
        train_wsi_names, train_wsi_labels, root=args.dataset_root, 
        cluster_labels=train_cluster_labels_tensor, transform=train_transform,
        shared_data=train_data_in_shared_memory
    )
    
    # --- 【新增】仅在 rank 0 建立 val_set 和 test_set ---
    val_set = None
    test_set = None
    if rank == 0:
        val_set = C16DatasetSharedMemory(
            val_wsi_names, val_wsi_labels, root=args.dataset_root, 
            cluster_labels=None, transform=test_transform,
            shared_data=val_data_in_shared_memory
        )
        test_set = C16DatasetSharedMemory(
            test_wsi_names, test_wsi_labels, root=args.dataset_root, 
            cluster_labels=None, transform=test_transform,
            shared_data=test_data_in_shared_memory
        )
    # ----------------------------------------------------

    # 设定 Sampler
    if world_size > 1:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    else:
        train_sampler = RandomSampler(train_set)

    if args.balanced_sampling:
        train_set = ClassBalancedDataset(dataset=train_set,oversample_thr=0.5)

    # 修正 DataLoader 的随机种子
    generator = torch.Generator()
    if args.fix_loader_random:
        generator.manual_seed(7784414403328510413)
    
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), pin_memory=True, num_workers=args.num_workers,
        persistent_workers=True, prefetch_factor=2, generator=generator, collate_fn=custom_collate_fn
    )
    
    val_loader = None
    test_loader = None
    if rank == 0:
        # 验证集和测试集通常使用 batch_size=1
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, collate_fn=default_collate)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, collate_fn=default_collate)
    # ------------------------------------------------------

    if args.mil_method == 'hmil':
        model = MIL(n_classes=args.num_classes,mil=args.mil_method).to(device)
        validation_loop = hmil_validation_loop
    else:
        model = Valina_MIL(input_dim=args.input_dim, mlp_dim=1024, n_classes=args.num_classes,
            mil=args.mil_method, dropout=args.dropout).to(device)
        validation_loop = abmil_validation_loop

    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True) 

    # 载入预训练权重
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
        print_and_log(f"进程 {rank}: 缺失的键: {missing_keys}", args.log_file, args.no_log)
        print_and_log(f"进程 {rank}: 意外的键: {unexpected_keys}", args.log_file, args.no_log)
    
    # 冻结部分权重
    if args.frozen:
        for name, param in model.named_parameters():
            if "predictor" not in name:
                param.requires_grad_(False)
    
    criterions = {
        'coarse_cls': BuildClsLoss(args),
        'fine_cls': BuildClsLoss(args),
        # 'com': CompactnessLoss() # 假设 CompactnessLoss 不需要特殊参数
    }
    
    # 'validation_loop' 也不使用 'loss_weights'，但为保持一致性，我们保留
    loss_weights = {
        'coarse_cls': 1.0, # 主要的分类损失权重通常为 1
        'fine_cls': 1.0, # 主要的分类损失权重通常为 1
        # 'com': args.com_loss_weight
    }
    
    optimizer_params = filter(lambda p: p.requires_grad, model.parameters())
    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(optimizer_params, lr=args.lr, weight_decay=args.weight_decay)

    # 设定学习率排程
    scheduler = None
    if args.lr_sche == 'cosine':
        steps = args.num_epoch * len(train_loader) if args.lr_supi else args.num_epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, 0)
    elif args.lr_sche == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.num_epoch // 2, 0.2)
    elif args.lr_sche == 'cycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.num_epoch, steps_per_epoch=len(train_loader))

    # ---> 仅评估模式
    if args.eval_only:
        if rank == 0:
            checkpoint = torch.load(os.path.join(args.model_path, 'ckp.pt'), weights_only=False)
            model_to_load = model.module if world_size > 1 else model
            model_to_load.load_state_dict(checkpoint['model'])
            
            print_and_log('信息: [仅评估模式] 正在验证集上评估...', args.log_file, args.no_log)
            validation_loop(args, model, val_loader, device, criterions, val_set_name="Validation")
            
            print_and_log('信息: [仅评估模式] 正在测试集上评估...', args.log_file, args.no_log)
            validation_loop(args, model, test_loader, device, criterions, val_set_name="Test")
        return

    # ---> 训练循环
    train_time_meter = AverageMeter()
    best_val_auc = 0.0
    
    for epoch in range(args.num_epoch):
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
        
        if args.mil_method == 'hmil':
            train_loss, start_time, end_time = hmil_training_loop(
                args, model, train_loader, optimizer, device, 
                amp_autocast, scheduler, epoch, rank)
        else:
            train_loss, start_time, end_time = abmil_training_loop(
                args, model, train_loader, optimizer, device,
                amp_autocast, criterions, loss_weights, scheduler, epoch, rank)
            
        train_time_meter.update(end_time - start_time)
        
        # 'train_loss' 是一个包含多个损失的字典
        total_loss = train_loss.get('total', 0.0) # 从字典中获取 'total'
        
        print_and_log(f'\rEpoch [{epoch+1}/{args.num_epoch}] 训练损失: {total_loss:.1E}, 时间: {train_time_meter.val:.3f}({train_time_meter.avg:.3f})', 
                      args.log_file, args.no_log)
        
        if rank == 0:
            # 'coarse_auc' (粗分类AUC) 是您关心的主要 AUC 指标。
            print_and_log(f'Epoch [{epoch+1}/{args.num_epoch}] 正在验证集上评估...', args.log_file, args.no_log)
            current_val_auc = validation_loop(args, model, val_loader, device, criterions, val_set_name="Validation")
            
            print_and_log(f'Epoch [{epoch+1}/{args.num_epoch}] 验证集 Coarse AUC: {current_val_auc:.4f}', 
                          args.log_file, args.no_log)

            # 检查是否为最佳模型
            if current_val_auc > best_val_auc:
                best_val_auc = current_val_auc
                print_and_log(f'---> 发现新的最佳模型! AUC: {best_val_auc:.4f} 在 epoch {epoch+1}', 
                              args.log_file, args.no_log)
                
                # 保存最佳模型
                model_state_dict = model.module.state_dict() if world_size > 1 else model.state_dict()
                checkpoint = {
                    'model': model_state_dict,
                    'lr_sche': scheduler.state_dict() if scheduler else None,
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'best_val_auc': best_val_auc # 存储当时的最佳指标
                }
                torch.save(checkpoint, os.path.join(args.model_path, 'ckp.pt'))
    
    # ---> 训练结束后的最终评估
    if rank == 0:
        print_and_log(f'训练完成。最佳验证集 AUC: {best_val_auc:.4f}', args.log_file, args.no_log)
        print_and_log('信息: 正在加载最佳模型 (ckp.pt) 用于最终测试...', args.log_file, args.no_log)
        best_checkpoint = torch.load(os.path.join(args.model_path, 'ckp.pt'), weights_only=False)
        model_to_load = model.module if world_size > 1 else model
        info = model_to_load.load_state_dict(best_checkpoint['model'])
        
        print_and_log(f'已加载最佳模型 (来自 epoch {best_checkpoint.get("epoch", "N/A")})。加载信息: {info}', 
                      args.log_file, args.no_log)
        print_and_log('信息: 正在 【测试集】 上进行最终评估', args.log_file, args.no_log)
        validation_loop(args, model, test_loader, device, criterions, val_set_name="Test")
    if world_size > 1:
        cleanup_ddp()
        

def main():
    """主函数：解析参数、准备资料并启动训练"""
    args = parse_args()
    print_and_log(args, args.log_file, args.no_log)
    world_size = args.world_size

    train_wsi_names, train_wsi_labels, train_cluster_labels, \
    val_wsi_names, val_wsi_labels, \
    test_wsi_names, test_wsi_labels = prepare_metadata(args)
    
    # 将资料预加载到共享内存
    print_and_log("正在加载训练集到共享内存...", args.log_file, args.no_log)
    train_data_in_shared_memory = C16DatasetSharedMemory.preload_to_shared_memory(
        train_wsi_names, 
        root_dir=args.dataset_root, 
        memory_limit_ratio=args.memory_limit_ratio,
    )
    
    val_data_in_shared_memory = {}
    test_data_in_shared_memory = {} 
    # ----------------------------------------------------
    print_and_log("数据已加载到共享内存。", args.log_file, args.no_log)
    
    spawn_args = (
        world_size, args, 
        train_wsi_names, train_wsi_labels, train_cluster_labels, train_data_in_shared_memory,
        val_wsi_names, val_wsi_labels, val_data_in_shared_memory,
        test_wsi_names, test_wsi_labels, test_data_in_shared_memory
    )
    # ------------------------------------

    if world_size > 1:
        mp.spawn(run_training_process, args=spawn_args, nprocs=world_size, join=True)
    else:
        # 单 GPU 模式
        run_training_process(0, *spawn_args)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MIL Training Script')

    # Dataset 
    parser.add_argument('--datasets', default='gc_v15', type=str, help='[gc_v15, gc_10k, gc_2625]')
    parser.add_argument('--dataset_root', default='/data/wsi/TCTGC50k-features/gigapath-coarse', type=str)
    parser.add_argument('--fix_loader_random', action='store_true', help='Fix dataloader random seed')
    parser.add_argument('--balanced_sampling', default=0, type=int)
    
    parser.add_argument('--num_classes', default=6, type=int, 
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
    # *** 关键：确保 'hmil' 是支持的选项 ***
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
    parser.add_argument('--save_logits', default=0, type=int)
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
    
    args = parser.parse_args()
    
    if args.datasets == 'gc_v15':
        args.train_label_path = '/data/wsi/TCTGC50k-labels/6_labels/TCTGC50k-v15-train.csv'
        # 假设验证集CSV文件的命名规范
        args.val_label_path = '/data/wsi/TCTGC50k-labels/6_labels/TCTGC50k-v15-test.csv' 
        args.test_label_path = '/data/wsi/TCTGC50k-labels/6_labels/TCTGC50k-v15-test.csv'
        args.train_cluster_path = f'../datatools/TCTGC50k/cluster/kmeans_{args.kmeans_k}.csv'
    elif args.datasets[:3] == 'gc_' and args.datasets[-1] == 'k':
        num = int((args.datasets.split('_')[1].split('k')[0]))
        args.train_label_path = f'/data/wsi/TCTGC10k-labels/6_labels/TCTGC{num}k-v15-train.csv'
        args.val_label_path = f'/data/wsi/TCTGC10k-labels/6_labels/TCTGC{num}k-v15-val.csv'
        args.test_label_path = f'/data/wsi/TCTGC10k-labels/6_labels/TCTGC{num}k-v15-test.csv'
        args.train_cluster_path = f'../datatools/TCTGC{num}k/cluster/kmeans_{args.kmeans_k}.csv'
    elif args.datasets == 'gc_2625':
        args.train_label_path = '/data/wsi/TCTGC2625-labels/6_labels/TCTGC-2625-train.csv'
        args.val_label_path = '/data/wsi/TCTGC2625-labels/6_labels/TCTGC-2625-test.csv' 
        args.test_label_path = '/data/wsi/TCTGC2625-labels/6_labels/TCTGC-2625-test.csv'
        args.dataset_root = '/home1/wsi/gc-all-features/frozen/gigapath1'
        args.train_cluster_path = None
    args.memory_limit_ratio = 0

    if 'gc' in args.datasets:
        args.num_classes = [2, 6] 
        args.mapping = "0:0, 1:1, 2:1, 3:1, 4:1, 5:1"
        args.class_labels = ['nilm', 'ascus', 'asch', 'lsil', 'hsil', 'agc']
        args.coarse_class_labels = ['neg', 'pos']
        args.memory_limit_ratio = 0
        args.n_class = args.num_classes 
    
    # 设定输出路径
    args.model_path = os.path.join(args.output_path, args.project, args.title)
    args.log_file = os.path.join(args.model_path, 'log.txt')
    os.makedirs(args.model_path, exist_ok=True)
    
    return args

if __name__ == '__main__':
    main()