import time
import torch
import wandb
import numpy as np
from copy import deepcopy
import torch.nn as nn
import pandas as pd
from dataloader import *
from model import *
from loss import *
from torch.utils.data import DataLoader, RandomSampler, default_collate, DistributedSampler
import argparse, os
from torch.nn.functional import one_hot
from contextlib import suppress
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from modules.vit_wsi.model_v1 import Model_V1
from timm.utils import AverageMeter,dispatch_clip_grad
from timm.models import  model_parameters
from augments import *
from utils import *

id2label = {
    0: 'nilm',
    1: 'ascus',
    2: 'asch',
    3: 'lsil',
    4: 'hsil',
    5: 'agc',
    6: 't',
    7: 'm',
    8: 'bv',}
label2id = {v: k for k, v in id2label.items()}

def setup(rank, world_size):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def main():
    # 解析参数
    args = parse_args()
    print_and_log(args, args.log_file)
    # 设置世界大小（GPU数量）
    world_size = args.world_size
    
    # 根据world_size决定是否使用多进程
    if world_size > 1:
        # 使用多进程启动分布式训练
        mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        # 单GPU训练，直接调用main_worker
        main_worker(0, 1, args)

def main_worker(rank, world_size, args):
    """每个GPU的工作进程"""
    # 设置设备
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # 只有在多GPU训练时才需要初始化分布式训练
    if world_size > 1:
        setup(rank, world_size)
    
    # 设置随机种子（确保所有进程有相同的随机种子）
    seed_torch(args.seed + rank)
    
    # 只在主进程记录日志
    args.rank = rank
    args.world_size = world_size
    args.no_log = args.no_log or (rank != 0)
    
    # 读取标签信息
    if 'gc' in args.datasets.lower():
        df_train = pd.read_csv(args.train_label_path)
        df_train = df_train.iloc[:int(len(df_train)*args.train_ratio)]
        
        df_cluster = pd.read_csv(args.train_cluster_path)
        # 合并到训练集 DataFrame，确保 wsi_name 对齐
        df_train = df_train.merge(
            df_cluster[['wsi_name', 'cluster_label']], 
            on='wsi_name', 
            how='left'  # 保留所有训练集样本，缺失的聚类标签设为 NaN
        )
        train_wsi_names = df_train['wsi_name'].values
        train_wsi_labels = df_train['wsi_label'].map(label2id).values
        df_test = pd.read_csv(args.test_label_path)
        test_wsi_names = df_test['wsi_name'].values
        test_wsi_labels = df_test['wsi_label'].map(label2id).values
        train_cluster_labels = df_train['cluster_label'].apply(
            lambda x: [int(i) for i in x.split()]).values

    acs, pre, rec, fs, auc, te_auc, te_fs=[],[],[],[],[],[],[]
    ckc_metric = [acs, pre, rec, fs, auc, te_auc, te_fs] # acs: [fold, fold] fold: [task1, task2]

    if not args.no_log:
        print_and_log('Dataset: ' + args.datasets, args.log_file)
    
    one_fold(args, ckc_metric, train_wsi_names, train_wsi_labels, test_wsi_names, test_wsi_labels, train_cluster_labels, device, rank)
    
    # 只有在多GPU训练时才需要清理分布式训练环境
    if world_size > 1:
        cleanup()

def one_fold(args, ckc_metric, train_p, train_l, test_p, test_l, train_c, device, rank):
    # --->initiation 
    if args.keep_psize_collate:
        collate_fn = collate_fn_wsi
    else:
        collate_fn = default_collate
    amp_autocast = torch.cuda.amp.autocast if args.amp else suppress
    
    train_c = [torch.tensor(labels) for labels in train_c]
    
    acs,pre,rec,fs,auc,te_auc,te_fs = ckc_metric

    # ***--->load data
    if args.datasets.lower() in ['gc_v15', 'gc_10k']:
        pad_augmenter = PatchFeatureAugmenter(augment_type='none')
        drop_augmenter = PatchFeatureAugmenter(kmeans_k=args.kmeans_k, kmeans_ratio=args.kmeans_ratio, kmeans_min=args.kmeans_min)
        
        train_transform = test_transform = None
        if args.patch_drop:
            train_transform = drop_augmenter
        elif args.patch_pad:
            train_transform = pad_augmenter
        if args.patch_pad:
            test_transform = pad_augmenter
        
        train_set = C16Dataset(train_p,train_l,root=args.dataset_root,cluster_labels=train_c,persistence=args.persistence,transform=train_transform)
        test_set = C16Dataset(test_p,test_l,root=args.dataset_root,cluster_labels=train_c,persistence=args.persistence,transform=test_transform)
    else:
        assert f'{args.datasets} dataset not found'
    
    ## 训练集是否进行类别平衡采样 默认不使用
    if args.imbalance_sampler:
        train_set = ClassBalancedDataset(train_set, oversample_thr=0.22)
    
    # 根据world_size决定使用哪种采样器
    if args.world_size > 1:
        # 使用分布式采样器
        train_sampler = DistributedSampler(
            train_set, 
            num_replicas=args.world_size, 
            rank=rank, 
            shuffle=True,
            seed=args.seed
        )
    else:
        # 单GPU训练，使用普通采样器
        train_sampler = RandomSampler(train_set) if args.imbalance_sampler else None
    
    generator = torch.Generator()
    if args.fix_loader_random:
        big_seed_list = 7784414403328510413
        generator.manual_seed(big_seed_list)
        
    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        sampler=train_sampler,  # 使用sampler而不是shuffle
        shuffle=(train_sampler is None),  # 如果没有sampler，则使用shuffle
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=True,
        prefetch_factor=2,
        generator=generator,
        collate_fn=collate_fn
    )
    
    # 测试集不需要分布式采样器
    if rank == 0:  # 只在主进程加载测试集
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    else:
        test_loader = None

    # 创建模型并移动到当前设备
    model = MIL(
        input_dim=args.input_dim,
        mlp_dim=512,
        n_classes=args.num_classes,
        mil=args.mil_method,
        dropout=args.dropout
    ).to(device)
    
    # 只有在多GPU训练时才使用DDP包装模型
    if args.world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False, gradient_as_bucket_view=True)

    if args.pretrain:
        # 主进程加载预训练权重，然后广播到所有进程
        if rank == 0:
            state_dict = torch.load(args.pretrain_model_path, map_location='cpu', weights_only=False)
            del state_dict['predictor.weight']
            del state_dict['predictor.bias']
        else:
            state_dict = None
        
        # 在多GPU训练时需要广播状态字典
        if args.world_size > 1:
            state_dict = broadcast_state_dict(state_dict, device, rank)
        
        # 加载权重
        if args.world_size > 1:
            missing_keys, unexpected_keys = model.module.load_state_dict(state_dict, strict=False)
        else:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
        if rank == 0:
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {unexpected_keys}")
    
    if args.frozen:
        # 根据是否使用DDP选择模型参数
        model_params = model.module.parameters() if args.world_size > 1 else model.parameters()
        for name, param in model_params:
            if "predictor" not in name:
                param.requires_grad_(False)
    
    # ***--->construct criterion 构造不同损失
    cls_criterion = BuildClsLoss(args)

    # optimizer
    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_sche == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, 0) if not args.lr_supi else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch*len(train_loader), 0)
    elif args.lr_sche == 'step':
        assert not args.lr_supi
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,args.num_epoch / 2, 0.2)
    elif args.lr_sche == 'const':
        scheduler = None
    elif args.lr_sche == 'cycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=args.lr,epochs=args.num_epoch,steps_per_epoch=len(train_loader))


    train_time_meter = AverageMeter()

    # 如果只用来评估测试集的性能
    if args.eval_only:
        if rank == 0:
            ckp = torch.load(os.path.join(args.model_path,'ckp.pt'), weights_only=False)
            if args.world_size > 1:
                model.module.load_state_dict(ckp['model'])
            else:
                model.load_state_dict(ckp['model'])
            val_loop(args, model, test_loader, device, cls_criterion, rank)
        return
    
    # 训练epoch
    for epoch in range(args.num_epoch):
        # 设置epoch给采样器，确保每个epoch的shuffle一致
        if args.world_size > 1:
            train_loader.sampler.set_epoch(epoch)
        
        train_loss, start, end = 0, 0, 0
        train_loss, start, end = train_loop(args, model, train_loader, optimizer, device, amp_autocast, cls_criterion, scheduler, epoch, rank)
        train_time_meter.update(end-start)
        
        if not args.no_log:
            print_and_log('\r Epoch [%d/%d] train loss: %.1E, time: %.3f(%.3f)' % 
                (epoch+1, args.num_epoch, train_loss, train_time_meter.val, train_time_meter.avg), args.log_file)
        
        # 只在主进程保存checkpoint
        if rank == 0:
            random_state = {
                'np': np.random.get_state(),
                'torch': torch.random.get_rng_state(),
                'py': random.getstate(),
                'loader': train_loader.generator.get_state() if args.fix_loader_random else '',
            }
            # 根据是否使用DDP选择模型状态字典
            model_state_dict = model.module.state_dict() if args.world_size > 1 else model.state_dict()
            
            ckp = {
                'model': model_state_dict,
                'lr_sche': scheduler.state_dict() if scheduler else None,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch+1,
                'random': random_state,
                'ckc_metric': [acs,pre,rec,fs,auc,te_auc,te_fs],
            }
            if epoch % args.save_epoch == 0 or epoch == args.num_epoch-1:
                ckp_file_name = f'epoch_{epoch}_model.pt'
                torch.save(ckp, os.path.join(args.model_path, ckp_file_name))

    if rank == 0:
        torch.save(ckp, os.path.join(args.model_path, 'ckp.pt'))
        # test
        if not args.no_log:
            best_std = torch.load(os.path.join(args.model_path, 'ckp.pt'), weights_only=False)
            if args.world_size > 1:
                info = model.module.load_state_dict(best_std['model'])
            else:
                info = model.load_state_dict(best_std['model'])
            print_and_log(info, args.log_file)
        
        print_and_log('Info: Evaluation for test set', args.log_file)
        aucs, acc, recs, precs, f1s, test_loss = val_loop(args, model, test_loader, device, cls_criterion, rank)
    
    return 

def train_loop(args, model, loader, optimizer, device, amp_autocast, cls_criterion, scheduler, epoch, rank):
    last_end = start = time.time()
    train_loss_log = 0.
    losses = {}
    loss_cls_meters = {}
    
    model.train()
    
    # 只在主进程打印参数信息
    if not args.no_log and epoch == 0:
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in model.parameters())
        print_and_log('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, n_parameters), args.log_file)
    
    for i, data in enumerate(loader):
        time_data_end = time.time()
        time_data = time_data_end - last_end
        
        train_loss = torch.tensor(0., device=device)
        optimizer.zero_grad()
        
        bag, label, file_path = data[0].to(device), data[1].to(device), data[2]
            
        with amp_autocast():
            train_logits = model(bag)
            cls_loss = cls_criterion(train_logits, label)
            losses['cls_loss'] = cls_loss.item()
            train_loss += cls_loss
            
            time_forward_end = time.time()
            time_forward = time_forward_end - time_data_end
        
        # 只在主进程打印前几个batch的信息
        if epoch < 3 and i < 3 and rank == 0:
            print_and_log(list(map(lambda x: os.path.basename(x), file_path)))
            print_and_log(bag.shape)
            print_and_log('time_data: %.3f, time_forward: %.3f' % (time_data, time_forward))
        
        train_loss = train_loss / args.accumulation_steps
        if args.clip_grad > 0.:
            dispatch_clip_grad(
                model_parameters(model),
                value=args.clip_grad, mode='norm')

        if (i+1) % args.accumulation_steps == 0:
            train_loss.backward()
            optimizer.step()
            if args.lr_supi and scheduler is not None:
                scheduler.step()
        
        for k, v in losses.items():
            if k not in loss_cls_meters:
                loss_cls_meters[k] = AverageMeter()
            loss_cls_meters[k].update(v, 1)
            
        # 只在主进程记录日志
        if (i % args.log_iter == 0 or i == len(loader)-1) and rank == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            if not args.no_log:
                print_and_log('[{}/{}] '.format(i, len(loader)-1)
                + ', '.join(['{}: {:.4f}'.format(k, v.avg) for k, v in loss_cls_meters.items()])
                + ', lr: {:.5f}'.format(lr), args.log_file)

        train_loss_log = train_loss_log + train_loss.item()
        last_end = time.time()
        
    end = time.time()
    train_loss_log = train_loss_log/len(loader)
    
    # 在多GPU训练时对所有进程的损失求平均
    if args.world_size > 1:
        train_loss_log_tensor = torch.tensor(train_loss_log, device=device)
        dist.all_reduce(train_loss_log_tensor, op=dist.ReduceOp.SUM)
        train_loss_log = train_loss_log_tensor.item() / args.world_size
    
    if not args.lr_supi and scheduler is not None:
        scheduler.step()
    
    return train_loss_log, start, end

def val_loop(args, model, loader, device, criterion, rank):
    """
    只在主进程进行评估
    """
    if rank != 0:
        return None, None, None, None, None, None
    
    model.eval()
    loss_cls_meter = AverageMeter()
    
    bag_logits, bag_labels, bag_onehot_labels, wsi_names = [], [], [], []
    pad_augmenter = PatchFeatureAugmenter(augment_type='none')
    
    with torch.no_grad():
        for i, data in enumerate(loader):
            bag, label, file_path = data[0].to(device), data[1].to(device), data[2]
            batch_size = bag.size(0)
            label_onehot = one_hot(label.view(batch_size,-1).contiguous(), num_classes=args.num_classes).squeeze(1).float()
            wsi_name = [os.path.basename(wsi_path) for wsi_path in data[2]]
            
            test_logits = model(bag)
            batch_size = bag.size(0)
            bag_labels.extend(data[1])
            bag_onehot_labels.extend(label_onehot)
            wsi_names.extend(wsi_name)
            test_logits = test_logits.detach()
            test_loss = criterion(test_logits.view(batch_size,-1).contiguous(), label)    
            
            if args.loss in ['ce']:
                if args.num_classes == 2:
                    bag_logits.extend(torch.softmax(test_logits,dim=-1)[:,1].cpu().numpy())
                else:
                    bag_logits.extend(torch.softmax(test_logits, dim=-1).cpu().numpy())
            elif args.loss in ['bce', 'softbce', 'ranking', 'asl', 'focal', 'aploss']:
                if args.num_classes == 2:
                    bag_logits.extend(torch.sigmoid(test_logits)[:,1].cpu().numpy())
                else:
                    bag_logits.extend(torch.sigmoid(test_logits).cpu().numpy())
            loss_cls_meter.update(test_loss, 1)
    
    class_labels = args.class_labels
    bag_onehot_labels = [label.cpu() for label in bag_onehot_labels]
    
    evaluation_func = multi_class_scores_mtl
    if args.loss in ['ce'] or not args.multi_label:
        evaluation_func = multi_class_scores_mtl_ce
        
    roc_auc, accuracies, recalls, precisions, fscores, thresholds, cancer_matrix, microbial_matrix = evaluation_func(bag_onehot_labels, bag_logits, class_labels, wsi_names, threshold=args.threshold)
    output_excel_path = os.path.join(args.model_path, 'metrics.xlsx')
    save_metrics_to_excel(roc_auc, accuracies, recalls, precisions, fscores, thresholds, cancer_matrix, microbial_matrix, class_labels, output_excel_path)
    
    if args.save_logits:
        output_logits_path = os.path.join(args.model_path, 'logits.csv')
        save_logits(bag_onehot_labels, bag_logits, class_labels, wsi_names, output_logits_path)
    
    if args.loss in ['ce']:
        loss_cls_meter = loss_cls_meter.avg
    else:
        loss_cls_meter = loss_cls_meter.sum
        
    return roc_auc, accuracies, recalls, precisions, fscores, loss_cls_meter

def broadcast_state_dict(state_dict, device, rank):
    """广播状态字典到所有进程"""
    if rank == 0:
        # 主进程发送状态字典
        objects = [state_dict]
        dist.broadcast_object_list(objects, src=0)
        return state_dict
    else:
        # 其他进程接收状态字典
        objects = [None]
        dist.broadcast_object_list(objects, src=0)
        return objects[0]

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MIL Training Script')

    # Dataset 
    parser.add_argument('--datasets', default='gc_v15', type=str, help='[ngc, gc_2625, gc_v15, gc_10k, gc_fine]')
    parser.add_argument('--dataset_root', default='/data/wsi/TCTGC50k-features/gigapath-coarse', type=str, help='Dataset root path')
    parser.add_argument('--fix_loader_random', action='store_true', help='Fix random seed of dataloader')
    parser.add_argument('--persistence', action='store_true', help='Load data into memory') 
    parser.add_argument('--num_classes', default=9, type=int, help='Number of classes 9')
    
    # Augment
    parser.add_argument('--patch_drop', default=1, type=float, help='if use patch_drop')
    parser.add_argument('--patch_pad', default=1, type=float, help='if use patch_padding')
    
    # Dataset aug    
    parser.add_argument('--imbalance_sampler', default=0, type=float, help='if use imbalance_sampler')
    parser.add_argument('--fine_concat', default=0, type=int, help='flatten the fine feature')

    # Train
    parser.add_argument('--num_epoch', default=100, type=int, help='Number of total training epochs [200]')
    parser.add_argument('--batch_size', default=1, type=int, help='Number of batch size')
    
    # Loss
    parser.add_argument('--loss', default='bce', type=str, help='Classification Loss [ce, bce, asl, softbce, ranking, aploss, focal]')
    parser.add_argument('--loss_drop_weight', default=1., type=float)
    parser.add_argument('--gamma_neg', default=4.0, type=float)
    parser.add_argument('--gamma_pos', default=1.0, type=float)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.25, type=float)
    parser.add_argument('--neg_weight', default=0.0, type=float, help='Weight for positive sample in SoftBCE')
    parser.add_argument('--neg_margin', default=0, type=float, help='if use neg_margin in ranking loss')
    parser.add_argument('--opt', default='adam', type=str, help='Optimizer [adam, adamw]')
    parser.add_argument('--save_best_model_stage', default=0., type=float, help='See DTFD')
    parser.add_argument('--seed', default=2024, type=int, help='random number [2021]' )
    parser.add_argument('--lr', default=2e-3, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--lr_sche', default='cosine', type=str, help='Deacy of learning rate [cosine, step, const]')
    parser.add_argument('--lr_supi', action='store_true', help='LR scheduler update per iter')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--accumulation_steps', default=1, type=int, help='Gradient accumulate')
    parser.add_argument('--clip_grad', default=.0, type=float, help='Gradient clip')
    parser.add_argument('--multi_label', default=1, type=int, help='if use multi label by ranking')

    # Model
    # mil meathod
    parser.add_argument('--mil_method', default='abmil', type=str, help='Model name [abmil, transmil, dsmil, clam, linear, tma]')
    parser.add_argument('--input_dim', default=1536, type=int, help='The dimention of patch feature')
    parser.add_argument('--act', default='relu', type=str, help='Activation func in the projection head [gelu,relu]')
    parser.add_argument('--dropout', default=0.25, type=float, help='Dropout in the projection head')
    parser.add_argument('--da_act', default='relu', type=str, help='Activation func in the DAttention [gelu,relu]')

    # Misc
    parser.add_argument('--output_path', type=str, default='./output-model', help='Output path')
    parser.add_argument('--project', default='gcv15', type=str, help='Project name of exp')
    parser.add_argument('--title', default='gigapath-abmil-0328', type=str, help='Title of exp')
    parser.add_argument('--log_iter', default=100, type=int, help='Log Frequency')
    parser.add_argument('--amp', action='store_true', help='Automatic Mixed Precision Training')
    parser.add_argument('--wandb', action='store_true', help='Weight&Bias')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers in the dataloader')
    parser.add_argument('--save_epoch', default=25, type=int, help='epoch number to save model')
    parser.add_argument('--save_logits', default=1, type=int, help='if save logits in eval loop')
    parser.add_argument('--no_log', action='store_true', help='Without log')
    parser.add_argument('--task_config', type=str, default='./configs/oh_5.yaml', help='Task config path')
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate')
    parser.add_argument('--keep_psize_collate', type=float, default=0, help='use collate to keep patch size')
    parser.add_argument('--threshold', default=0.5, type=float, help='threshold for evaluation')
    
    # Ablation for augmentation k-means
    parser.add_argument('--kmeans-k', default=5, type=int, help='k for k-means')
    parser.add_argument('--kmeans-ratio', default=0.3, type=float, help='iter for k-means')
    parser.add_argument('--kmeans-min', default=20, type=int, help='ratio for k-means')
    
    parser.add_argument('--pretrain', default=0., type=float)
    parser.add_argument('--frozen', default=0., type=float, help='if frozen the mil network and train linear layer only')
    parser.add_argument('--pretrain_model_path', default='/home/huangjialong/projects/BiomedCLIP-PUNCE/MIL/output-model/gc_10k/gigapath-abmil-bce-drop0-50e/epoch_49_model.pt', type=str)
    
    parser.add_argument('--train_ratio', default=1, type=float, help='ratio of training set')
    
    # 新增分布式训练相关参数
    parser.add_argument('--world_size', type=int, default=1, help='World Size for distributed training')
    
    args = parser.parse_args()
    
    # 原有的数据集配置逻辑...
    if args.datasets == 'gc_v15':
        args.train_label_path = '/data/wsi/TCTGC50k-labels/6_labels/TCTGC50k-v15-train.csv'
        args.test_label_path = '/data/wsi/TCTGC50k-labels/6_labels/TCTGC50k-v15-test.csv'
        args.dataset_root = '/data/wsi/TCTGC50k-features/gigapath-coarse'
        args.num_classes = 6
        args.train_cluster_path = f'../datatools/TCTGC50k/cluster/kmeans_{args.kmeans_k}.csv'
        args.class_labels = ['nilm', 'ascus', 'asch', 'lsil', 'hsil', 'agc']
    elif args.datasets == 'gc_10k':
        args.train_label_path = '/data/wsi/TCTGC10k-labels/6_labels/TCTGC10k-v15-train.csv'
        args.test_label_path = '/data/wsi/TCTGC10k-labels/6_labels/TCTGC10k-v15-test.csv'
        args.dataset_root = '/data/wsi/TCTGC50k-features/gigapath-coarse'
        args.train_cluster_path = f'../datatools/TCTGC10k/cluster/kmeans_{args.kmeans_k}.csv'
        args.num_classes = 6
        args.class_labels = ['nilm', 'ascus', 'asch', 'lsil', 'hsil', 'agc']
    else:
        assert f'{args.datasets} is not supported'
    
    if not os.path.exists(os.path.join(args.output_path,args.project)):
        os.mkdir(os.path.join(args.output_path,args.project))
    args.model_path = os.path.join(args.output_path,args.project,args.title)
    args.log_file = os.path.join(args.model_path, 'log.txt')
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    args.fix_loader_random = True
    
    return args

if __name__ == '__main__':
    main()