import time, os
from utils import print_and_log
from timm.utils import AverageMeter, dispatch_clip_grad
from timm.models import model_parameters
from utils import print_and_log
from torch.nn.functional import one_hot
import torch
import torch.distributed as dist
import numpy as np
import torch.nn.functional as F
import time
from timm.utils import AverageMeter, dispatch_clip_grad
from timm.models import model_parameters
import torch.distributed as dist
import os

def gh_training_loop(args, model, loader, optimizer, device, amp_autocast, criterions, loss_weights, scheduler, epoch, rank, target_weight, pi):
    """
    单个 epoch 的训练循环 (专为双分支 Valina_MIL 模型设计)
    """
    start_time = time.time()

    loss_meters = {name: AverageMeter()
                   for name in list(criterions.keys()) + ['total']}

    model.train()

    if epoch == 0 and rank == 0:
        n_parameters = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in model.parameters())
        print_and_log(
            f'Total parameters: {num_total_param}, Tunable parameters: {n_parameters}', args.log_file)

    for i, ((bag1_list, bag2_list), labels_batch, file_path_list) in enumerate(loader):
        
        labels_batch = labels_batch.to(device) 
        current_batch_size = len(bag1_list)
        
        if current_batch_size == 0:
            print_and_log(f"Warning: Skipping empty batch {i}", args.log_file, args.no_log)
            continue

        if epoch < 3 and i < 3 and rank == 0:
            print_and_log(f"Batch {i} Sample Files: {list(map(lambda x: os.path.basename(x), file_path_list))}",
                          args.log_file, args.no_log)
            print_and_log(f"First Bag1 shape: {bag1_list[0].shape}, First Bag2 shape: {bag2_list[0].shape}",
                          args.log_file, args.no_log)

        # 为模型输出初始化列表
        batch_logits_coarse_1_list, batch_logits_fine_1_list, batch_gh_proj_1_list = [], [], []
        batch_logits_coarse_2_list, batch_logits_fine_2_list, batch_gh_proj_2_list = [], [], []

        with amp_autocast():
            # 内层循环：逐个处理 batch 中的 bag
            for j in range(current_batch_size):
                bag1 = bag1_list[j].unsqueeze(0).to(device)
                bag2 = bag2_list[j].unsqueeze(0).to(device)
                
                logits_coarse_1, logits_fine_1, gh_proj_1 = model(bag1)
                logits_coarse_2, logits_fine_2, gh_proj_2 = model(bag2)

                # 收集输出
                batch_logits_coarse_1_list.append(logits_coarse_1)
                batch_logits_fine_1_list.append(logits_fine_1)
                batch_logits_coarse_2_list.append(logits_coarse_2)
                batch_logits_fine_2_list.append(logits_fine_2)

                if gh_proj_1 is not None:
                    batch_gh_proj_1_list.append(gh_proj_1)
                if gh_proj_2 is not None:
                    batch_gh_proj_2_list.append(gh_proj_2)

            # --- 内层循环结束 ---

            # 将输出列表拼接为批次张量
            logits_coarse_1_batch = torch.cat(batch_logits_coarse_1_list, dim=0)
            logits_fine_1_batch = torch.cat(batch_logits_fine_1_list, dim=0)
            logits_coarse_2_batch = torch.cat(batch_logits_coarse_2_list, dim=0)
            logits_fine_2_batch = torch.cat(batch_logits_fine_2_list, dim=0)

            # 从已在 device 上的 labels_batch 中获取标签
            labels_coarse_batch = labels_batch[:, 0]
            labels_fine_batch = labels_batch[:, 1]
            
            losses = {}
            if epoch >= args.warmup_epoch: 
                loss_coarse_1 = criterions['coarse_cls'](logits_coarse_1_batch, labels_coarse_batch, args.num_classes[0])
                loss_coarse_2 = criterions['coarse_cls'](logits_coarse_2_batch, labels_coarse_batch, args.num_classes[0])
                losses['coarse_cls'] = (loss_coarse_1 + loss_coarse_2) / 2.0
                
                loss_fine_1 = criterions['fine_cls'](logits_fine_1_batch, labels_fine_batch, args.num_classes[1])
                loss_fine_2 = criterions['fine_cls'](logits_fine_2_batch, labels_fine_batch, args.num_classes[1])
                losses['fine_cls'] = (loss_fine_1 + loss_fine_2) / 2.0

            if loss_weights.get('con', 0.0) > 0 and len(batch_gh_proj_1_list) > 0 and len(batch_gh_proj_2_list) > 0:
                gh_proj_1_batch = torch.cat(batch_gh_proj_1_list, dim=0)
                gh_proj_2_batch = torch.cat(batch_gh_proj_2_list, dim=0)

                t1 = F.normalize(gh_proj_1_batch, dim=1)
                t2 = F.normalize(gh_proj_2_batch, dim=1)
                
                losses['con'] = criterions['con'](t1, t2, pi)
            else:
                loss_weights['con'] = 0.0 

            total_loss = sum(losses[name] * loss_weights[name]
                             for name in losses if loss_weights.get(name, 0.0) > 0)

        total_loss_scaled = total_loss / args.accumulation_steps
        total_loss_scaled.backward()

        if (i + 1) % args.accumulation_steps == 0:
            if args.clip_grad > 0.:
                dispatch_clip_grad(model_parameters(
                    model), value=args.clip_grad, mode='norm')
            optimizer.step()
            optimizer.zero_grad()
            if args.lr_supi and scheduler:
                scheduler.step()

        for name, loss in losses.items():
            if loss_weights.get(name, 0.0) > 0: 
                loss_meters[name].update(loss.item(), current_batch_size) 
        loss_meters['total'].update(total_loss.item(), current_batch_size)

        if (i % args.log_iter == 0 or i == len(loader) - 1) and rank == 0:
            lr = optimizer.param_groups[0]['lr']
            loss_str = ', '.join(
                [f'{name}: {meter.avg:.4f}' for name, meter in loss_meters.items() if meter.count > 0])
            print_and_log(
                f'[{i}/{len(loader)-1}] {loss_str}, lr: {lr:.5f}', args.log_file, args.no_log)

    end_time = time.time()

    avg_losses_summary = {}
    for name, meter in loss_meters.items():
        if meter.count > 0: 
            avg_loss = torch.tensor(meter.avg, device=device)
            if args.world_size > 1:
                dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            avg_losses_summary[name] = avg_loss.item()

    if not args.lr_supi and scheduler:
        scheduler.step()

    return avg_losses_summary, start_time, end_time
