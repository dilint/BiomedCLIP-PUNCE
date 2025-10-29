import time, os
from utils import print_and_log
from timm.utils import AverageMeter, dispatch_clip_grad
from timm.models import model_parameters
from utils import (
    print_and_log, 
    evaluation_cancer_sigmoid,
    evaluation_cancer_softmax,
    save_logits
)
import torch
from torch.nn.functional import one_hot
import torch.distributed as dist

def abmil_training_loop(args, model, loader, optimizer, device, amp_autocast, criterions, loss_weights, scheduler, epoch, rank):
    """單個 epoch 的訓練迴圈，支援多個損失函數"""
    start_time = time.time()
    
    # 為每種損失和總損失建立 AverageMeter
    loss_meters = {name: AverageMeter() for name in list(criterions.keys()) + ['total']}
    
    model.train()
    
    if epoch == 0 and rank == 0:
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in model.parameters())
        print_and_log(f'Total parameters: {num_total_param}, Tunable parameters: {n_parameters}', args.log_file)
    
    # === MODIFICATION START: Dataloader 返回的是 list ===
    # for i, (bag, label, file_path) in enumerate(loader):
    for i, (bag_list, label_list, file_path_list) in enumerate(loader):
        
        # 累加器，用於收集 batch 中的所有樣本
        batch_logits_list = []
        batch_labels_list = []
        
        # 獲取 dataloader 傳來的實際 batch size (list 的長度)
        current_batch_size = len(bag_list)
        if current_batch_size == 0:
            print_and_log(f"Warning: Skipping empty batch {i}", args.log_file, args.no_log)
            continue
            
        with amp_autocast():
            # 遍歷 list 中的每個樣本
            for bag, label, file_path in zip(bag_list, label_list, file_path_list):
                
                # --- 這是錯誤 'AttributeError: 'list' object has no attribute 'to'' 的修復 ---
                # 現在 bag 和 label 是 Tensors, 我們可以 .to(device)
                bag = bag.to(device)
                label = torch.tensor([label[0]]).to(device)
                if epoch < 3 and i < 3 and rank == 0:
                    # 只打印一次日誌 (此 batch 的第一個樣本)
                    if file_path == file_path_list[0]: 
                        print_and_log(f"Batch {i} Sample Files: {list(map(lambda x: os.path.basename(x), file_path_list))}", args.log_file, args.no_log)
                        print_and_log(f"First Sample Shape: {bag.shape}", args.log_file, args.no_log)
                
                # Forward pass (單個樣本)
                train_logits_sample = model(bag)
                
                # 收集結果
                batch_logits_list.append(train_logits_sample)
                batch_labels_list.append(label)

            # 將收集的樣本組合成一個 batch Tensors
            train_logits_batch = torch.cat(batch_logits_list, dim=0)
            labels_batch = torch.cat(batch_labels_list, dim=0)
            
            # 分別計算各個損失 (在整個 batch Tensors 上)
            losses = {}
            losses['cls'] = criterions['cls'](train_logits_batch, labels_batch)

            # 根據權重加總所有損失
            total_loss = sum(losses[name] * loss_weights[name] for name in losses)
        # === MODIFICATION END ===
            
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
        # 更新每種損失的 AverageMeter
        # *** 使用 'current_batch_size' (list 的長度) 而不是 'bag.size(0)' ***
        for name, loss in losses.items():
            loss_meters[name].update(loss.item(), current_batch_size)
        loss_meters['total'].update(total_loss.item(), current_batch_size)

        if (i % args.log_iter == 0 or i == len(loader) - 1) and rank == 0:
            lr = optimizer.param_groups[0]['lr']
            loss_str = ', '.join([f'{name}: {meter.avg:.4f}' for name, meter in loss_meters.items()])
            print_and_log(f'[{i}/{len(loader)-1}] {loss_str}, lr: {lr:.5f}', args.log_file, args.no_log)

    end_time = time.time()
    
    # 彙總所有進程的平均損失，用於 epoch 結束時的日誌打印
    avg_losses_summary = {}
    for name, meter in loss_meters.items():
        avg_loss = torch.tensor(meter.avg, device=device)
        if args.world_size > 1:
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
        avg_losses_summary[name] = avg_loss.item()
    
    if not args.lr_supi and scheduler:
        scheduler.step()
        
    return avg_losses_summary, start_time, end_time

# === MODIFICATION START: 'validation_loop' 函數已按照 HMIL 範例的風格修改 ===

def abmil_validation_loop(args, model, loader, device, criterions, val_set_name):
    """
    評估迴圈，只在主進程 (rank 0) 執行。
    - 採用 hmil_validation_loop 的文件命名 (使用 val_set_name)
    - 採用 hmil_validation_loop 的返回值 (返回主要指標)
    - 保留此模型原有的 (非分層) 損失和評估邏輯
    """

    model.eval()
    loss_meter = AverageMeter()
    all_logits_tensors, all_labels_tensors, all_wsi_names = [], [], []
    with torch.no_grad():
        for bag, label, file_path in loader:
            bag, label = bag.to(device), label[:,1].to(device)
            batch_size = bag.size(0)
            logits = model(bag)
            loss = criterions['cls'](logits, label)
            
            loss_meter.update(loss.item(), batch_size)
            
            # 收集 Tensors (在 CPU 上)
            all_labels_tensors.append(label.cpu())
            all_wsi_names.extend([os.path.basename(p) for p in file_path])
            
            if args.loss in ['ce']:
                probabilities = torch.softmax(logits, dim=-1)
            else: # bce, asl, etc.
                probabilities = torch.sigmoid(logits)
            all_logits_tensors.append(probabilities.cpu())

    # --- 迴圈結束 ---

    if not all_wsi_names:
        print_and_log(f"Validation Error: No data processed in {val_set_name}. Skipping metrics.", args.log_file, args.no_log)
        return None

    all_labels_tensor = torch.cat(all_labels_tensors, dim=0)
    all_logits_tensor = torch.cat(all_logits_tensors, dim=0)

        
    all_labels_np = all_labels_tensor.cpu().numpy()
    all_logits_np = all_logits_tensor.cpu().numpy()

    suffix = val_set_name
    output_excel_path = os.path.join(args.model_path, f'metrics{suffix}.xlsx')
    output_logits_path = os.path.join(args.model_path, f'logits{suffix}.csv')

    eval_func = evaluation_cancer_sigmoid if (args.loss not in ['ce'] and args.multi_label) else evaluation_cancer_softmax
    
    main_metric = None
    main_metric = eval_func(all_labels_np, all_logits_np, args.class_labels, output_excel_path)
    
    if args.save_logits:
        save_logits(all_labels_np, all_logits_np, args.class_labels, all_wsi_names, output_logits_path)

    print_and_log(f'Validation Loss ({val_set_name}): {loss_meter.avg:.4f}', args.log_file, args.no_log)
    
    return main_metric
