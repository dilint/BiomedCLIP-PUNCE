import time, os
from utils import print_and_log
from timm.utils import AverageMeter, dispatch_clip_grad
from timm.models import model_parameters
from utils import (
    print_and_log, 
    evaluation_cancer_sigmoid,
    evaluation_cancer_softmax,
    save_logits,
    get_preds_from_sigmoid_logits,
    get_preds_from_softmax_logits,
    parse_mapping,
    evaluation_cancer,
)
from torch.nn.functional import one_hot
import torch
import torch.distributed as dist
import numpy as np

def abmil_training_loop(args, model, loader, optimizer, device, amp_autocast, criterions, loss_weights, scheduler, epoch, rank):
    """
    [已修改]
    单个 epoch 的训练循环，支持多个损失函数 (coarse & fine)。
    """
    start_time = time.time()
    
    loss_meters = {name: AverageMeter() for name in list(criterions.keys()) + ['total']}
    
    model.train()
    
    if epoch == 0 and rank == 0:
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in model.parameters())
        print_and_log(f'Total parameters: {num_total_param}, Tunable parameters: {n_parameters}', args.log_file)
    
    for i, (bag_list, label_list, file_path_list) in enumerate(loader):
        
        batch_logits_coarse_list = []
        batch_logits_fine_list = []
        batch_labels_coarse_list = []
        batch_labels_fine_list = []
        
        current_batch_size = len(bag_list)
        if current_batch_size == 0:
            print_and_log(f"Warning: Skipping empty batch {i}", args.log_file, args.no_log)
            continue
            
        with amp_autocast():
            for bag, label, file_path in zip(bag_list, label_list, file_path_list):
                
                bag = bag.unsqueeze(0).to(device)
                
                label_coarse = torch.tensor([label[0]]).to(device)
                label_fine = torch.tensor([label[1]]).to(device)

                if epoch < 3 and i < 3 and rank == 0:
                    if file_path == file_path_list[0]: 
                        print_and_log(f"Batch {i} Sample Files: {list(map(lambda x: os.path.basename(x), file_path_list))}", args.log_file, args.no_log)
                        print_and_log(f"First Sample Shape: {bag.shape}", args.log_file, args.no_log)
                
                logits_coarse_sample, logits_fine_sample = model(bag)
                
                batch_logits_coarse_list.append(logits_coarse_sample)
                batch_logits_fine_list.append(logits_fine_sample)
                batch_labels_coarse_list.append(label_coarse)
                batch_labels_fine_list.append(label_fine)

            logits_coarse_batch = torch.cat(batch_logits_coarse_list, dim=0)
            logits_fine_batch = torch.cat(batch_logits_fine_list, dim=0)
            labels_coarse_batch = torch.cat(batch_labels_coarse_list, dim=0)
            labels_fine_batch = torch.cat(batch_labels_fine_list, dim=0)
            
            losses = {}
            losses['coarse_cls'] = criterions['coarse_cls'](logits_coarse_batch, labels_coarse_batch, args.num_classes[0])
            losses['fine_cls'] = criterions['fine_cls'](logits_fine_batch, labels_fine_batch, args.num_classes[1])
            
            total_loss = sum(losses[name] * loss_weights[name] for name in losses)
            
        total_loss_scaled = total_loss / args.accumulation_steps
        total_loss_scaled.backward()
        
        if (i + 1) % args.accumulation_steps == 0:
            if args.clip_grad > 0.:
                dispatch_clip_grad(model_parameters(model), value=args.clip_grad, mode='norm')
            optimizer.step()
            optimizer.zero_grad()
            if args.lr_supi and scheduler:
                scheduler.step()

        for name, loss in losses.items():
            loss_meters[name].update(loss.item(), current_batch_size)
        loss_meters['total'].update(total_loss.item(), current_batch_size)

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


def abmil_validation_loop(args, model, loader, device, criterions, val_set_name):

    model.eval()
    
    loss_names = list(criterions.keys()) + ['total']
    loss_meters = {name: AverageMeter() for name in loss_names}
    
    all_logits_coarse_tensors, all_labels_coarse_tensors = [], []
    all_logits_fine_tensors, all_labels_fine_tensors = [], []
    all_wsi_names = []
    
    with torch.no_grad():
        for bag, label_tuple, file_path in loader:
            bag = bag.to(device)
            
            label_coarse = label_tuple[:, 0].to(device)
            label_fine = label_tuple[:, 1].to(device)

            batch_size = bag.size(0)
            logits_coarse, logits_fine = model(bag)
            
            losses = {}
            if 'coarse_cls' in criterions:
                losses['coarse_cls'] = criterions['coarse_cls'](logits_coarse, label_coarse, args.num_classes[0])
            if 'fine_cls' in criterions:
                losses['fine_cls'] = criterions['fine_cls'](logits_fine, label_fine, args.num_classes[1])
            
            total_loss = sum(losses.get(name, 0.0) for name in loss_names if name != 'total')
            
            for name, loss in losses.items():
                loss_meters[name].update(loss.item(), batch_size)
            loss_meters['total'].update(total_loss.item(), batch_size)
            
            all_labels_coarse_tensors.append(label_coarse.cpu())
            all_labels_fine_tensors.append(label_fine.cpu())
            all_wsi_names.extend([os.path.basename(p) for p in file_path])
            
            if args.loss in ['ce']: 
                probabilities_coarse = torch.softmax(logits_coarse, dim=-1)
                probabilities_fine = torch.softmax(logits_fine, dim=-1)
            else: 
                probabilities_coarse = torch.sigmoid(logits_coarse)
                probabilities_fine = torch.sigmoid(logits_fine)
            
            all_logits_coarse_tensors.append(probabilities_coarse.cpu())
            all_logits_fine_tensors.append(probabilities_fine.cpu())

    if not all_wsi_names:
        print_and_log(f"Validation Error: No data processed in {val_set_name}. Skipping metrics.", args.log_file, args.no_log)
        return None

    all_labels_coarse_np = torch.cat(all_labels_coarse_tensors, dim=0).cpu().numpy()
    all_logits_coarse_np = torch.cat(all_logits_coarse_tensors, dim=0).cpu().numpy()
    all_labels_fine_np = torch.cat(all_labels_fine_tensors, dim=0).cpu().numpy()
    all_logits_fine_np = torch.cat(all_logits_fine_tensors, dim=0).cpu().numpy()
    
    num_classes_coarse = all_logits_coarse_np.shape[1]
    num_classes_fine = all_logits_fine_np.shape[1]
    
    loss_str = ', '.join([f'Val_{name}: {meter.avg:.4f}' for name, meter in loss_meters.items()])
    print_and_log(f'Validation Losses ({val_set_name}): {loss_str}', args.log_file, args.no_log)

    suffix = val_set_name
    
    if args.loss in ['ce'] or not args.multi_label:
        get_preds_coarse_func = get_preds_from_softmax_logits
    else: 
        get_preds_coarse_func = get_preds_from_sigmoid_logits

    print_and_log(f'--- Evaluating Coarse Metrics ({val_set_name}) ---', args.log_file, args.no_log)
    output_excel_path_coarse = os.path.join(args.model_path, f'metrics_coarse{suffix}.xlsx')
    class_labels_coarse = args.coarse_class_labels
    
    coarse_pred_labels, coarse_roc_auc_list, coarse_thresholds = \
        get_preds_coarse_func(
            all_labels_coarse_np,
            all_logits_coarse_np,
            num_classes_coarse
        )

    auc_coarse = evaluation_cancer(
        bag_labels=all_labels_coarse_np,
        pred_labels=coarse_pred_labels,
        class_labels=class_labels_coarse,
        output_path=output_excel_path_coarse,
        roc_auc_list=coarse_roc_auc_list,
        thresholds_list=coarse_thresholds
    )
    
    if args.save_logits:
        output_logits_path_coarse = os.path.join(args.model_path, f'logits_coarse{suffix}.csv')
        labels_coarse_onehot_np = one_hot(torch.cat(all_labels_coarse_tensors), num_classes=num_classes_coarse).cpu().numpy()
        save_logits(labels_coarse_onehot_np, all_logits_coarse_np, class_labels_coarse, all_wsi_names, output_logits_path_coarse)

    if args.loss not in ['ce'] and args.multi_label:
        get_preds_fine_func = get_preds_from_sigmoid_logits
    else:
        get_preds_fine_func = get_preds_from_softmax_logits
        
    print_and_log(f'--- Evaluating Fine Metrics ({val_set_name}) ---', args.log_file, args.no_log)
    output_excel_path_fine = os.path.join(args.model_path, f'metrics_fine{suffix}.xlsx')
    class_labels_fine = args.class_labels
    
    fine_pred_labels, fine_roc_auc_list, fine_thresholds = \
        get_preds_fine_func(
            all_labels_fine_np,
            all_logits_fine_np,
            num_classes_fine
        )

    main_metric = evaluation_cancer(
        bag_labels=all_labels_fine_np,
        pred_labels=fine_pred_labels,
        class_labels=class_labels_fine,
        output_path=output_excel_path_fine,
        roc_auc_list=fine_roc_auc_list,
        thresholds_list=fine_thresholds
    )
    if args.save_logits:
        output_logits_path_fine = os.path.join(args.model_path, f'logits_fine{suffix}.csv')
        labels_fine_onehot_np = one_hot(torch.cat(all_labels_fine_tensors), num_classes=num_classes_fine).cpu().numpy()
        save_logits(labels_fine_onehot_np, all_logits_fine_np, class_labels_fine, all_wsi_names, output_logits_path_fine)
    
    return main_metric