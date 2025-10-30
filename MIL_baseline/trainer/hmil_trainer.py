import torch
import torch.nn.functional as F
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import roc_auc_score
# from tqdm import tqdm # Removed
# import logging # Removed

from loss import SupConLoss
import time, os
from timm.utils import AverageMeter, dispatch_clip_grad
from utils import print_and_log # 假設 print_and_log 從 utils 匯入
from timm.models import model_parameters
from utils import (
    print_and_log, 
    save_logits,
    evaluation_cancer,
    get_preds_from_softmax_logits,
    gen_mapping_dict
)
from torch.nn.functional import one_hot

def compute_hierarchical_loss(logits_coarse: torch.Tensor,
                            logits_fine: torch.Tensor,
                            labels_coarse: torch.Tensor,
                            labels_fine: torch.Tensor,
                            cfg: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute hierarchical classification losses.
    
    This function computes three types of losses:
    1. Semantic loss for coarse classification
    2. Semantic loss for fine classification
    3. Regularization loss for hierarchical consistency
    
    Args:
        logits_coarse: Coarse classification logits
        logits_fine: Fine classification logits
        labels_coarse: Coarse classification labels
        labels_fine: Fine classification labels
        cfg: Configuration object
        
    Returns:
        Tuple of (semantic_loss_coarse, semantic_loss_fine, regularization_loss)
    """
    # Compute semantic losses
    loss_semantic_coarse = torch.nn.CrossEntropyLoss(reduction='mean')(logits_coarse, labels_coarse)
    loss_semantic_fine = torch.nn.CrossEntropyLoss(reduction='mean')(logits_fine, labels_fine)
    
    # Compute hierarchical regularization loss
    map_dict = gen_mapping_dict(cfg)
    preds_fine = F.softmax(logits_fine, dim=1)
    preds_fine_to_coarse = torch.zeros(preds_fine.shape[0], cfg.num_classes[0]).cuda()
    
    for fine_class, coarse_class in map_dict.items():
        preds_fine_to_coarse[:, coarse_class] += preds_fine[:, fine_class]
    
    loss_regularization = torch.nn.CrossEntropyLoss(reduction='mean')(preds_fine_to_coarse, labels_coarse)
    
    return loss_semantic_coarse, loss_semantic_fine, loss_regularization


def compute_attention_matching_loss(batch_attention_coarse: List[torch.Tensor],
                                  batch_attention_fine: List[torch.Tensor],
                                  cfg: Any) -> torch.Tensor:
    """
    Compute hierarchical attention matching loss.
    
    This loss ensures that the attention patterns at fine and coarse levels
    are consistent with the hierarchical structure.
    
    Args:
        batch_attention_coarse: List of coarse attention maps
        batch_attention_fine: List of fine attention maps
        cfg: Configuration object
        
    Returns:
        Attention matching loss
    """
    loss_sim = 0.
    map_dict = gen_mapping_dict(cfg)
    for attention_coarse, attention_fine in zip(batch_attention_coarse, batch_attention_fine):
        attention_fine_to_coarse = torch.zeros(cfg.num_classes[0], attention_fine.shape[1]).cuda()
        for fine_class, coarse_class in map_dict.items():
            attention_fine_to_coarse[coarse_class, :] += attention_fine[fine_class, :]
        loss_sim += torch.sum((1 - torch.nn.CosineSimilarity(dim=0)(attention_coarse, attention_fine_to_coarse)))/attention_coarse.shape[1]
    
    return loss_sim / len(batch_attention_coarse)


def compute_temperature(epoch: int, num_epochs: int) -> float:
    """
    Compute dynamic temperature for contrastive loss.
    
    Args:
        epoch: Current epoch number
        num_epochs: Total number of epochs
        
    Returns:
        Temperature value
    """
    temp_low, temp_high = 0.1, 1.0
    return (temp_high - temp_low) * (1 + math.cos(2 * math.pi * epoch / num_epochs)) / 2 + temp_low


def compute_contrastive_loss(semantics_features: torch.Tensor,
                           patient_fine_labels: torch.Tensor,
                           epoch: int,
                           cfg: Any) -> Tuple[torch.Tensor, float]:
    """
    Compute contrastive loss with dynamic temperature.
    
    Args:
        semantics_features: Feature vectors for contrastive learning
        patient_fine_labels: Fine-grained labels for supervision
        epoch: Current epoch number
        cfg: Configuration object
        
    Returns:
        Tuple of (contrastive_loss, temperature)
    """
    alpha = max(0.5, 1 - (epoch / cfg.num_epoch)**2)
    tau = compute_temperature(epoch, cfg.num_epoch)
    
    contrastive_loss = SupConLoss(temperature=tau).cuda()
    return contrastive_loss(semantics_features, patient_fine_labels) / len(semantics_features), tau


def hmil_training_loop(args, model, loader, optimizer, device, amp_autocast, scheduler, epoch, rank):
    """
    單個 epoch 的訓練迴圈，整合了 HMIL (train_phase) 的訓練邏輯。
    使用 AverageMeter (來自框架) 替換 TrainingMetrics (來自 train_phase)。
    """
    start_time = time.time()
    
    # 為每種 HMIL 損失和總損失建立 AverageMeter (來自框架存根)
    loss_names = ['sem_coarse', 'sem_fine', 'reg', 'attn', 'contrast', 'total']
    loss_meters = {name: AverageMeter() for name in loss_names}
    
    model.train()
    
    if epoch == 0 and rank == 0:
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in model.parameters())
        print_and_log(f'Total parameters: {num_total_param}, Tunable parameters: {n_parameters}', args.log_file, args.no_log)
    
    # --- 開始 'train_phase' 邏輯 ---
    try:
        for iter_idx, (cell_images, patient_labels, _) in enumerate(loader):
            batch_attention_coarse = []
            batch_attention_fine = []
            batch_semantics = []
            
            batch_logits_coarse = []
            batch_logits_fine = []
            batch_labels_coarse = []
            batch_labels_fine = []
            
            # 獲取 dataloader 的實際 batch size
            current_batch_size = len(cell_images)
            
            with amp_autocast():
                # Process each sample in the batch (來自 train_phase 的內部迴圈)
                for cell_image, patient_label in zip(cell_images, patient_labels):
                    cell_image = cell_image.to(device)
                    # 標籤來自 Dataloader，應為 (coarse, fine)
                    label_coarse = torch.tensor(patient_label[0]).to(device)
                    label_fine = torch.tensor(patient_label[1]).to(device)
                    
                    # Forward pass
                    attention_maps, logits, features = model(cell_image)
                    batch_semantics.append([features, label_fine])
                    
                    # Process outputs
                    logits_coarse = logits[0].squeeze()
                    logits_fine = logits[1].squeeze()
                    
                    batch_logits_coarse.append(logits_coarse.unsqueeze(0))
                    batch_logits_fine.append(logits_fine.unsqueeze(0))
                    batch_labels_coarse.append(label_coarse.unsqueeze(0))
                    batch_labels_fine.append(label_fine.unsqueeze(0))
                    
                    batch_attention_coarse.append(attention_maps[0])
                    batch_attention_fine.append(attention_maps[1])
                    
                    # `metrics.update_batch_metrics` 被移除，因為我們只關心損失
                
                # Compute losses (在 amp_autocast 內)
                batch_logits_coarse = torch.cat(batch_logits_coarse, dim=0)
                batch_logits_fine = torch.cat(batch_logits_fine, dim=0)
                batch_labels_coarse = torch.cat(batch_labels_coarse, dim=0)
                batch_labels_fine = torch.cat(batch_labels_fine, dim=0)
                
                # 'cfg' 在這個框架中是 'args'
                loss_semantic_coarse, loss_semantic_fine, loss_regularization = compute_hierarchical_loss(
                    batch_logits_coarse, batch_logits_fine, batch_labels_coarse, batch_labels_fine, args
                )
                
                loss_attention = compute_attention_matching_loss(batch_attention_coarse, batch_attention_fine, args)
                
                # Compute contrastive loss
                semantics_features = torch.stack([sem[0] for sem in batch_semantics])
                patient_fine_labels = torch.stack([sem[1] for sem in batch_semantics])
                loss_contrastive, tau = compute_contrastive_loss(semantics_features, patient_fine_labels, epoch, args)
                
                # Combine losses
                alpha = 1 - (epoch / args.num_epoch)**2 # cfg.num_epochs -> args.num_epoch
                loss_classification = loss_semantic_fine + loss_regularization
                # 'loss' 是總損失
                loss = alpha * loss_semantic_coarse + loss_classification + (1 - alpha) * loss_contrastive
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad > 0:
                dispatch_clip_grad(model_parameters(model), args.clip_grad)
            optimizer.step()

            # Scheduler step (if per-iteration, 根據框架邏輯)
            if scheduler is not None:
                if args.lr_sche == 'cycle' or (args.lr_sche == 'cosine' and args.lr_supi):
                    scheduler.step()
            
            # --- 這是 "修改打印樣式" 的核心 ---
            # 更新 AverageMeters，使用 dataloader 的 'current_batch_size'
            loss_meters['sem_coarse'].update(loss_semantic_coarse.item(), current_batch_size)
            loss_meters['sem_fine'].update(loss_semantic_fine.item(), current_batch_size)
            loss_meters['reg'].update(loss_regularization.item(), current_batch_size)
            loss_meters['attn'].update(loss_attention.item(), current_batch_size)
            loss_meters['contrast'].update(loss_contrastive.item(), current_batch_size)
            loss_meters['total'].update(loss.item(), current_batch_size)
            
            if (iter_idx % args.log_iter == 0 or iter_idx == len(loader) - 1) and rank == 0:
                lr = optimizer.param_groups[0]['lr']
                loss_str = ', '.join([f'{name}: {meter.avg:.4f}' for name, meter in loss_meters.items()])
                print_and_log(f'[{iter_idx}/{len(loader)-1}] {loss_str}, lr: {lr:.5f}', args.log_file, args.no_log)
    except Exception as e:
        print_and_log(f"Error in training loop (epoch {epoch}, iter {iter_idx}): {e}", args.log_file, args.no_log)
        raise e
    # --- 'train_phase' 邏輯結束 ---
    
    # Scheduler step (if per-epoch, 根據框架邏輯)
    if scheduler is not None:
        # 處理 'cosine' (without lr_supi) 和 'step'
        if args.lr_sche in ['cosine', 'step'] and not args.lr_supi:
            scheduler.step()
    end_time = time.time()
    
    # 準備返回給 'run_training_process' 的字典
    avg_losses_summary = {name: meter.avg for name, meter in loss_meters.items()}
            
    return avg_losses_summary, start_time, end_time

def hmil_validation_loop(args, model, loader, device, criterions, val_set_name):
    model.eval()
    
    loss_names = ['sem_coarse', 'sem_fine', 'reg', 'attn', 'contrast', 'total']
    loss_meters = {name: AverageMeter() for name in loss_names}
    
    all_logits_coarse, all_labels_coarse = [], []
    all_logits_fine, all_labels_fine = [], []
    all_wsi_names = []

    epoch = args.num_epoch 
    with torch.no_grad():
        for iter_idx, (cell_image, label_tuple, file_path_list) in enumerate(loader):

            cell_image = cell_image.squeeze(0).to(device) 
            label_tuple = label_tuple.squeeze(0)
            label_coarse = label_tuple[0].to(device)
            label_fine = label_tuple[1].to(device) 
            
            if label_coarse.dim() > 0: 
                label_coarse = label_coarse.squeeze(0)
            if label_fine.dim() > 0:
                label_fine = label_fine.squeeze(0)
            file_path = file_path_list[0]

            attention_maps, logits, features = model(cell_image)
            
            logits_coarse = logits[0].squeeze()
            logits_fine = logits[1].squeeze()

            batch_logits_coarse = logits_coarse.unsqueeze(0)
            batch_logits_fine = logits_fine.unsqueeze(0)
            batch_labels_coarse = label_coarse.unsqueeze(0)
            batch_labels_fine = label_fine.unsqueeze(0)
            
            loss_semantic_coarse, loss_semantic_fine, loss_regularization = compute_hierarchical_loss(
                batch_logits_coarse, batch_logits_fine, batch_labels_coarse, batch_labels_fine, args
            )
            
            loss_attention = compute_attention_matching_loss(
                [attention_maps[0]], [attention_maps[1]], args
            )
            
            semantics_features = features.unsqueeze(0)
            patient_fine_labels = label_fine.unsqueeze(0)
            loss_contrastive, tau = compute_contrastive_loss(
                semantics_features, patient_fine_labels, epoch, args
            )
            
            alpha = 1 - (epoch / args.num_epoch)**2
            loss_classification = loss_semantic_fine + loss_regularization
            loss = alpha * loss_semantic_coarse + loss_classification + (1 - alpha) * loss_contrastive

            loss_meters['sem_coarse'].update(loss_semantic_coarse.item(), 1)
            loss_meters['sem_fine'].update(loss_semantic_fine.item(), 1)
            loss_meters['reg'].update(loss_regularization.item(), 1)
            loss_meters['attn'].update(loss_attention.item(), 1)
            loss_meters['contrast'].update(loss_contrastive.item(), 1)
            loss_meters['total'].update(loss.item(), 1)
            
            all_logits_coarse.append(logits_coarse.cpu())
            all_labels_coarse.append(label_coarse.cpu())
            all_logits_fine.append(logits_fine.cpu())
            all_labels_fine.append(label_fine.cpu())
            all_wsi_names.append(os.path.basename(file_path))

    if not all_wsi_names:
        print_and_log("Validation Error: No data processed. Skipping metrics.", args.log_file, args.no_log)
        return

    all_logits_coarse_tensor = torch.stack(all_logits_coarse, dim=0)
    all_labels_coarse_tensor = torch.stack(all_labels_coarse, dim=0)
    all_logits_fine_tensor = torch.stack(all_logits_fine, dim=0)
    all_labels_fine_tensor = torch.stack(all_labels_fine, dim=0)
    
    loss_str = ', '.join([f'Val_{name}: {meter.avg:.4f}' for name, meter in loss_meters.items()])
    print_and_log(f'Validation Losses: {loss_str}', args.log_file, args.no_log)

    suffix = val_set_name
    
    all_labels_coarse_np = all_labels_coarse_tensor.cpu().numpy()
    all_labels_fine_np = all_labels_fine_tensor.cpu().numpy()
    
    probs_coarse_np = torch.softmax(all_logits_coarse_tensor, dim=-1).cpu().numpy()
    probs_fine_np = torch.softmax(all_logits_fine_tensor, dim=-1).cpu().numpy()
    
    print_and_log(f'--- Evaluating Coarse Metrics (Binary) ---', args.log_file, args.no_log)
    output_excel_path_coarse = os.path.join(args.model_path, f'metrics_coarse{suffix}.xlsx')
    num_classes_coarse = args.num_classes[0]
    class_labels_coarse = args.coarse_class_labels
    
    coarse_pred_labels, coarse_roc_auc_list, coarse_thresholds = \
        get_preds_from_softmax_logits(
            all_labels_coarse_np,
            probs_coarse_np,
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
        labels_coarse_onehot_np = one_hot(all_labels_coarse_tensor, num_classes=num_classes_coarse).cpu().numpy()
        save_logits(labels_coarse_onehot_np, probs_coarse_np, class_labels_coarse, all_wsi_names, output_logits_path_coarse)

    print_and_log(f'--- Evaluating Fine Metrics (Multi-class) ---', args.log_file, args.no_log)
    output_excel_path_fine = os.path.join(args.model_path, f'metrics_fine{suffix}.xlsx')
    output_logits_path_fine = os.path.join(args.model_path, f'logits_fine{suffix}.csv')
    num_classes_fine = args.num_classes[1]
    class_labels_fine = args.class_labels
    
    fine_pred_labels, fine_roc_auc_list, fine_thresholds = \
        get_preds_from_softmax_logits(
            all_labels_fine_np,
            probs_fine_np,
            num_classes_fine
        )

    evaluation_cancer(
        bag_labels=all_labels_fine_np,
        pred_labels=fine_pred_labels,
        class_labels=class_labels_fine,
        output_path=output_excel_path_fine,
        roc_auc_list=fine_roc_auc_list,
        thresholds_list=fine_thresholds
    )
    
    if args.save_logits:
        labels_fine_onehot_np = one_hot(all_labels_fine_tensor, num_classes=num_classes_fine).cpu().numpy()
        save_logits(labels_fine_onehot_np, probs_fine_np, class_labels_fine, all_wsi_names, output_logits_path_fine)

    return auc_coarse