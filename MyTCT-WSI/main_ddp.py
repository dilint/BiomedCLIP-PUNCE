import torch
import math
import sys
import os
import argparse
import random
import datetime
import numpy as np
import time
import logging
from typing import Iterable
from pathlib import Path
import torch.distributed as dist
from models.model_v1 import build_model
from datasets.tct_wsi import build_dataset, collate_fn_wsi
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup

def setup_logger(rank, logger_file_path):
    logger = logging.getLogger(f'process_{rank}')
    logger.setLevel(logging.INFO if rank == 0 else logging.CRITICAL + 1)  # 主进程INFO级别，其他进程不记录
    logger.handlers.clear()  # 清除可能已经存在的handlers，防止重复添加

    if rank == 0:
        log_name = 'traing.log'
        final_log_file = os.path.join(logger_file_path, log_name)

        file_handler = logging.FileHandler(final_log_file)
        file_handler.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "[%(asctime)s][%(levelname)s]: %(message)s"
        )

        file_handler.setFormatter(formatter) 
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger



def get_args_parser():
    parser = argparse.ArgumentParser('My TCT WSI CLS Model', add_help=False)

    #Training
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_epoch', default=200, type=int)
    parser.add_argument('--seed', default=2025, type=int)
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count(), help='Number of GPUs to use')

    #Data
    parser.add_argument('--root', default='/root/autodl-tmp/data', type=str, help='dataset root')

    #Model
    parser.add_argument('--in_dims', default=256, type=int)
    parser.add_argument('--depth', default=1, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--proj_drop', default=0.02, type=float)
    parser.add_argument('--attn_drop', default=0.02, type=float)
    parser.add_argument('--drop_path', default=0.01, type=float)

    #loss function
    # parser.add_argument('--recon_loss_coef', default=0.7, type=float)
    # parser.add_argument('--kl_loss_coef', default=0.3, type=float)


    #output
    parser.add_argument('--output_dir', default='./exp', type=str, help='output dir')
    parser.add_argument('--tb_dir', default='/root/tf-logs/lgj', type=str, help='tesorboard dir')
    parser.add_argument('--save_iter', default=20, type=int)
    return parser


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, logger: logging.Logger):
    
    model.train()
    criterion.train()

    log_interval = 50
    train_states = {}
    for i, data in enumerate(data_loader):
        optimizer.zero_grad()

        feature = data[0].to(device) # shape[B, N, M, C]
        mask = data[1].to(device)  # shape[B, N, M]
        label = data[2].to(device)  # shape[B, num_class]

        logit = model(feature, mask)

        loss_dict = criterion(logit, label)
        losses = sum(loss_dict[k] for k in loss_dict.keys())

        loss_value = losses.item()
        loss_dict_class = {f'train_{k}': v.item() for k, v in loss_dict.items()}
        
        
        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        
        losses.backward()
        optimizer.step()
    
        train_states.update(train_loss=loss_value, **loss_dict_class)

        if (i + 1) % log_interval == 0:
            logger.info(f'Epoch[{epoch+1}]({i+1}/{len(data_loader)}): ' + ', '.join([f'{k}:{v:.4f}' for k,v in train_states.items()]))
    
    return train_states

def eval_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, data_loader: Iterable, 
                   device: torch.device, epoch: int, rank: int, world_size: int, logger: logging.Logger):
    
    model.eval()
    criterion.eval()

    local_loss = 0.0
    local_count = 0
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            
            feature = data[0].to(device) # shape[B, N, M, C]
            mask = data[1].to(device)  # shape[B, N, M]
            label = data[2].to(device)  # shape[B, num_class]

            logit = model(feature, mask)

            loss_dict = criterion(logit, label)
            local_loss += sum(loss_dict[k] for k in loss_dict.keys()).item() * feature.shape[0]
            local_count += feature.shape[0]

            all_logits.append(logit.cpu())  # 将logits移到CPU
            all_labels.append(label.cpu())  # 将labels移到CPU

    local_loss_tensor = torch.tensor(local_loss).to(device)
    local_count_tensor = torch.tensor(local_count).to(device)

    # 将每个进程的结果收集到一个列表中
    gathered_logits = [torch.zeros_like(torch.cat(all_logits, dim=0)) for _ in range(world_size)]
    gathered_labels = [torch.zeros_like(torch.cat(all_labels, dim=0)) for _ in range(world_size)]

    dist.all_reduce(local_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_count_tensor, op=dist.ReduceOp.SUM)

    # 收集所有进程的数据到主进程
    if rank == 0:
        dist.gather(torch.cat(all_logits, dim=0), gather_list=gathered_logits, dst=0)
        dist.gather(torch.cat(all_labels, dim=0), gather_list=gathered_labels, dst=0)
    else:
        dist.gather(torch.cat(all_logits, dim=0), gather_list=None, dst=0)
        dist.gather(torch.cat(all_labels, dim=0), gather_list=None, dst=0)
    
    if rank == 0:
        final_logits = torch.cat(gathered_logits, dim=0)
        final_labels = torch.cat(gathered_labels, dim=0)

        # 调用metrics函数计算评价指标
        metrics_result = compute_metrics(final_logits, final_labels)

        global_loss = local_loss_tensor.item() / local_count_tensor.item()
        
        #logger.info(f'Evaluate results at epoch{epoch+1}: ' + ', '.join([f'{k}:{v:.4f}' for k,v in metrics_result.items()]))
        return metrics_result, global_loss
    else:
        return None, None


def setup(rank, world_size):
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_loop(rank, args, logger):

    # print(args)
    setup(rank, args.world_size)
    device = torch.device(f"cuda:{rank}")

    # fix the seed for reproducibility
    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    model, criterion = build_model(args)
    model.to(device)
    criterion.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print('number of params:', n_parameters)

    num_training_steps = args.num_epoch * len(dataloader_train)
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% 的 warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr*args.world_size, weight_decay=args.weight_decay) #先将未被DPP包装的模型参数传入opt，再DDP
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
    ddp_model = DDP(model, device_ids=[rank])

    dataset_train = build_dataset(image_set='train', args=args)  
    dataset_val = build_dataset(image_set='val', args=args)

    sampler_train = DistributedSampler(dataset_train, num_replicas=args.world_size, rank=rank)
    sampler_val = DistributedSampler(dataset_val, num_replicas=args.world_size, rank=rank, shuffle=False)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=sampler_train,
        num_workers=args.num_workers,
        collate_fn=collate_fn_wsi,
        pin_memory=True,
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        sampler=sampler_val,
        num_workers=args.num_workers,
        collate_fn=collate_fn_wsi,
        drop_last=True,
        pin_memory=True,
    )

    output_dir = args.output_dir
    tb_dir = args.tb_dir
    # resume

    # end resume
    writer = SummaryWriter(log_dir=os.path.join(tb_dir, os.path.basename(output_dir)))
    print("Start training")
    start_time = time.time()

    best_val_loss = float('inf')
    save_iter = args.save_iter
    for epoch in range(args.start_epoch, args.num_epoch):
        sampler_train.set_epoch(epoch)
        train_states = train_one_epoch(
            ddp_model, criterion, dataloader_train, optimizer, device, epoch, logger)
        lr_scheduler.step()
        eval_states, eval_loss = eval_one_epoch(ddp_model, criterion, dataloader_val, device, epoch, rank, args.world_size, logger)

        if rank == 0:
            for k,v in train_states.items():
                writer.add_scalar(k, v, epoch+1)
            for k,v in eval_states.items():
                writer.add_scalar(k, v, epoch+1)

            if (epoch + 1) % save_iter == 0:
                checkpoint_path = os.path.join(output_dir, f'checkpoint_{epoch+1:04}.pth')
                torch.save({
                    'model': ddp_model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch+1,
                },checkpoint_path)
                logger.info(f'Saving checkpoint at epoch{epoch+1}')

            if eval_loss < best_val_loss:
                torch.save({
                    'model': ddp_model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch+1,
                },os.path.join(output_dir, 'best_checkpoint.pth'))
                best_val_loss = eval_loss
                logger.info(f'Best checkpoint saving at epoch{epoch+1}')
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('End training')
    if rank == 0:
        logger.info('Total time {}'.format(total_time_str))


def main():
    parser = argparse.ArgumentParser('My WSI training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args_dict = vars(args)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    args_dict['output_dir'] = os.path.join(args_dict['output_dir'], timestamp)
    args = argparse.Namespace(**args_dict)

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if local_rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    dist.barrier()
    logger = setup_logger(local_rank, args.output_dir)
    
    train_loop(local_rank, args, logger)
    

if __name__ == '__main__':

    main()