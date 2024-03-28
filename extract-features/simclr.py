import wandb
from omegaconf import DictConfig
import logging

import numpy as np
from PIL import Image
import glob

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18, resnet34
from torchvision import transforms

from models.model_simclr import SimCLR, SimCLR_custome
from models.model_backbone import resnet50_baseline, biomedCLIP_backbone
from models.model_adapter import LinearAdapter

from tqdm import tqdm
import os
import argparse


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CIFAR10Pair(CIFAR10):
    def __init__(self,
                 transform,
                 labeled: int = 10000,
                 unlabeled: int = 40000,
                 **kargs):
        super().__init__(**kargs)        
        self.transform = transform
        self.labeled, self.unlabeled = labeled, unlabeled
        self.targets = self._binarize_cifar10_class(self.targets)
        self.data, self.targets, self.prior = self._make_pu_label_from_binary_label(self.data, self.targets)
            
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)  # .convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        return torch.stack(imgs), target  # stack a positive pair
    
    ## add function ##
    def _binarize_cifar10_class(self, y):
        """将类别分为animal和vehicle"""
        # 先转化为numpy
        y = np.array(y)
        y_bin = np.ones(len(y), dtype=int)
        y_bin[(y == 2) | (y == 3) | (y == 4) | (y == 5) | (y == 6) | (y == 7)] = 0
        return y_bin
    
    def _make_pu_label_from_binary_label(self, x, y):
        """挑选出一定的正样本数作为已标注标签"""
        """from https://github.com/kiryor/nnPUlearning"""
        y = np.array(y)
        labels = np.unique(y)
        positive, negative = labels[1], labels[0]
        labeled, unlabeled = self.labeled, self.unlabeled
        assert(len(x) == len(y))
        perm = np.random.permutation(len(y))
        x, y = x[perm], y[perm]
        n_p = (y == positive).sum()
        n_lp = labeled
        n_n = (y == negative).sum()
        n_u = unlabeled
        if labeled + unlabeled == len(x):
            n_up = n_p - n_lp
        elif unlabeled == len(x):
            n_up = n_p
        else:
            raise ValueError("Only support |P|+|U|=|X| or |U|=|X|.")
        _prior = float(n_up) / float(n_u)
        xlp = x[y == positive][:n_lp]
        xup = np.concatenate((x[y == positive][n_lp:], xlp), axis=0)[:n_up]
        xun = x[y == negative]
        x = np.asarray(np.concatenate((xlp, xup, xun), axis=0))
        print(x.shape)
        y = np.asarray(np.concatenate((np.ones(n_lp), -np.ones(n_u))))
        perm = np.random.permutation(len(y))
        y[y==-1]=negative
        x, y = x[perm], y[perm]
        return x, y, _prior

class Whole_Slide_Patchs_Ngc(Dataset):
    # pos is 0 and neg is 1, because all patches of the neg wsi are neg,
    # but pos wsi includes pos and neg patches 
    def __init__(self,
                 data_dir,
                 train_label_path,
                 transform,
                 is_ngc,
                 load_cpu,):
        # get img_path
        NGC_SUB_PATHS = [
            'Unannotated_KSJ/Unannotated-KSJ-TCTNGC-NILM',
            'Unannotated_KSJ/Unannotated-KSJ-TCTNGC-POS',
            'Unannotated_XIMEA/Unannotated-XIMEA-TCTNGC-NILM',
            'Unannotated_XIMEA/Unannotated-XIMEA-TCTNGC-POS'
        ]
        GC_SUB_PATHS = [
            'NILM',
            'POS'
        ]
        if is_ngc:
            sub_paths = NGC_SUB_PATHS
        else:
            sub_paths = GC_SUB_PATHS            
        data_roots = list(map(lambda x: os.path.join(data_dir, x), sub_paths)) 
        wsi_dirs = []
        train_wsi_lists = []
        img_paths = []
        with open(train_label_path, 'r') as f:
            lines = f.readlines()
            for row in lines:
                row = row.strip().split(',')
                train_wsi_lists.append(row[0])
        for data_root in data_roots:
            wsi_dirs.extend([os.path.join(data_root, subdir) for subdir in os.listdir(data_root)])
        
        for wsi_path in wsi_dirs:
            wsi_name = os.path.basename(wsi_path)
            if wsi_name not in train_wsi_lists:
                continue
            img_paths.extend(glob.glob(os.path.join(wsi_path, '*.jpg')))
        self.img_paths = img_paths
        # the size is too big
        self.transform = transform
        self.load_cpu = False
        if load_cpu:
            self.load_cpu = True
            self.imgs = []
            for img_path in self.img_paths:
                self.imgs.append(Image.open(img_path))
        
    def __getitem__(self, idx):
        path = self.img_paths[idx]
        if self.load_cpu:
            img = self.imgs[idx]
        else:
            img = Image.open(path)
        target = 0 
        if 'NILM' in str(path):
            target = 1
        imgs =  [self.transform(img), self.transform(img)]
        return torch.stack(imgs), target
    
    def __len__(self):
        return len(self.img_paths)

    def __str__(self) -> str:
        return f'the length of patchs in {self.img_paths} is {self.__len__()}'


def infonce(x, t=0.5):
    x = F.normalize(x, dim=1)
    x_scores =  (x @ x.t()).clamp(min=1e-7)  # normalized cosine similarity scores
    x_scale = x_scores / t   # scale with temperature

    # (2N-1)-way softmax without the score of i-th entry itself.
    # Set the diagonals to be large negative values, which become zeros after softmax.
    x_scale = x_scale - torch.eye(x_scale.size(0)).to(x_scale.device) * 1e5

    # targets 2N elements.
    targets = torch.arange(x.size()[0])
    targets[::2] += 1  # target of 2k element is 2k+1
    targets[1::2] -= 1  # target of 2k+1 element is 2k
    return F.cross_entropy(x_scale, targets.long().to(x_scale.device))

def punce(x, y, prior, temperature):
    device = x.device
    x = F.normalize(x, dim=1)
    y = y.unsqueeze(1).repeat(1, 2).reshape(-1)
    
    x_scores =  (x @ x.t()).clamp(min=1e-7)  # normalized cosine similarity scores
    x_scale = x_scores / temperature   # scale with temperature
    x_scale = x_scale - torch.eye(x_scale.size(0)).to(x_scale.device) * 1e5

    # common negative item: Nx1
    neg_item = torch.exp(x_scale)
    neg_item = torch.sum(neg_item, dim=1)
    
    # for positive sample
    pos_index = y==1
    n_pos = len(pos_index)
    pos_q = x[pos_index, :]
    neg_item_p = neg_item[pos_index]
    
    pos_item = torch.einsum('nc,cp->np', [pos_q, pos_q.T]) / temperature # 2N*2N
    pos_item = pos_item - torch.eye(pos_item.size(0)).to(device) * 1e5 # exclue self
    pos_item = torch.exp(pos_item)
    pos_item = pos_item / neg_item_p.unsqueeze(1) # N_px(N_p)
    pos_item = pos_item + torch.eye(pos_item.size(0)).to(device) # exclue self
    pos_item = torch.log(pos_item)
    
    l_p = torch.sum(pos_item, dim=1) # N_p
    l_p /= (n_pos - 1)
    l_p = -torch.sum(l_p)
    
    # for unlabeled sample
    unlabel_index = y==0
    unlabel_q = x[unlabel_index, :]
    neg_item_u = neg_item[unlabel_index]

    homo_index = torch.arange(y.size()[0])
    homo_index[::2] += 1  # target of 2k element is 2k+1
    homo_index[1::2] -= 1  # target of 2k+1 element is 2k 
    
    l_q_k = F.cross_entropy(x_scores, homo_index.to(device), reduction='none')
    l_q_k = l_q_k[unlabel_index]
    l_q_k = torch.sum(l_q_k)
    
    un_pos_item = torch.einsum('nc,cp->np', [unlabel_q, pos_q.T]) # N_nXN_p
    un_pos_item /= temperature
    un_pos_item = torch.exp(un_pos_item)
    un_pos_item = un_pos_item / neg_item_u.unsqueeze(1)
    un_pos_item = torch.log(un_pos_item)
    
    l_u_p = torch.sum(un_pos_item, dim=1) # N_n
    l_u_p = l_u_p / (n_pos+1) * prior
    l_u_p = -torch.sum(l_u_p)
    l_u = l_u_p + (prior / (n_pos+1) + 1 - prior) * l_q_k
    
    loss = (l_p + l_u) / x.shape[0]
    return loss

def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

# color distortion composed by color jittering and color dropping.
# See Section A of SimCLR: https://arxiv.org/abs/2002.05709
def get_color_distortion(s=0.5):  # 0.5 for CIFAR10 by default
    # s is the strength of color distortion
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_gray])
    return color_distort

def train(args) -> None:
    assert torch.cuda.is_available()
    cudnn.benchmark = True

    train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          get_color_distortion(s=0.5),
                                          transforms.ToTensor()])
    # train_transform = transforms.Compose([
    #                                       transforms.RandomHorizontalFlip(p=0.5),
    #                                       get_color_distortion(s=0.5),
    #                                       transforms.ToTensor()])
    
    # set dataset 
    if args.dataset == 'cifar10':
        train_set = CIFAR10Pair(root=args.data_dir,
                                train=True,
                                transform=train_transform,
                                download=True)
    elif args.dataset == 'ngc':
        train_set = Whole_Slide_Patchs_Ngc(
            data_dir=args.data_dir,
            train_label_path=args.train_label_path,
            transform=train_transform,
            is_ngc=True,
            load_cpu=args.load_cpu
        )
    elif args.dataset == 'gc':
        train_set = Whole_Slide_Patchs_Ngc(
            data_dir=args.data_dir,
            train_label_path=args.train_label_path,
            transform=train_transform,
            is_ngc=False,
            load_cpu=args.load_cpu
        )
    
    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        train_loader = DataLoader(train_set, 
                                batch_size=args.batch_size,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=args.workers,
                                sampler=train_sampler)
    else:
        train_loader = DataLoader(train_set,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.workers,
                                drop_last=True)
    print(len(train_set))
    
    if args.backbone == 'resnet50':
        backbone = resnet50_baseline(pretrained=True)
        input_dim = 1024
    elif args.backbone == 'biomedCLIP':
        backbone, _ = biomedCLIP_backbone()
        input_dim = 512
    for name, param in backbone.named_parameters():
        param.requires_grad = False
    adapter = LinearAdapter(input_dim)
    for _, param in adapter.named_parameters():
        param.requires_grad = True
    base_model = nn.Sequential(backbone, adapter)
    model = SimCLR_custome(base_model, feature_dim=input_dim)
    model = model.cuda()
    
    if args.ddp:
        model = nn.parallel.DistributedDataParallel(model)
        
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True)
    
    # cosine annealing lr
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            step,
            args.epochs * len(train_loader),
            args.learning_rate,  # lr_lambda computes multiplicative factor
            1e-3))

    epoch_start = 1
    if args.auto_resume:
        ckp = torch.load(os.path.join(args.model_path, args.ckp_path))
        epoch_start = ckp['epoch']+1
        model.load_state_dict(ckp['projector'])
        adapter.load_state_dict(ckp['adapter'])
        optimizer.load_state_dict(ckp['optimizer'])
        scheduler.load_state_dict(ckp['lr_sche'])

    # SimCLR training
    model.train()
    for epoch in range(epoch_start, args.epochs + 1):
        loss_meter = AverageMeter("SimCLR_loss")
        train_bar = tqdm(train_loader)
        for x, y in train_bar:
            sizes = x.size()
            x = x.view(sizes[0] * 2, sizes[2], sizes[3], sizes[4]).cuda(non_blocking=True)

            optimizer.zero_grad()
            # feature, rep = model(x)
            rep = model(x)
            if args.loss_function == 'infonce':
                loss = infonce(rep, args.temperature)
            else:
                loss = punce(rep, y, args.prior, args.temperature)
                
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_meter.update(loss.item(), x.size(0))
            train_bar.set_description("Train epoch {}, SimCLR loss: {:.4f}".format(epoch, loss_meter.avg))
            if args.wandb:
                if args.ddp and args.local_rank != 0:
                    pass
                else:
                    wandb.log(
                        {
                            "pretrain/train_loss": loss_meter.avg
                        }
                    )
            
        # save checkpoint very log_interval epochs
        if args.ddp and args.local_rank != 0:
            pass
        else:
            ckp = {
                'adapter': adapter.state_dict(),
                'projector': {k: v for k, v in model.state_dict().items() if k.startswith('projector')},
                'lr_sche': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'wandb_id': wandb.run.id if args.wandb else '',
            }

            if epoch >= args.log_interval and epoch % args.log_interval == 0:
                print("==> Save checkpoint. Train epoch {}, SimCLR loss: {:.4f}".format(epoch, loss_meter.avg))
                torch.save(ckp, os.path.join(args.model_path, '{}_epoch{}.pt'.format(args.title, epoch)))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch SimCLR Training')
    
    parser.add_argument('--auto_resume', action='store_true', help='automatically resume training')
    # dataset
    parser.add_argument('--dataset', type=str, default='ngc', choices=['cifar10', 'ngc', 'gc'])
    parser.add_argument('--load_cpu', action='store_true')
    parser.add_argument('--data_dir', type=str, default='/home1/wsi/ngc-output-filter/meanmil')
    parser.add_argument('--train_label_path', type=str, default='datatools/ngc-2023/ngc_labels/train_label.csv')
    # parser.add_argument('--target_patch_size', type=int, nargs='+', default=(1333, 800))
    
    # model
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'biomedCLIP'])
    # parser.add_argument('--proj_hidden_dim', default=128, type=int, help='dimension of projected features')
    
    # train 
    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--log_interval', default=10, type=int)
    
    # loss options
    parser.add_argument('--loss_function', default='infonce', type=str, choices=['infonce', 'punce'])
    parser.add_argument('--prior', default=0.25, type=float, help='prior parameter for punce')
    parser.add_argument('--optimzer', default='sgd', type=str, choices=['adam', 'sgd'])
    parser.add_argument('--learning_rate', default=0.6, type=float, help='initial lr = 0.3 * batch_size / 256')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1.0e-6, type=float, help='"optimized using LARS [...] and weight decay of 10−6"')
    parser.add_argument('--temperature', default=0.5, type=float)
    
    # wandb
    parser.add_argument('--wandb', action='store_true', help='Weight&Bias')
    parser.add_argument('--project', default='simclr-puc', type=str)
    parser.add_argument('--title', default='resnet50-simclr-test', type=str)
    parser.add_argument('--model_path', default='output-model', type=str)
    parser.add_argument('--ckp_path', type=str)
    # ddp
    parser.add_argument('--ddp', action='store_true', help="if user ddp")
    parser.add_argument("--local_rank", type=int)
    
    args = parser.parse_args()
    
    args.model_path = os.path.join(args.model_path, args.project, args.title)
    os.makedirs(args.model_path, exist_ok=True)
    
    if args.ddp:
        torch.cuda.set_device(args.local_rank) 
        torch.distributed.init_process_group(backend='nccl')    
    
    if args.ddp and args.local_rank != 0:
        pass
    else:
        if args.wandb:
            wandb.login()
            if args.auto_resume:
                ckp = torch.load(os.path.join(args.model_path, args.ckp_path))
                wandb.init(project=args.project, name=args.title,config=args,dir=os.path.join(args.model_path),id=ckp['wandb_id'],resume='must')
            else:
                wandb.init(project=args.project, name=args.title,config=args,dir=os.path.join(args.model_path))
    
    train(args)
