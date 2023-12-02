import hydra
from omegaconf import DictConfig
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.models import resnet18, resnet34
import torch.optim.lr_scheduler as lr_scheduler

from models import SimCLR
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
from datasets import CIFAR10PU

logger = logging.getLogger(__name__)


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


class LinModel(nn.Module):
    """Linear wrapper of encoder."""
    def __init__(self, encoder: nn.Module, feature_dim: int, n_classes: int):
        super().__init__()
        self.enc = encoder
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.lin = nn.Linear(self.feature_dim, self.n_classes)

    def forward(self, x):
        return self.lin(self.enc(x))


class NNPULoss(nn.Module): 
    def __init__(self, 
                 prior, 
                 loss=(lambda x: torch.sigmoid(-x)), 
                 beta=0.,
                 gamma=1., 
                 loss_weight=1.0,
                 nnPU=True):
        super(NNPULoss,self).__init__()
        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0,1)")
        self.prior = torch.tensor(prior)
        self.beta  = beta
        self.gamma = gamma
        self.loss  = loss
        self.nnPU  = nnPU
        self.loss_weight = loss_weight
        self.positive = 1
        self.negative = -1
        self.min_count = torch.tensor(1.)
        self.number_of_negative_loss = 0

    def forward(self, 
                input, 
                target, 
                avg_factor=None,
                test=False):
        input, target = input.view(-1), target.view(-1)
        assert(input.shape == target.shape)
        positive = target == self.positive
        unlabeled = target == self.negative
        positive, unlabeled = positive.type(torch.float), unlabeled.type(torch.float) 
    
        if input.is_cuda:
            self.min_count = self.min_count.cuda()
        self.prior = self.prior.cuda()

        n_positive, n_unlabeled = torch.max(self.min_count, torch.sum(positive)), torch.max(self.min_count, torch.sum(unlabeled))

        # Loss function for positive and unlabeled
        ## All loss functions are unary, such that l(t,y) = l(z) with z = ty
        y_positive  = self.loss(input).view(-1)  # l(t, 1) = l(input, 1)  = l(input * 1)
        y_unlabeled = self.loss(-input).view(-1) # l(t,-1) = l(input, -1) = l(input * -1)
        
        ### Risk computation
        positive_risk     = torch.sum(y_positive  * positive  / n_positive)
        positive_risk_neg = torch.sum(y_unlabeled * positive  / n_positive)
        unlabeled_risk    = torch.sum(y_unlabeled * unlabeled / n_unlabeled)
        negative_risk     = unlabeled_risk - self.prior * positive_risk_neg

        # Update Gradient 
        if negative_risk < -self.beta and self.nnPU:
            # Can't understand why they put minus self.beta
            output = self.prior * positive_risk - self.beta
            x_out  =  - self.gamma * negative_risk  
            self.number_of_negative_loss += 1
        else:
            # Rpu = pi_p * Rp + max{0, Rn} = pi_p * Rp + Rn
            output = self.prior * positive_risk + negative_risk
            x_out  = self.prior * positive_risk + negative_risk

        return x_out 


def run_epoch(model, dataloader, epoch, loss_fun=None, optimizer=None, scheduler=None):
    if optimizer:
        model.train()
    else:
        model.eval()

    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('acc')
    loader_bar = tqdm(dataloader)
    
    for x, y in loader_bar:
        x, y = x.cuda(), y.cuda()
        logits = model(x)
        loss = loss_fun(logits, y)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

        acc = (torch.sign(logits) == y).float().mean()
        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(acc.item(), x.size(0))
        if optimizer:
            loader_bar.set_description("Train epoch {}, loss: {:.4f}, acc: {:.4f}"
                                       .format(epoch, loss_meter.avg, acc_meter.avg))
        else:
            loader_bar.set_description("Test epoch {}, loss: {:.4f}, acc: {:.4f}"
                                       .format(epoch, loss_meter.avg, acc_meter.avg))

    return loss_meter.avg, acc_meter.avg


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


@hydra.main(config_path='./', config_name='simclr_config.yml')
def finetune(args: DictConfig) -> None:
    train_transform = transforms.Compose([transforms.RandomResizedCrop(32),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ToTensor()])
    test_transform = transforms.ToTensor()

    data_dir = args.data_dir
    train_set = CIFAR10PU(root=data_dir, train=True, transform=train_transform, download=False,
                          labeled=10000, unlabeled=40000)
    test_set = CIFAR10PU(root=data_dir, train=False, transform=test_transform, download=False)

    # n_classes = 2
    # indices = np.random.choice(len(train_set), 10*n_classes, replace=False)
    # sampler = SubsetRandomSampler(indices)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # Prepare model
    base_encoder = eval(args.backbone)
    pre_model = SimCLR(base_encoder, projection_dim=args.projection_dim).cuda()
    pre_model.load_state_dict(torch.load('/root/project/biomed-clip-puNCE/reproduce/punce/logs/SimCLR/cifar10/'+'simclr_{}_epoch{}.pt'.format(args.backbone, args.load_epoch)))
    model = LinModel(pre_model.enc, feature_dim=pre_model.feature_dim, n_classes=1)
    model = model.cuda()

    # Fix encoder
    model.enc.requires_grad = False
    parameters = [param for param in model.parameters() if param.requires_grad is True]  # trainable parameters.
    # optimizer = Adam(parameters, lr=0.001)

    nnpuloss = NNPULoss(prior=train_set.prior, nnPU=True)
    
    optimizer = torch.optim.Adam(
        parameters,
        lr = 0.01,   # lr = 0.1 * batch_size / 256, see section B.6 and B.7 of SimCLR paper.
        weight_decay=0.005)

    # cosine annealing lr
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs)


    optimal_loss, optimal_acc = 1e5, 0.
    for epoch in range(1, args.finetune_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, epoch, nnpuloss, optimizer, scheduler)
        test_loss, test_acc = run_epoch(model, test_loader, epoch, nnpuloss)

        if train_loss < optimal_loss:
            optimal_loss = train_loss
            optimal_acc = test_acc
            logger.info("==> New best results")
            torch.save(model.state_dict(), 'simclr_lin_{}_best.pth'.format(args.backbone))

    logger.info("Best Test Acc: {:.4f}".format(optimal_acc))


if __name__ == '__main__':
    finetune()


