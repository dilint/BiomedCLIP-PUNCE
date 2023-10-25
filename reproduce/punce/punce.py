import hydra
from omegaconf import DictConfig
import logging

import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet34
from torchvision import transforms

from models import SimCLR
from datasets import CIFAR10Pair, CIFAR10PU
from tqdm import tqdm


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


def nt_xent(x, t=0.5):
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


def punce_loss(x, y, prior, temperature=0.5, ):
    q = F.normalize(x, dim=1)
    labels = y.repeat(2, 1).T.reshape(-1)
    # common negative item: Nx1
    neg_item = torch.einsum('nc,ck->nk', [q, q.T]) # NXN
    neg_item = neg_item - torch.eye(neg_item.size(0)).to(neg_item.device) * 1e5  # exclude self
    neg_item /= temperature
    neg_item = torch.exp(neg_item)
    neg_item = torch.sum(neg_item, dim=1)
    neg_item = torch.log(neg_item) # N
    
    # for positive sample
    pos_index = labels==1
    pos_q = q[pos_index, :]
    neg_item_p = neg_item[pos_index]
    
    pos_item = torch.einsum('nc,ck->nk', [pos_q, pos_q.T]) # N_pxN_p
    # exclude i
    diag = torch.diag_embed(torch.diag(pos_item))
    pos_item -= diag
    pos_item /= temperature
    pos_item = torch.log(torch.exp(pos_item))
    
    l_p = torch.sum(pos_item, dim=1) # N_p 
    l_p /= (len(pos_item) - 1)
    l_p -= neg_item_p
    l_p = -torch.sum(l_p)
    
    # for unlabeled sample
    unlabel_index = labels==-1
    unlabel_q = q[unlabel_index, :]
    neg_item_u = neg_item[unlabel_index]
    q_k_targets = torch.arange(labels.size()[0])
    q_k_targets[::2] += 1  # target of 2k element is 2k+1
    q_k_targets[1::2] -= 1  # target of 2k+1 element is 2k
    
    un_q_k = torch.einsum('nc,nc->n', [unlabel_q, q[q_k_targets[unlabel_index], :] ]).unsqueeze(-1) # N_nX1
    un_pos_item = torch.einsum('nc,cp->np', [unlabel_q, pos_q.T]) # N_nXP
    un_pos_item = torch.cat((un_q_k, un_pos_item), dim=1) # N_nX(P+1)
    un_pos_item /= temperature
    
    l_u_p = torch.sum(un_pos_item, dim=1) # N_n
    l_u_p = l_u_p / (pos_q.size()[0] + 1)
    
    un_neg_item = un_q_k.squeeze(1) / temperature # N_n
    l_u_n = un_neg_item
    
    l_u = prior * l_u_p + (1 - prior) * l_u_n # N_n
    l_u -= neg_item_u
    l_u = -torch.sum(l_u)
    
    loss = (l_p + l_u) / q.shape[0]
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
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


@hydra.main(config_path='./', config_name='simclr_config.yml')
def train(args: DictConfig) -> None:
    assert torch.cuda.is_available()
    cudnn.benchmark = True

    train_transform = transforms.Compose([transforms.RandomResizedCrop(32),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          get_color_distortion(s=0.5),
                                          transforms.ToTensor()])
    data_dir = args.data_dir  # get absolute path of data dir
    train_set = CIFAR10PU(root=data_dir,
                            train=True,
                            transform=train_transform,
                            download=True,
                            labeled=10000,
                            unlabeled=40000)

    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers,
                              drop_last=True)

    # Prepare model
    assert args.backbone in ['resnet18', 'resnet34']
    base_encoder = eval(args.backbone)
    model = SimCLR(base_encoder, projection_dim=args.projection_dim).cuda()
    logger.info('Base model: {}'.format(args.backbone))
    logger.info('feature dim: {}, projection dim: {}'.format(model.feature_dim, args.projection_dim))

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

    # SimCLR training
    model.train()
    for epoch in range(1, args.epochs + 1):
        loss_meter = AverageMeter("SimCLR_loss")
        train_bar = tqdm(train_loader)
        for x, y in train_bar:
            sizes = x.size()
            x = x.view(sizes[0] * 2, sizes[2], sizes[3], sizes[4]).cuda(non_blocking=True)
            # give targets to two images
            # _y = [0] * len(x) * 2
            # for i in range(len(y)):
            #     _y[2*i], _y[2*i+1] = y[i], y[i]
            # y = _y
            
            optimizer.zero_grad()
            feature, rep = model(x)
            loss = punce_loss(rep, y, train_set.prior, args.temperature)
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_meter.update(loss.item(), x.size(0))
            train_bar.set_description("Train epoch {}, SimCLR loss: {:.4f}".format(epoch, loss_meter.avg))

        # save checkpoint very log_interval epochs
        if epoch >= args.log_interval and epoch % args.log_interval == 0:
            logger.info("==> Save checkpoint. Train epoch {}, SimCLR loss: {:.4f}".format(epoch, loss_meter.avg))
            torch.save(model.state_dict(), 'simclr_{}_epoch{}.pt'.format(args.backbone, epoch))


if __name__ == '__main__':
    train()


