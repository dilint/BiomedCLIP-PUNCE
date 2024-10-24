import torch.nn as nn
import torch
from torch.nn import MarginRankingLoss
import numpy as np

class MyBCELossWithLogits(nn.Module):
    def __init__(self, reduction='mean'):
        super(MyBCELossWithLogits, self).__init__()
        self.reduction = reduction
        
    def forward(self, logits, target):
        pred = torch.sigmoid(logits)
        output = target * torch.log(pred) + (1 - target) * torch.log(1 - pred)
        if self.reduction == 'mean':
            loss = -torch.mean(output)
        elif self.reduction == 'sum':
            loss = -torch.sum(output)
        elif self.reduction == 'none':
            loss = -output
        return loss

class MySoftBCELoss(nn.Module):
    def __init__(self, reduction='mean', neg_weight=1.0):
        super(MySoftBCELoss, self).__init__()
        self.reduction = reduction
        self.neg_weight = neg_weight
        
    def forward(self, logits, target):
        pred = torch.sigmoid(logits)
        pred = torch.clamp(pred, min=1e-7, max=1-1e-7)
        labels = torch.argmax(target, dim=1)
        output = []
        for i, label in enumerate(labels):
            if label == 0:
                one_loss = target[i] * torch.log(pred[i]) + (1 - target[i]) * torch.log(1 - pred[i])
                output.append(one_loss)
            else:
                # only calculate the loss for the related positive class and NILM class 
                one_loss = (target[i][label] * torch.log(pred[i][label])) + self.neg_weight * torch.log(1 - pred[i][0])
                output.append(one_loss)
        output = torch.stack(output)
        if self.reduction == 'mean':
            loss = -torch.mean(output)
        elif self.reduction == 'sum':
            loss = -torch.sum(output)
        elif self.reduction == 'none':
            loss = -output
        return loss

class RankingLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(RankingLoss, self).__init__()
        self.reduction = reduction


    def forward(self, logits, target):
        pos_num = logits.shape[1] - 1
        margin = 1 / pos_num
        # for example, pos_num is 4, than the margin is 0, 0.25, 0.5, 0.75
        losses = [MarginRankingLoss(margin=margin * i) for i in range(0, pos_num+1)] 
        pred = torch.sigmoid(logits)
        pred = torch.clamp(pred, min=1e-7, max=1-1e-7)
        labels = torch.argmax(target, dim=1)
        output = []
        for i, label in enumerate(labels):
            if label == 0:
                output.append(torch.tensor(0, dtype=torch.float32))
            else:
                one_loss = 0
                for j in range(1, pos_num+1):
                    x1 = logits[i][label].unsqueeze(0)
                    x2 = logits[i][j].unsqueeze(0)
                    x_neg = logits[i][0].unsqueeze(0)
                    y = torch.tensor([1], device=logits.device)
                    one_loss += losses[abs(label - j)](x1, x2, y)
                one_loss += losses[-1](x1, x_neg, y)
                output.append(one_loss)
        output = torch.stack(output)
        if self.reduction == 'mean':
            loss = torch.mean(output)
        elif self.reduction == 'sum':
            loss = torch.sum(output)
        elif self.reduction == 'none':
            loss = output
        return loss

class RankingAndSoftBCELoss(nn.Module):
    def __init__(self, reduction='mean', neg_weight=1., weight_ranking=1., weight_bce=1.):
        super(RankingAndSoftBCELoss, self).__init__()
        self.bce_loss = MySoftBCELoss(reduction=reduction, neg_weight=neg_weight)
        self.ranking_loss = RankingLoss(reduction=reduction)
        self.weight_ranking = weight_ranking
        self.weight_bce = weight_bce

    def forward(self, logits, target):
        ranking_loss = self.ranking_loss(logits, target)
        bce_loss = self.bce_loss(logits, target)
        loss = self.weight_ranking * ranking_loss + self.weight_bce * bce_loss
        return loss