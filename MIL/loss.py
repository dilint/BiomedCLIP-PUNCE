import torch.nn as nn
import torch
from torch.nn import MarginRankingLoss
import numpy as np
from utils import calc_iou

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
                output.append(torch.mean(one_loss))
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
    def __init__(self, reduction='mean', neg_margin=0):
        super(RankingLoss, self).__init__()
        self.reduction = reduction
        self.neg_margin = neg_margin

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
                output.append(torch.tensor(0, dtype=torch.float32, device=logits.device))
            else:
                one_loss = 0
                for j in range(1, pos_num+1):
                    x1 = logits[i][label].unsqueeze(0)
                    x2 = logits[i][j].unsqueeze(0)
                    x_neg = logits[i][0].unsqueeze(0)
                    y = torch.tensor([1], device=logits.device)
                    one_loss += losses[abs(label - j)](x1, x2, y)
                one_loss += self.neg_margin * losses[-1](x1, x_neg, y)
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
    def __init__(self, reduction='mean', neg_weight=1., weight_ranking=1., weight_bce=1.,neg_margin=0):
        super(RankingAndSoftBCELoss, self).__init__()
        self.bce_loss = MySoftBCELoss(reduction=reduction, neg_weight=neg_weight)
        self.ranking_loss = RankingLoss(reduction=reduction, neg_margin=neg_margin)
        self.weight_ranking = weight_ranking
        self.weight_bce = weight_bce

    def forward(self, logits, target):
        ranking_loss = self.ranking_loss(logits, target)
        bce_loss = self.bce_loss(logits, target)
        loss = self.weight_ranking * ranking_loss + self.weight_bce * bce_loss
        return loss

class APLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets):
 
        ########################AP-LOSS##########################
        classification_grads,classification_losses=AP_loss(logits,targets)
        #########################################################

        ctx.save_for_backward(classification_grads)
        return classification_losses

    @staticmethod
    def backward(ctx, out_grad1):
        g1, =ctx.saved_tensors
        return g1*out_grad1,None


def AP_loss(logits,targets):
    
    delta=1.0

    grad=torch.zeros(logits.shape).cuda()
    metric=torch.zeros(1).cuda()

    if torch.max(targets)<=0:
        return grad, metric
  
    labels_p=(targets==1)
    fg_logits=logits[labels_p]
    threshold_logit=torch.min(fg_logits)-delta

    ######## Ignore those negative j that satisfy (L_{ij}=0 for all positive i), to accelerate the AP-loss computation.
    valid_labels_n=((targets==0)&(logits>=threshold_logit))
    valid_bg_logits=logits[valid_labels_n] 
    valid_bg_grad=torch.zeros(len(valid_bg_logits)).cuda()
    ########

    fg_num=len(fg_logits)
    prec=torch.zeros(fg_num).cuda()
    order=torch.argsort(fg_logits)
    max_prec=0

    for ii in order:
        tmp1=fg_logits-fg_logits[ii] 
        tmp1=torch.clamp(tmp1/(2*delta)+0.5,min=0,max=1)
        tmp2=valid_bg_logits-fg_logits[ii]
        tmp2=torch.clamp(tmp2/(2*delta)+0.5,min=0,max=1)
        a=torch.sum(tmp1)+0.5
        b=torch.sum(tmp2)
        tmp2/=(a+b)
        current_prec=a/(a+b)
        if (max_prec<=current_prec):
            max_prec=current_prec
        else:
            tmp2*=((1-max_prec)/(1-current_prec))
        valid_bg_grad+=tmp2
        prec[ii]=max_prec 

    grad[valid_labels_n]=valid_bg_grad
    grad[labels_p]=-(1-prec) 

    fg_num=max(fg_num,1)

    grad /= (fg_num)
    
    metric=torch.sum(prec,dim=0,keepdim=True)/fg_num

    return grad, 1-metric

if __name__ == '__main__':
    logits = [[0.05, 0.05, 0.3, 0.4, 0.2],
              [0.05, 0.05, 0.3, 0.4, 0.2]]
    targets = [[0, 0, 0, 1, 0],
               [0, 0, 1, 0, 0]]
    logits = torch.tensor(logits).cuda()
    targets = torch.tensor(targets).cuda()
    aploss = APLoss()
    loss = APLoss.apply(logits, targets)
    
    print(loss)