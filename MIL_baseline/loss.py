import torch.nn as nn
import torch
from torch.nn import MarginRankingLoss
import torch.nn.functional as F
from torch.nn.functional import one_hot
import numpy as np
from utils import calc_iou
from typing import Optional, Union

id2labelcode = {
        0: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        1: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        2: [0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        3: [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        4: [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        5: [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    }

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


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', eps=1e-8):
        """
        一個更穩健的 Focal Loss 實現
        alpha: 類別權重
        gamma: 難易樣本調節因子
        reduction: 'mean'/'sum'/'none'
        eps: 用於數值穩定性的小值
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, inputs, targets):
        # inputs: 模型的原始 logits
        # targets: 真實標籤
        
        # 1. 計算概率，這是潛在的不穩定源頭，但無法避免
        probs = torch.sigmoid(inputs)
        
        # 2. 為 BCE 計算準備一個被 "clamp" 過的 probs，防止 log(0)
        probs_clamp = probs.clamp(self.eps, 1.0 - self.eps)
        
        # 3. 計算標準的 BCE Loss (來自 clamp 過的 probs)
        bce_loss = F.binary_cross_entropy(probs_clamp, targets, reduction='none')
        
        # 4. 計算 p_t (來自原始的、未 clamp 的 probs)
        # 這樣可以更好地反映模型原始的置信度
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # 5. 計算 focal weight
        focal_weight = (1 - p_t).pow(self.gamma)
        
        # 6. 計算 alpha weight
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # 7. 組合最終的 loss
        loss = alpha_weight * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DynamicCyclicGamma:
    def __init__(self, gamma_min=1.0, gamma_max=4.0, gamma_target=1.5, period=10, decay_rate=0.01):
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

        self.gamma_target = gamma_target
        self.period = period
        self.decay_rate = decay_rate
        self.gamma_avg = (gamma_max+gamma_min)/2.
        self.delta_gamma = (gamma_max-gamma_min)/2.

    def get_gamma_neg(self, now_step):
        cy_gamma = self.gamma_avg + self.delta_gamma*math.sin(2*math.pi*(now_step/self.period))

        gamma_neg = cy_gamma * math.exp(-self.decay_rate * now_step) + \
                self.gamma_target * (1-math.exp(-self.decay_rate*now_step))
        
        return gamma_neg

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False, ft_cls=None, num_classes=9):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        self.flag = True

        self.ft_cls = ft_cls
        self.num_classes = num_classes
        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        
        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))
        
        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets

            if self.ft_cls is not None:
                # 需要按照微调需求手动更改
                # 根据目前的测试结果看，漏的情况的原因：1）阳性类的得分不够；2）0类的得分高了
                
                # 由于1和0经常比较相近，因此我们还可以考虑不对1类动手的方案
                gamma_neg = [1.0] + [1.0] + [10.] + [10.] + [10.] + [10.] + [1.]*3
                gamma_pos = [self.gamma_pos] * 9
                #weights = [0.] + [1.]*5 + [0.]*4
                weights = [0.] + [1.] + [2.]*4 + [0.]*3
            else:
                gamma_neg = self.gamma_neg
                gamma_pos = self.gamma_pos
                weights = torch.tensor([1.]*9, device=x.device)

            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          gamma_pos * self.targets + gamma_neg * self.anti_targets)

            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss = self.loss * self.asymmetric_w
        

        if self.ft_cls is not None and 1==1:
            assert self.loss.shape[-1] == 10
            if self.ft_cls == 1:
                print("移除阳性类的loss")
                self.loss *= torch.tensor([1.] + [0.]*5 + [0.]*4).to(x.device) # 移除阳性类的loss
            elif self.ft_cls == 2: # 移除阴性类的loss:
                print("移除阴性类的loss")
                self.loss = self.loss*weights

        return -self.loss.sum(dim=1).mean()


class BuildClsLoss(nn.Module):
    def __init__(self, args):
        super(BuildClsLoss, self).__init__()
        if args.loss == 'bce':
            criterion = nn.BCEWithLogitsLoss()
        elif args.loss == 'ce':
            criterion = nn.CrossEntropyLoss()
        elif args.loss == 'softbce':
            criterion = MySoftBCELoss(neg_weight=args.neg_weight)
        elif args.loss == 'ranking':
            criterion = RankingAndSoftBCELoss(neg_weight=args.neg_weight, neg_margin=args.neg_margin)
        elif args.loss == 'aploss':
            criterion = APLoss()
        elif args.loss == 'asl':
            criterion = AsymmetricLossOptimized(gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos, ft_cls=None)
        elif args.loss == 'focal':
            criterion = BinaryFocalLoss(alpha=args.alpha, gamma=args.gamma)
            
        self.criterion = criterion
        self.args = args
        
    def forward(self, train_logits, label):
        args = self.args
        criterion = self.criterion
        batch_size = train_logits.shape[0]
        if self.args.multi_label:
            label_onehot = torch.tensor([id2labelcode[item.item()] for item in label], device=train_logits.device)
        else:
            label_onehot = one_hot(label, num_classes=args.num_classes).float()
        if args.loss in ['ce']:
            logit_loss = criterion(train_logits.view(batch_size,-1),label)
        elif args.loss in ['bce', 'softbce', 'ranking', 'focal', 'asl']:
            logit_loss = criterion(train_logits.view(batch_size,-1),label_onehot)
        elif args.loss == 'aploss':
            logit_loss = criterion.apply(train_logits.view(batch_size,-1),label_onehot)
        assert not torch.isnan(logit_loss)
        return logit_loss
    
    
class GHLoss(nn.Module):
    def __init__(self, l, sinkhorn_iterations):
        super().__init__()
        self.l = l
        self.sinkhorn_iterations = sinkhorn_iterations

    def loss(self, x, y, stat):
        gamma = 0.1
        q_x = sinkhorn(x, self.l, self.sinkhorn_iterations, stat)
        q_y = sinkhorn(y, self.l, self.sinkhorn_iterations, stat)
        return  -torch.mean(torch.sum(q_x * F.log_softmax(y/gamma, dim=1), dim=1))-\
                torch.mean(torch.sum(q_y * F.log_softmax(x/gamma, dim=1), dim=1))

    def forward(self, c1, c2, stat=None):
        loss = self.loss(c1, c2, stat)

        return loss

@torch.no_grad()
def sinkhorn(out, l, sinkhorn_iterations, stat=None):

    Q = torch.exp(out / l).t()
    
    N = Q.shape[1] 
    K = Q.shape[0] 

    r = (torch.ones((K, 1)) / K).cuda()
    c = (torch.ones((N, 1)) / N).cuda()
    if stat is None:
        inv_K = 1. / K    
    else:
        inv_K = stat.clone().detach().reshape(K,1).float()
        inv_K = inv_K/torch.sum(inv_K)
    inv_N = 1. / N

    for _ in range(sinkhorn_iterations):
        r = inv_K / (Q @ c)    
        c = inv_N / (Q.T @ r)  

    Q = r * Q * c.t()
    Q = Q.t()

    Q *= N 

    return Q


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning loss implementation.
    
    This loss function implements the supervised contrastive learning approach
    described in https://arxiv.org/pdf/2004.11362.pdf. It also supports
    unsupervised contrastive loss as used in SimCLR.
    
    The loss encourages features of samples from the same class to be similar
    while pushing features of samples from different classes apart.
    """
    
    def __init__(self,
                 temperature: float = 0.1,
                 contrast_mode: str = 'all',
                 base_temperature: float = 0.1):
        """
        Initialize the supervised contrastive loss.
        
        Args:
            temperature: Temperature parameter for scaling logits
            contrast_mode: Mode for contrastive learning ('one' or 'all')
            base_temperature: Base temperature for loss scaling
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        
    def forward(self,
                features: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the supervised contrastive loss.
        
        Args:
            features: Hidden vectors of shape [batch_size, n_views, ...]
            labels: Ground truth labels of shape [batch_size]
            mask: Contrastive mask of shape [batch_size, batch_size]
                 mask_{i,j}=1 if sample j has the same class as sample i
                 Can be asymmetric.
        
        Returns:
            A loss scalar.
            
        Raises:
            ValueError: If features dimensions are incorrect or if both labels
                      and mask are provided.
        """
        device = torch.device('cuda' if features.is_cuda else 'cpu')
        
        # Validate input dimensions
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                           'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
            
        batch_size = features.shape[0]
        
        # Validate and prepare mask
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
            
        # Prepare contrastive features
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        # Select anchor features based on mode
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f'Unknown mode: {self.contrast_mode}')
            
        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        
        # Numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Prepare mask for contrastive learning
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute loss
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
        return loss

if __name__ == '__main__':
    num_classes = 10
    batch_size = 5
    logits = torch.randn(batch_size, num_classes)
    y = F.sigmoid(logits)
    labels = torch.randint(0, num_classes, (batch_size,))
    labels_onehot = F.one_hot(labels, num_classes)
    print(y.shape, labels_onehot.shape)
    # loss = AsymmetricLossOptimized()
    # loss(logits, labels)
    
    