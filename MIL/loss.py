import torch.nn as nn
import torch

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
    def __init__(self, reduction='mean'):
        super(MySoftBCELoss, self).__init__()
        self.reduction = reduction
        
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
                one_loss = (target[i][label] * torch.log(pred[i][label])) + torch.log(1 - pred[i][0])
                output.append(one_loss)
        output = torch.stack(output)
        if self.reduction == 'mean':
            loss = -torch.mean(output)
        elif self.reduction == 'sum':
            loss = -torch.sum(output)
        elif self.reduction == 'none':
            loss = -output
        return loss