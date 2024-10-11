import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearHead(nn.Module):
    def __init__(self, feat_dim, n_classes):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(feat_dim, n_classes))
    
    def forward(self, x):
        return self.classifier(x)