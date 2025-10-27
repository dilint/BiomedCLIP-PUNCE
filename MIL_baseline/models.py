import torch
from torch import nn
from modules.hmil import HMIL

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class MIL(nn.Module):
    def __init__(self, mil='abmil', n_classes=2, dropout=0.25):
        super(MIL, self).__init__()
        if mil == 'hmil':
            self.online_encoder = HMIL(n_classes=n_classes)
        
        else:
            raise ValueError(f'MIL type "{mil}" not supported')

    def forward(self, x):
        return self.online_encoder(x)