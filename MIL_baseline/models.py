import torch
from torch import nn
from modules.hmil import HMIL
from modules.abmil import *
from modules.transmil import *
from modules.wsi_vit import WSI_ViT
from modules.vit_nc25 import MILClassifier

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
    def __init__(self, mil='abmil', n_classes=[2,6], dropout=0.25):
        super(MIL, self).__init__()
        if mil == 'hmil':
            self.online_encoder = HMIL(n_classes=n_classes)
        
        else:
            raise ValueError(f'MIL type "{mil}" not supported')

    def forward(self, x):
        return self.online_encoder(x)
    
class Valina_MIL(nn.Module):
    def __init__(self, input_dim=1024, mlp_dim=512,n_classes=2,mil='abmil',dropout=0.25,head=8,act='gelu'):
        super(Valina_MIL, self).__init__()
        self.patch_to_emb = [nn.Linear(input_dim, mlp_dim)]
        
        if act.lower() == 'relu':
            self.patch_to_emb += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self.patch_to_emb += [nn.GELU()]

        self.dp = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.patch_to_emb = nn.Sequential(*self.patch_to_emb)

        if mil == 'transmil':
            self.online_encoder = Transmil(input_dim=mlp_dim,head=head)
        elif mil == 'abmil':
            self.online_encoder = Abmil(input_dim=mlp_dim,act=act)
        elif mil == 'wsi_vit':
            self.online_encoder = WSI_ViT(input_dim=mlp_dim, dim=mlp_dim, depth=4)
        elif mil == 'vit_nc25':
            mlp_ori = mlp_dim
            mlp_dim = 1024
            self.online_encoder = MILClassifier(in_dim=mlp_ori, fc_dim=mlp_dim)
        elif mil == 'linear':
            self.patch_to_emb = nn.Identity()
            self.online_encoder = nn.Identity()
            
        else:
            raise ValueError(f'MIL type "{mil}" not supported')

        self.predictor1 = nn.Linear(mlp_dim,n_classes[0])
        self.predictor2 = nn.Linear(mlp_dim,n_classes[1])

    def forward(self, x, return_attn=False, return_feat=False):
        x = self.patch_to_emb(x)
        x = self.dp(x)
        # ps = x.size(1)

        if return_attn:
            x,attn = self.online_encoder(x,return_attn=True)
        else:
            x = self.online_encoder(x)

        prob1 = self.predictor1(x)
        prob2 = self.predictor2(x)

        if return_attn:
            return prob1, prob2, attn
        elif return_feat:
            return prob1, prob2, x
        else:
            return prob1, prob2
        