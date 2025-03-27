import torch
from torch import nn

from modules.abmil import *
from modules.transmil import *

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
    def __init__(self, input_dim=1024, mlp_dim=512,n_classes=2,mil='abmil',dropout=0.25,head=8,act='gelu'):
        super(MIL, self).__init__()
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
        elif mil == 'linear':
            self.patch_to_emb = nn.Identity()
            self.online_encoder = nn.Identity()
            
        else:
            raise ValueError(f'MIL type "{mil}" not supported')

        self.predictor = nn.Linear(mlp_dim,n_classes)

    def forward(self, x, return_attn=False):
        x = self.patch_to_emb(x)
        x = self.dp(x)
        # ps = x.size(1)

        if return_attn:
            x,attn = self.online_encoder(x,return_attn=True)
        else:
            x = self.online_encoder(x)

        x = self.predictor(x)

        if return_attn:
            return x,attn
        else:
            return x
        
class MIL_MTL(MIL):
    def __init__(self, num_classes, num_task, input_dim=1024, mlp_dim=512,n_classes=2,mil='abmil',dropout=0.25,head=8,act='gelu'):
        super(MIL_MTL, self).__init__(input_dim, mlp_dim,n_classes,mil,dropout,head,act)
        self.predictor = None
        if mil == 'linear':
            self.head_task = nn.ModuleList([nn.Linear(input_dim, num_classes[i]) for i in range(num_task)])
        else:
            self.head_task = nn.ModuleList([nn.Linear(mlp_dim, num_classes[i]) for i in range(num_task)])
        
    def forward(self, x, task_id, return_attn=False):
        x = self.patch_to_emb(x)
        x = self.dp(x)
        # ps = x.size(1)
        if return_attn:
            x, attn = self.online_encoder(x,return_attn=True)
        else:
            x = self.online_encoder(x)

        xs = [self.head_task[i](x) for i in range(len(self.head_task))]
        x = torch.cat(xs, dim=-1)
        # x = self.head_task[task_id](x)
        # x = self.head_task[task_id[0]](x)
        
        if return_attn:
            return x,attn
        else:
            return x

if __name__ == '__main__':
    b, n, c = 1, 5, 512
    device = 'cuda'
    data = torch.randn((b, n, c)).to(device)
    model = MIL_MTL(num_classes=[2,3,4],num_task=3, input_dim=c).to(device)
    output = model(data, 1)
    print(output.shape)
 