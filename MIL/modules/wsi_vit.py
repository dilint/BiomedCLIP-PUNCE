import torch
from torch import nn

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange
from .tome import bipartite_soft_matching

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., use_tome=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            #self.layers.append(nn.ModuleList([
            #    Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
            #    FeedForward(dim, mlp_dim, dropout = dropout) # 不要FFN
            #]))
            self.layers.append(Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))

        self.use_tome = use_tome

    def forward(self, x):
        #for attn, ff in self.layers:
        for attn in self.layers:
            # 在这里做tome
            B, N, D = x.shape
            
            #if self.use_tome:
            #    merge, _ = bipartite_soft_matching(x, max(1, N//3), class_token=False)
            #    print('ori shape: ', x.shape)
            #    x = merge(x, mode='mean')
            #    print('after merge shape: ', x.shape)

            x = attn(x) + x
            #x = ff(x) + x


        return x

import torch.nn.functional as F

class MultiHeadAttentionPool(torch.nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadAttentionPool, self).__init__()
        self.num_heads = num_heads
        self.att_mlp = torch.nn.Linear(input_dim, num_heads)

    def forward(self, x):
        alpha = self.att_mlp(x)
        alpha = F.leaky_relu(alpha)

        alpha = F.softmax(alpha) # [n, h]

        out = 0
        for head in range(self.num_heads):
            out = out + alpha[:, head].unsqueeze(-1) * x

        return torch.sum(out, dim=0, keepdim=True)

class WSI_ViT(nn.Module):
    def __init__(self, input_dim=256, dim=256, depth=1, heads=8, mlp_dim=256, dim_head = 64, dropout=0.1, use_tome=False):
        super().__init__()
        #assert (seq_len % patch_size) == 0

        patch_dim = input_dim
        self.to_patch_embedding = nn.Sequential(
            #Rearrange('b c (n p) -> b n (p c)', p = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            #nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
        )

        self.cls0_token = nn.Parameter(torch.randn(1, 1, dim))
        nn.init.normal_(self.cls0_token, std=1e-6)

        # self.cls15_token = nn.Parameter(torch.randn(1, 1, dim))
        # nn.init.normal_(self.cls15_token, std=1e-6)

        # self.cls69_token = nn.Parameter(torch.randn(1, 1, dim))
        # nn.init.normal_(self.cls69_token, std=1e-6)    
        
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, use_tome=use_tome)
            
        self.norm = nn.LayerNorm(dim)

    def forward(self, series):
        #print(series.shape)
        x = self.to_patch_embedding(series)
        #print('!!!!!!!!!!!!!!!!!!!', x.shape)
        b, n, _ = x.shape
        cls0_token = self.cls0_token.expand(b, -1, -1).cuda()

        h = torch.cat((cls0_token, x), dim=1)
        h = self.transformer(h)

        h = self.norm(h)
        h0 = h[:, 0] #

        return h0


if __name__ == '__main__':

    v = ViT(
        input_dim = 256,
        dim = 256,
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
    )

    time_series = torch.randn(1, 100, 256)
    patch_feature, logits = v(time_series) # (4, 1000)
    print(logits.shape, patch_feature.shape)

