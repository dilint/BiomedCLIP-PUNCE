import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from timm.layers import Mlp, DropPath, PatchDropout, LayerType, get_act_layer, get_norm_layer, trunc_normal_, trunc_normal_tf_
from timm.models._manipulate import named_apply

from typing import Optional, Type
from functools import partial

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

class MaskedAttention(nn.Module):

    def __init__(
            self,
            in_dims: int = 256,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert in_dims % num_heads == 0, 'dim should be divisible by num_heads'
        self.in_dims = in_dims
        self.num_heads = num_heads
        self.head_dim = in_dims // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(in_dims, in_dims * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_dims, in_dims)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # mask shape [B, N]
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k) # shape [B, num_heads, N, head_dim]

        
        q = q * self.scale
        attn = q @ k.transpose(-2, -1) # shape [B, num_heads, N, N]
        
        #new, adding mask
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, N, 1).unsqueeze(1).repeat(1, self.num_heads, 1, 1) # shape [B, N] --> shape [B, N, N] --> shape [B, num_heads, N, N]
            #print(attn.shape, mask.shape)
            attn = attn.masked_fill(mask == 0, -1e20)
        #end new

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    

class MaskedAttentionPoolLatent(nn.Module):
    """ Attention pooling w/ latent query
    """

    def __init__(
            self,
            in_dims: int = 256,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            latent_len: int = 1,
            pool_type: str = 'token',
            norm_layer: Optional[nn.Module] = None,
            drop: float = 0.0,
    ):
        super().__init__()
        
        assert in_dims % num_heads == 0
        self.in_dims = in_dims
        self.num_heads = num_heads
        self.head_dim = in_dims // num_heads
        self.scale = self.head_dim ** -0.5
        self.pool = pool_type
        
        self.latent_dim = in_dims
        self.latent_len = latent_len
        self.latent = nn.Parameter(torch.zeros(1, self.latent_len, in_dims))

        self.q = nn.Linear(in_dims, in_dims, bias=qkv_bias)
        self.kv = nn.Linear(in_dims, in_dims * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(in_dims, in_dims)
        self.proj_drop = nn.Dropout(drop)

        self.norm = norm_layer(in_dims) if norm_layer is not None else nn.Identity()
        self.mlp = Mlp(in_dims, int(in_dims * mlp_ratio))

        self.init_weights()

    def init_weights(self):
        
        trunc_normal_tf_(self.latent, std=self.latent_dim ** -0.5)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # mask shape [B, N]
        B, N, C = x.shape

        q_latent = self.latent.expand(B, -1, -1) # shape [B, latent_len, embed_dim]
        q = self.q(q_latent).reshape(B, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2) # shape [B, num_heads, latent_len, head_dim]

        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0) # shape [B, num_heads, N, head_dim]

        q, k = self.q_norm(q), self.k_norm(k)

       
        q = q * self.scale
        attn = q @ k.transpose(-2, -1) # shape [B, num_heads, latent_len, N]

        #new
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.latent_len, 1).unsqueeze(1).repeat(1, self.num_heads, 1, 1) # shape [B, N] --> shape [B, latent_len, N] --> shape [B, num_heads, self.latent_len, N]
            attn = attn.masked_fill(mask == 0, -1e20)
        #end new

        attn = attn.softmax(dim=-1)
        x = attn @ v
        
        x = x.transpose(1, 2).reshape(B, self.latent_len, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x + self.mlp(self.norm(x))

        # optional pool if latent seq_len > 1 and pooled output is desired
        if self.pool == 'token':
            x = x[:, 0]
        elif self.pool == 'avg':
            x = x.mean(1)
        return x
    

class Block(nn.Module):
    def __init__(
            self,
            in_dims: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(in_dims)
        self.attn = MaskedAttention(
            in_dims=in_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(in_dims)
        self.mlp = mlp_layer(
            in_features=in_dims,
            hidden_features=int(in_dims * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
    
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        
        x = x + self.drop_path1(self.attn(self.norm1(x), mask))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x, mask


class VisionTransformer_WSI(nn.Module):
    def __init__(
            self,
            in_dims: int = 256,
            num_classes: int = 10,
            global_pool: Literal['', 'avg', 'token', 'map'] = 'token',
            embed_dim: int = 256,
            depth: int = 6,
            num_heads: int = 8,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            class_token: bool = True,
            reg_tokens: int = 0,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: Literal['skip', ''] = '',
            norm_layer: Optional[LayerType] = None,
            act_layer: Optional[LayerType] = None,
            block_fn: Type[nn.Module] = Block,
            mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        assert global_pool in ('', 'avg', 'token', 'map')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.in_dims = in_dims
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        

        # self.patch_proj = nn.Sequential(
        #     nn.LayerNorm(embed_dim),
        #     nn.Linear(in_dims, embed_dim)
        # )

        self.patch_proj = nn.Linear(in_dims, embed_dim)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        
        # if patch_drop_rate > 0:
        #     self.patch_drop = PatchDropout(
        #         patch_drop_rate,
        #         num_prefix_tokens=self.num_prefix_tokens,
        #     )
        # else:
        #     self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            block_fn(
                in_dims=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        if global_pool == 'map':
            self.attn_pool = MaskedAttentionPoolLatent(
                in_dims=self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights()

    def init_weights(self) -> None:
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(init_weights_vit_timm, self)


    def forward_features(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.patch_proj(x)
        # x = self.patch_drop(x)
        x = self.norm_pre(x)
        for block in self.blocks:
            x, mask = block(x, mask)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, mask: torch.Tensor = None, pre_logits: bool = False) -> torch.Tensor:
        if self.attn_pool is not None:
            x = self.attn_pool(x, mask)
        elif self.global_pool == 'avg':
            x = x[:, self.num_prefix_tokens:].mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]  # class token
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.forward_features(x, mask)
        x = self.forward_head(x, mask)
        return x




def init_weights_vit_timm(module: nn.Module, name: str = '') -> None:
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()