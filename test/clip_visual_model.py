# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type, List
from mmcv.cnn import constant_init, trunc_normal_init


class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
        
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias
        )
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
      
    def forward(self, x: torch.Tensor, return_attention=False) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x,attn = self.attn(x)
        if return_attention:
            return attn
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,  
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        # qkv with shape (3, B, nHead, N, C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, N, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, N, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, N, -1).permute(0, 2, 1, 3).reshape(B, N, -1)
        x = self.proj(x)

        return x, attn

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,C,H,W = x.shape
        x = self.proj(x).permute(0, 2, 3, 1)
        return x


def get_abs_pos(
    abs_pos: torch.Tensor, has_cls_token: bool, hw: List[int]
) -> torch.Tensor:
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    h = hw[0]
    w = hw[1]
    if has_cls_token:
        cls_token_pos, grid_pos = abs_pos[:, 0:1], abs_pos[:, 1:]
    xy_num = grid_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_grid_pos = F.interpolate(
            grid_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )
        return torch.cat((cls_token_pos, new_grid_pos.flatten(2).permute(0, 2, 1)), dim=1)
    else:
        return abs_pos


class CLIPViT(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        init_cfg=None
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.patch_size = patch_size
        self.pretrain_use_cls_token = True
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        pretrain_img_size = 224
        self.init_cfg = init_cfg
        num_patches = (pretrain_img_size // patch_size) * (
            pretrain_img_size // patch_size
        )
        
        embed_len = num_patches + 1
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            self.blocks.append(block)

        self.norm = norm_layer(embed_dim)
        trunc_normal_init(self.pos_embed, std=.02)
        trunc_normal_init(self.cls_token, std=.02)
        self.init_weights()

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_init(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m.bias, 0)
                constant_init(m.weight, 1.0)
        self.apply(_init_weights)
        
    


    def forward(self, x: torch.Tensor):
        
        x = self.patch_embed(x) #shape [B,h_r,w_r,C]
        B,h_r,w_r,_ = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x.permute(0,3,1,2).flatten(2).permute(0,2,1)), dim=1) # B N C, N = 1+hr*wr

        x = x + get_abs_pos(
            self.pos_embed, self.pretrain_use_cls_token, [h_r, w_r]
        )
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
     
        return x[:, 0] # return cls token embedding
    
    def get_last_selfattention(self, x):
        x = self.patch_embed(x) #shape [B,h_r,w_r,C]
        B,h_r,w_r,_ = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x.permute(0,3,1,2).flatten(2).permute(0,2,1)), dim=1) # B N C, N = 1+hr*wr

        x = x + get_abs_pos(
            self.pos_embed, self.pretrain_use_cls_token, [h_r, w_r]
        )
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                return blk(x, return_attention=True)
            
    def load_from(self, pretrained:str):
        print(f'load backbone from: {pretrained}')
        clip_visual_path = {
            'biomedclip':'/home1/wsi/biomed_clip_visual.pth',
        }
        state_dict = torch.load(clip_visual_path[pretrained])
        backbone_state_dict = {}
        for key,value in state_dict.items():
            if 'trunk' in key:
                new_key = key.replace('trunk.','')
                backbone_state_dict[new_key] = value

        msg = self.load_state_dict(backbone_state_dict, strict=False)
        print('Missing keys: {}'.format(msg.missing_keys))
        print('Unexpected keys: {}'.format(msg.unexpected_keys))
        print(f"=> backbone loaded successfully '{pretrained}'")