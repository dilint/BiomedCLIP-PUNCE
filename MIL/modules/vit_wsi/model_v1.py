import torch
import torch.nn as nn
import torch.nn.functional as F
from .vision_transformer_wsi import *

class CellAggregator(nn.Module):
    def __init__(
            self,
            in_dims: int = 256,
            num_heads: int = 8,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.in_dims = in_dims
        self.num_heads = num_heads

        self.cell_proj = nn.Linear(in_dims, in_dims)
        self.block_fn = Block(
            in_dims=in_dims,
            num_heads=num_heads,
            proj_drop=0.02,
            attn_drop=0.02,
            drop_path=0.01
        )
        self.norm = norm_layer(in_dims)
        self.att_pool = MaskedAttentionPoolLatent(
            in_dims=self.in_dims,
            num_heads=self.num_heads,
            norm_layer=norm_layer,
        )
        self.apply(init_weights_vit_timm)


    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        
        x = self.cell_proj(x)
        x, _ = self.block_fn(x, mask)
        x = self.att_pool(self.norm(x), mask)

        return x

class Model_V1(nn.Module):
    def __init__(
            self,
            in_dims: int = 256,
            num_classes: list = [1, 5, 3],
            depth: int = 6,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
            head_drop: float = 0.,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert in_dims % num_heads == 0
        self.in_dims = in_dims
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.depth = depth

        self.mlp_ratio = mlp_ratio
        self.head_drop = head_drop
        self.proj_drop = proj_drop
        self.attn_drop = attn_drop
        self.drop_path = drop_path
        self.norm_layer = norm_layer

        self.cell_fusion = CellAggregator(
            in_dims = self.in_dims,
            num_heads = self.num_heads,
            norm_layer = self.norm_layer
        )

        self.vit_heads = nn.ModuleList([
            VisionTransformer_WSI(
                in_dims=self.in_dims,
                num_heads=self.num_heads,
                num_classes=self.num_classes[i],
                embed_dim=self.in_dims,
                depth=self.depth,
                mlp_ratio=self.mlp_ratio,
                drop_rate=self.head_drop,
                proj_drop_rate=self.proj_drop,
                attn_drop_rate=self.attn_drop,
        ) for i in range(len(self.num_classes))])
                                       
        

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # mask shape [B, N, M], B is batch_size for wsi, N represent num_patch, M represent num_cell
        B,N,M,C = x.shape
        device = mask.device

        x = x.reshape(B*N, M, C)
        mask = mask.reshape(B*N, M)

        x = self.cell_fusion(x, mask).reshape(B, N, C)

        mask = mask.reshape(B, N, M)
        mask = torch.eq(torch.sum(mask, dim=-1), 0).to(device) #shape [B, N]

        #print(x.shape)
        logits = [self.vit_heads[i](x, mask) for i in range(len(self.num_classes))]

        logits = torch.cat((logits), dim=-1) # shape [B, 10]

        return logits

class AsymmetricCriterion(nn.Module):
    def __init__(self, num_classes, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super().__init__()
        assert num_classes > 2, f'Multilabel and multiclass'
        self.num_classes = num_classes
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
    
    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        losses= {}
        #losses['loss_asym_all'] = -loss.sum()
        for i in range(self.num_classes):
            losses[f'loss_class{i}'] = -loss[:,i].mean()

        return losses
    

def build_model(args):
    device = torch.device(args.device)

    model = Model_V1(
        in_dims=args.in_dims,
        num_classes=args.num_classes,
        depth=args.depth,
        num_heads=args.num_heads,
        proj_drop=args.proj_drop,
        attn_drop=args.attn_drop,
        drop_path=args.drop_path,
    )

    criterion = AsymmetricCriterion(sum(args.num_classes))
    # criterion.to(device)

    return model, criterion