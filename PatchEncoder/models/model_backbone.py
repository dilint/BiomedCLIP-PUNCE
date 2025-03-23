# modified from Pytorch official resnet.py
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import timm
import torch.nn.functional as F
from torchvision import models, transforms
from torchsummary import summary
import open_clip 
from transformers import CLIPModel, CLIPProcessor, ViTMAEModel
from open_clip.timm_model import TimmModel
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import timm
from timm.layers import SwiGLUPacked

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


# dinov2: https://github.com/facebookresearch/dinov2
# BiomedCLIP: https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
# CLIP: https://github.com/mlfoundations/open_clip
# PLIP: https://github.com/PathologyFoundation/plip

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
    
def resnet_backbone(pretrained, name):
    if name == 'resnet50':
        model_baseline = models.resnet50(pretrained=pretrained)
        model_baseline.layer4 = torch.nn.Identity()
    elif name == 'resnet34':
        model_baseline = models.resnet34(pretrained=pretrained)
    elif name == 'resnet18':
        model_baseline = models.resnet18(pretrained=pretrained)
        
    model_baseline.fc = torch.nn.Identity()
    pretrained_dict = model_zoo.load_url(model_urls[name])
    model_baseline.load_state_dict(pretrained_dict, strict=False)
    return model_baseline


class Normalize_module(nn.Module):
        def __init__(self):
            super(Normalize_module, self).__init__()
        def forward(self, x):
            return F.normalize(x, dim=-1)

def biomedCLIP_backbone(without_head: bool = False):
    
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    if without_head:
        model = model.visual.trunk
    else:
        model = model.visual
    model = nn.Sequential(model, Normalize_module())
    return model, preprocess_val

def clip_backbone(without_head: bool = False):
    model, _ ,preprocess_val = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
    return model, preprocess_val

def plip_backbone():
    backbone = CLIPModel.from_pretrained("vinid/plip")
    preprocess_val = CLIPProcessor.from_pretrained("vinid/plip")
    return backbone, preprocess_val
    
    
class ResnetBackbone(nn.Module):
    def __init__(self, pretrained, name):
        super(ResnetBackbone, self).__init__()
        if name == 'resnet50':
            model_baseline = models.resnet50(pretrained=pretrained)
            model_baseline.layer4 = torch.nn.Identity()
        elif name == 'resnet34':
            model_baseline = models.resnet34(pretrained=pretrained)
        elif name == 'resnet18':
            model_baseline = models.resnet18(pretrained=pretrained)
            
        model_baseline.fc = torch.nn.Identity()
        pretrained_dict = model_zoo.load_url(model_urls[name])
        model_baseline.load_state_dict(pretrained_dict, strict=False)
        self.model = model_baseline
        self.preprocess_val = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])
        
    def forward(self, x):
        return self.model(x)
    
class BiomedclipBackbone(nn.Module):
    def __init__(self, without_head: bool = False):
        super(BiomedclipBackbone, self).__init__()
        
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        if without_head:
            model = model.visual.trunk
        else:
            model = model.visual
        self.model = model
        self.preprocess_val = preprocess_val
    
    def forward(self, x):
        features = self.model(x)
        features /= features.norm(p=2, dim=-1, keepdim=True)
        return features
        
class ClipBackbone(nn.Module):
    def __init__(self, without_head: bool = False):
        super(ClipBackbone, self).__init__()
        model, _ ,preprocess_val = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
        self.model = model
        self.preprocess_val = preprocess_val

    def forward(self, x):
        model = self.model
        features = model.encode_image(x)
        features /= features.norm(p=2, dim=-1, keepdim=True)
        return features

  
class Dinov2Backbone(nn.Module):
    def __init__(self):
        super(Dinov2Backbone, self).__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.preprocess_val = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])
        
    def forward(self, x):
        model = self.model
        features = model(x)
        features /= features.norm(p=2, dim=-1, keepdim=True)
        return features
    
class GigapathBackbone(nn.Module):
    def __init__(self):
        super(GigapathBackbone, self).__init__()
        self.model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        self.preprocess_val = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        
    def forward(self, x):
        model = self.model
        return model(x)


class PlipBackbone(nn.Module):
    def __init__(self, without_head: bool = False):
        super(PlipBackbone, self).__init__()
        self.model = CLIPModel.from_pretrained("vinid/plip")
        self.preprocess_val = CLIPProcessor.from_pretrained("vinid/plip")

    def forward(self, x):
        model = self.model
        features = model.vision_model(x)[1]
        features = model.visual_projection(features)
        features /= features.norm(p=2, dim=-1, keepdim=True)
        return features


class MaeBackbone(nn.Module):
    def __init__(self):
        super(MaeBackbone, self).__init__()
        self.model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
        self.preprocess_val = None
        
    def forward(self, x):
        model = self.model
        features = torch.mean(model(x).last_hidden_state, dim=1)
        return features


class Virchow2Backbone(nn.Module):
    def __init__(self):
        super(Virchow2Backbone, self).__init__()
        model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        self.model = model
        self.preprocess_val = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    def forward(self, x):
        model = self.model
        output = model(x)
        class_token = output[:, 0]    # size: 1 x 1280
        patch_tokens = output[:, 5:]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those

        # concatenate class token and average pool of patch tokens
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560
        return embedding

class Uni2Backbone(nn.Module):
    def __init__(self):
        super(Uni2Backbone, self).__init__()
                
        timm_kwargs = {
                    'img_size': 224, 
                    'patch_size': 14, 
                    'depth': 24,
                    'num_heads': 24,
                    'init_values': 1e-5, 
                    'embed_dim': 1536,
                    'mlp_ratio': 2.66667*2,
                    'num_classes': 0, 
                    'no_embed_class': True,
                    'mlp_layer': timm.layers.SwiGLUPacked, 
                    'act_layer': torch.nn.SiLU, 
                    'reg_tokens': 8, 
                    'dynamic_img_size': True
                }
        model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        self.model = model
        self.preprocess_val = transform

    def forward(self, x):
        model = self.model
        return model(x)
    
class CustomeVitBase(nn.Module):
    def __init__(self):
        super(CustomeVitBase, self).__init__()
        timm_model_name = 'vit_base_patch16_224'
        timm_model_pretrained = False
        timm_pool = ""
        timm_proj = 'linear'
        image_size = 224
        embed_dim=512
        model = TimmModel(
            timm_model_name,
            embed_dim,
            pretrained=timm_model_pretrained,
            pool=timm_pool,
            proj=timm_proj,
            image_size=image_size,
        )
        # model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        self.model = model
        self.preprocess_val = None

    def forward(self, x):
        features = self.model(x)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features
    

if __name__ == '__main__':

    model1 = CustomeVitBase().model
    img = torch.rand(1,3,224,224)
    f = model1(img)
    print(f.shape)
    