# modified from Pytorch official resnet.py
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchsummary import summary
import open_clip
from transformers import CLIPModel,CLIPProcessor


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

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

class PlipBackbone(nn.Module):
    def __init__(self, without_head: bool = False):
        super(PlipBackbone, self).__init__()
        backbone = CLIPModel.from_pretrained("vinid/plip")
        preprocess_val = CLIPProcessor.from_pretrained("vinid/plip")
        self.model = backbone
        self.preprocess_val = preprocess_val

    def forward(self, x):
        model = self.model
        features = model.vision_model(x)[1]
        features = model.visual_projection(features)
        features /= features.norm(p=2, dim=-1, keepdim=True)
        return features

if __name__ == '__main__':
    # data = torch.randn((1,3,224,224))
    # device = 'cuda'
    # data = data.to(device)
    # backbone = BiomedclipBackbone(device)
    # backbone.to(device)
    # # norm_layer = Normalize_module().to(device)
    # # backbone = nn.Sequential(backbone,norm_layer)
    # features1 = backbone(data)
    # features1 /= features1.norm(p=2, dim=-1, keepdim=True)
    
    model = ClipBackbone()
    model.to('cuda')
    print(summary(model, (3,224,224)))
