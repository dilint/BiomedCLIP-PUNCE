# modified from Pytorch official resnet.py
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary
import open_clip

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
resnet50_model_path = ''

def resnet_backbone(pretrained, name):
    if name == 'resnet50':
        resnet = models.resnet50(pretrained=False)
        resnet.fc = nn.Sequential(
                        nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 128)
                    )
        pth = torch.load(resnet50_model_path)
        weights = pth['state_dict']
        new_weights = dict()
        for weight_key in weights.keys(): 
            if 'module.encoder_q.' in weight_key:
                new_key = weight_key.replace('module.encoder_q.', '')
                new_weights[new_key] = weights[weight_key]
        print(resnet.load_state_dict(new_weights, strict=False))
        return resnet
    
    elif name == 'resnet34':
        model_baseline = models.resnet34(pretrained=pretrained)
    elif name == 'resnet18':
        model_baseline = models.resnet18(pretrained=pretrained)
        
    model_baseline.fc = torch.nn.Identity()
    pretrained_dict = model_zoo.load_url(model_urls[name])
    model_baseline.load_state_dict(pretrained_dict, strict=False)
    return model_baseline


def biomedCLIP_backbone(without_head: bool = False):
    class Normalize_module(nn.Module):
        def __init__(self):
            super(Normalize_module, self).__init__()
        def forward(self, x):
            return F.normalize(x, dim=-1)
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    if without_head:
        model = model.visual.trunk
    else:
        model = model.visual
    model = nn.Sequential(model, Normalize_module())
    return model, preprocess_val

if __name__ == '__main__':
    model1 = resnet50_baseline()
    model = resnet_backbone()
    print(model1)
    print(model)