import torch


dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
print(dinov2_vitb14)