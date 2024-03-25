import torch.nn as nn

class Biomed_Adapter(nn.Module):
    def __init__(self):
        super(Biomed_Adapter, self).__init__()
        self.hide_layer = nn.Linear(512, 512)
    
    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        x = self.hide_layer(x)
        return x
    
class Resnet50_Adapter(nn.Module):
    def __init__(self):
        super(Resnet50_Adapter, self).__init__()
        self.hide_layer = nn.Linear(1024, 1024)
    
    def forward(self, x):
        x = self.hide_layer(x)
        return x