import torch.nn as nn
from torchsummary import summary


class MLPContrastive(nn.Module):
    def __init__(self):
        super(MLPContrastive, self).__init__()
        
        # Backbone network
        self.backbone = nn.Sequential(
            nn.Linear(784, 5000),
            nn.BatchNorm1d(5000),
            nn.ReLU(inplace=True),
            nn.Linear(5000, 5000),
            nn.BatchNorm1d(5000),
            nn.ReLU(inplace=True),
            nn.Linear(5000, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(inplace=True)
        )
        
        # Projector network
        self.projector = nn.Sequential(
            nn.Linear(50, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 50)
        )
        
        # Online head network
        self.online_head = nn.Linear(50, 1)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.projector(x)
        x = self.online_head(x)
        return x

if __name__ == '__main__':
    model = MLPContrastive()
    print(model)
    summary(model.cuda(), (784, ))