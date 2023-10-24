import torch.nn as nn
from torchsummary import summary

class AllConv(nn.Module):
    def __init__(self):
        super(AllConv, self).__init__()
        self.backbone_cnn = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 10, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True)
        )
        self.backbone_lin = nn.Sequential(
            nn.Linear(640, 1000, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 1000, bias=True),
            nn.ReLU(inplace=True)
        )
        self.projector = nn.Sequential(
            nn.Linear(1000, 1000, bias=True),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 1000, bias=True)
        )
        self.online_head = nn.Linear(1000, 1, bias=True)

    def forward(self, x):
        x = self.backbone_cnn(x)
        x = x.view(x.size(0), -1)
        x = self.backbone_lin(x)
        x = self.projector(x)
        x = self.online_head(x)
        return x

if __name__ == '__main__':
    model = AllConv()
    print(model)
    summary(model.cuda(), (3, 32, 32))