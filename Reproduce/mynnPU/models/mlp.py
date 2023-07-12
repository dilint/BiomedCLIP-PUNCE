from torch import nn
from torchsummary import summary

# Create the model
class MLP(nn.Module):
  def __init__(self, input_dim= 28*28*1):
    super(MLP, self).__init__()
    self.features = nn.Sequential(
        nn.Linear(input_dim, 300, bias= False),
        nn.BatchNorm1d(300),
        nn.ReLU(),
        nn.Linear(300, 300, bias= False),
        nn.BatchNorm1d(300),
        nn.ReLU(),
        nn.Linear(300, 300, bias= False),
        nn.BatchNorm1d(300),
        nn.ReLU(),
        nn.Linear(300, 300, bias= False),
        nn.BatchNorm1d(300),
        nn.ReLU(),
        nn.Linear(300, 1)   
    )

  def forward(self, input):
    output = input.view(input.size(0),-1) # OK with pytorch original
    output = self.features(output)
    return output

if __name__ == '__main__':
    print(MLP())
    summary(MLP().cuda(), (1,28,28),)