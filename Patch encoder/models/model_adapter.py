import torch.nn as nn

class LinearAdapter(nn.Module):
    def __init__(self, input_dim):
        super(LinearAdapter, self).__init__()
        self.hide_layer = nn.Linear(input_dim, input_dim)
    
    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        x = self.hide_layer(x)
        return x
