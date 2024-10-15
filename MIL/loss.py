import torch.nn as nn

class MylossFunc(nn.Module):
    def __init__(self, deta):
        super(MylossFunc, self).__init__()
        self.deta = deta
        
    
    def forward(self, out, label):
        out = torch.nn.functional.softmax(out, dim=1)
        m = torch.max(out, 1)[0]
        penalty = self.deta * torch.ones(m.size())
        loss = torch.where(m>0.5, m, penalty)
        loss = torch.sum(loss)
        loss = Variable(loss, requires_grad=True)
        return