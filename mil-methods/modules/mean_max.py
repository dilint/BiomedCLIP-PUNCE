import torch.nn as nn

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class MeanMIL(nn.Module):
    def __init__(self,input_dim,n_classes=1,dropout=True,act='relu',test=False):
        super(MeanMIL, self).__init__()
        self.test = test
        head = [nn.Linear(input_dim,192)]

        if act.lower() == 'relu':
            head += [nn.ReLU()]
        elif act.lower() == 'gelu':
            head += [nn.GELU()]

        if dropout:
            head += [nn.Dropout(0.25)]
            
        head += [nn.Linear(192,n_classes)]
        
        self.head = nn.Sequential(*head)

        self.apply(initialize_weights)

    
    def foward_test(self, x):
        x = self.head(x)
        return x
            
    def forward(self,x):
        if self.test:
            return self.foward_test(x)
        
        x = self.head(x)
        x = x.mean(dim=1)
        return x

class MaxMIL(nn.Module):
    def __init__(self,input_dim,n_classes=1,dropout=True,act='relu',test=False):
        super(MaxMIL, self).__init__()
        self.test = test
        head = [nn.Linear(input_dim,512)]

        if act.lower() == 'relu':
            head += [nn.ReLU()]
        elif act.lower() == 'gelu':
            head += [nn.GELU()]

        if dropout:
            head += [nn.Dropout(0.25)]

        head += [nn.Linear(512,n_classes)]
        self.head = nn.Sequential(*head)

        self.apply(initialize_weights)

    
    def foward_test(self, x):
        x = self.head(x)
        return x
            
    def forward(self,x):
        if self.test:
            return self.foward_test(x)
        
        x = self.head(x)
        x,_ = x.max(dim=1)
        return x
