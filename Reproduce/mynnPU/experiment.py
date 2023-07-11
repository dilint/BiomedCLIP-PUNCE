from dataTools.dataset import MNIST_Chainer, load_dataset
from torch.utils.data import DataLoader
import torch

from models.allConv import AllConv
from trainer.train import Trainer
from losses.puLoss import PULoss

SEED = 0
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
XYtrain, XYtest, prior = load_dataset("mnist", 1000, 60000)
prior = torch.tensor(prior)

dataset = {'train': MNIST_Chainer(XYtrain,transform= None),
           'valid': MNIST_Chainer(XYtest, transform= None)}           
batch_size =   3000
dataloader = {'train': DataLoader(dataset['train'], batch_size= batch_size, shuffle= True, drop_last= True, **kwargs),       # drop_last= True
              'validtrain': DataLoader(dataset['train'], batch_size= batch_size, shuffle= False, **kwargs),
              'valid': DataLoader(dataset['valid'], batch_size= batch_size, shuffle= False, **kwargs)}

# print(prior)
lr = 0.001 #0.0001
n_epochs   = 100
kwargs2 = {
          'train_Dataloader': dataloader['train'],
          'valid_Dataloader': dataloader['valid'],
          'validtrain_Dataloader': dataloader['validtrain'],
          'epochs': n_epochs,   
          'notebook': False,        
          }
# print(kwargs2)

model = AllConv().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr= lr, weight_decay=0.005)

trainer_uPU  = Trainer('uPU', 
                    model,
                    device, 
                    PULoss(prior= prior, nnPU= False),
                    prior,
                    optimizer,
                    lr_scheduler = None,
                    **kwargs2)
trainer_uPU.run_trainer()
# clear_output()