import os
from Reproduce.mynnPU.dataTools.mnist import MNIST_Chainer, load_dataset, PU_MNIST, PN_MNIST
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output 
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from models.mlp import MLP  
from models.mlpContrastive import MLPContrastive
from trainer.train import Trainer
from losses.puLoss import PULoss

SEED = 0
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
XYtrain, XYtest, prior = load_dataset("mnist", 10000, 50000)
prior = torch.tensor(prior)
batch_size =   1024

dataset = {'train': MNIST_Chainer(XYtrain),
           'valid': MNIST_Chainer(XYtest)}           
dataloader = {'train': DataLoader(dataset['train'], batch_size= batch_size, shuffle= True, drop_last= True, **kwargs),       # drop_last= True
              'validtrain': DataLoader(dataset['train'], batch_size= batch_size, shuffle= False, **kwargs),
              'valid': DataLoader(dataset['valid'], batch_size= batch_size, shuffle= False, **kwargs)}

# print(prior)
lr = 0.01 #0.0001
n_epochs   = 200
kwargs2 = {
          'train_Dataloader': dataloader['train'],
          'valid_Dataloader': dataloader['valid'],
          'validtrain_Dataloader': dataloader['validtrain'],
          'epochs': n_epochs,   
          'notebook': True,        
          }
# print(kwargs2)

model = MLPContrastive().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr= lr, weight_decay=0.005)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=kwargs2["epochs"])

trainer_nnPU  = Trainer('nnPU', 
                    model,
                    device, 
                    PULoss(prior= prior, nnPU= True),
                    prior,
                    optimizer,
                    lr_scheduler = scheduler,
                    **kwargs2)

trainer_nnPU.run_trainer()
print(trainer_nnPU.criterion.number_of_negative_loss)
clear_output()