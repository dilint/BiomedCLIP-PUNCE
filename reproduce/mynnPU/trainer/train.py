# Heavily inspired by the Trainer class of the link below
# https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-training-3-4-8242d31de234
import os 
import torch
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from datetime import datetime

# 读取位置
import os
import json

current_path = os.path.abspath(__file__)
parent_path = os.path.dirname(os.path.dirname(current_path))
config_file = os.path.join(parent_path, "config.json")

with open(config_file, "r") as f:
    config_data = json.load(f)

# 提取所需的值
root_path = config_data["root_path"]

class Trainer:
  def __init__(self,
               name                   : str,
               model                  : torch.nn.Module,
               device                 : torch.device,
               criterion              : torch.nn.Module,
               prior                  : float,
               
               optimizer              : torch.optim.Optimizer,
               train_Dataloader       : torch.utils.data.Dataset,                                           
               valid_Dataloader       : torch.utils.data.Dataset = None,

               validtrain_Dataloader  : torch.utils.data.Dataset = None,

               lr_scheduler           : torch.optim.lr_scheduler = None,
               epochs                 : int = 100, # 100
               epoch                  : int = 0,
               notebook               : bool = True,
               n_gpu                  : int = 1):
  
    self.name             = name 
    self.model            = model
    self.device           = device
    self.criterion        = criterion
    self.prior            = prior
    self.optimizer        = optimizer
    self.lr_scheduler     = lr_scheduler
    self.train_Dataloader = train_Dataloader
    self.valid_Dataloader = valid_Dataloader
    
    self.validtrain_Dataloader = validtrain_Dataloader

    self.epochs           = epochs
    self.epoch            = epoch
    self.notebook         = notebook 

    self.train_loss       = []
    self.valid_loss       = []

    self.train_error      = []
    self.valid_error      = []
    self.valid_acc        = []

    self.train_ROC_AUC    = []
    self.valid_ROC_AUC    = []

    self.learning_rate    = []


    self.test1            = []
    self.test2            = []

    self.n_gpu            = n_gpu
    self.device_ids = list(range(10))[:n_gpu]
      
    # 指定要用到的设备
    self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
    # 模型加载到设备0
    self.model = self.model.cuda(device=self.device_ids[0])
    self.device = self.device_ids[0]
    
    
  def run_trainer(self):

    if self.notebook:
      from tqdm.notebook import tqdm, trange
    else:
      from tqdm import tqdm, trange
    
    progressbar = trange(self.epochs-self.epoch, desc='Progress', leave= False)
    
    for i in progressbar:
      """Epoch Counter"""
      self.epoch += 1

      """Training block"""
      self._train()

      # """Validation/Train block (From Chainer) /"""
      # if self.validtrain_Dataloader is not None:
      #   self._validateTrain()
      if self.epoch % 10 == 0:
        self.valid_acc.append(self._validate())
      
      # """Validation block"""
      # if self.valid_Dataloader is not None:
      #   self._validate()

      """Learning rate scheduler block"""

      if self.lr_scheduler is not None:
        if self.valid_Dataloader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            self.lr_scheduler.step(self.validation_loss[i])  # learning rate scheduler step with validation loss
        else:
            self.lr_scheduler.step()  # learning rate scheduler step

    self._save_checkpoint()
    progressbar.close()

    # return self.train_loss, self.valid_loss, self.train_dice_coef, self.valid_dice_coef,  self.learning_rate


  def run_validate(self, save_dir: str):
      self._load_checkpoint(save_dir)
      """Validation block"""
      if self.valid_Dataloader is not None:
        acc = self._validate()
        print("The accuracy of this checkpooint is {}".format(acc))
      else:
        print("Do not have valid dataloader!")
    # return self.train_loss, self.valid_loss, self.train_dice_coef, self.valid_dice_coef,  self.learning_rate


  def run_resume(self, save_dir: str):
    self._load_checkpoint(save_dir)
    self.run_trainer()
    

  def _train(self):

    if self.notebook:
      from tqdm.notebook import tqdm, trange
    else:
      from tqdm import tqdm, trange
    
    self.model.train() # train mode
    train_losses      = []  # accumulate the losses here
    train_batch_ROC   = []
    batch_train_error = (np.zeros(4))

    batch_iter = tqdm(enumerate(self.train_Dataloader), 'Training', total=len(self.train_Dataloader),
                      leave= False)
    
    for i, data in batch_iter:
      input, target = data
      input, target = input.to(self.device), target.to(self.device) # Send to device (GPU or CPU)

      self.optimizer.zero_grad() # Set grad to zero
      
      output  = self.model(input) # One forward pass 
      output  = output.squeeze()

      loss, x_out   = self.criterion(output, target) # Calculate loss
      loss_value    = loss.item()
      train_losses.append(loss_value)
      x_out.backward() # one backward pass
      self.optimizer.step() # update the parameters
      batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar

      # Normalization for ROC (for binary data evaluation/accuracy)
      roc_target = target.cpu().numpy()
      roc_output = torch.tanh(output)
      roc_output = roc_output.detach().cpu().numpy()
      fpr, tpr, thresholds = metrics.roc_curve(roc_target, roc_output)
      ROC = metrics.auc(fpr, tpr)
      train_batch_ROC.append(ROC)

      # Error Chainer
      batch_train_error += self._batch_train_error(output, target)

    # if self.validtrain_Dataloader is None: 
    self.train_loss.append(np.mean(np.array(train_losses)))
    self.train_ROC_AUC.append(np.mean(train_batch_ROC))
    self.learning_rate.append(self.optimizer.param_groups[0]['lr'])
    # Error Chainer
    self.train_error.append(self._train_error(batch_train_error))

    batch_iter.close()

  def _validateTrain(self):

    if self.notebook:
      from tqdm.notebook import tqdm, trange
    else:
      from tqdm import tqdm, trange
    
    self.model.train() # train mode
    train_losses      = []  # accumulate the losses here
    train_batch_ROC   = []
    batch_train_error = (np.zeros(4))


    batch_iter = tqdm(enumerate(self.validtrain_Dataloader), 'ValidTrain', total=len(self.validtrain_Dataloader),
                      leave= False)
    
    for i, data in batch_iter:
      with torch.no_grad():
        input, target = data

        input, target = input.to(self.device), target.to(self.device) # Send to device (GPU or CPU)
        
        output  = self.model(input) # One forward pass 
        output  = output.squeeze()

        loss, x_out   = self.criterion(output, target) # Calculate loss
        loss_value    = loss.item()
        train_losses.append(loss_value)
        batch_iter.set_description(f'ValidTrain: (loss {loss_value:.4f})')  # update progressbar

        # Normalization for ROC (for binary data evaluation/accuracy)
        roc_target = target.cpu().numpy()
        roc_output = torch.tanh(output)
        roc_output = roc_output.detach().cpu().numpy()
        fpr, tpr, thresholds = metrics.roc_curve(roc_target, roc_output)
        ROC = metrics.auc(fpr, tpr)
        train_batch_ROC.append(ROC)

        # Error Chainer
        batch_train_error += self._batch_train_error(output, target)

    self.train_loss.append(np.mean(np.array(train_losses)))
    self.train_ROC_AUC.append(np.mean(train_batch_ROC))
    self.learning_rate.append(self.optimizer.param_groups[0]['lr'])
    ##Error Chainer
    self.train_error.append(self._train_error(batch_train_error))

    batch_iter.close()
    
  def _validate(self):
    if self.notebook:
      from tqdm.notebook import tqdm, trange
    else:
      from tqdm import tqdm, trange

    self.model.eval() # evaluation mode
    valid_losses      = [] # accumulate the losses here        
    valid_batch_ROC   = []
    iter_valid_error = 0


    batch_iter = tqdm(enumerate(self.valid_Dataloader), 'Validation', total=len(self.valid_Dataloader),
                      leave=False)

    for i, data in batch_iter:
      input, target = data
      input, target = input.to(self.device), target.to(self.device) # Send to device (GPU or CPU)

      with torch.no_grad():
        output      = self.model(input)
        output      = output.squeeze()
        # loss,x_out  = self.criterion(output, target)
        # loss_value  = loss.item()
        # valid_losses.append(loss_value)
        # batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')

        # Normalization for ROC (for binary data evaluation/accuracy)
        roc_target = target.cpu().numpy()
        roc_output = torch.tanh(output)
        roc_output = roc_output.detach().cpu().numpy()
        fpr, tpr, thresholds = metrics.roc_curve(roc_target, roc_output)
        ROC = metrics.auc(fpr, tpr)
        valid_batch_ROC.append(ROC)

        # Error chainer
        iter_valid_error += self._valid_error(output,target)

    self.valid_loss.append(np.mean(np.array(valid_losses)))
    self.valid_ROC_AUC.append(np.mean(ROC))
    # Error Chainer 
    iter_valid_error = float(iter_valid_error) / len(self.valid_Dataloader.dataset)
    self.valid_error.append(iter_valid_error)
    batch_iter.close()
    accuracy = 1 - iter_valid_error
    return accuracy

  def plot_loss(self, to_save= False):
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(np.arange(self.epochs), self.train_loss)
    plt.plot(np.arange(self.epochs), self.valid_loss)
    plt.legend(['train_loss','valid_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss ')
    
    plt.title(self.name +' Train/valid loss')

    if to_save:
      name = '_TrainValid_Loss'
      name = self.name + name + '.png'
      path = '/content/drive/MyDrive/Colab Notebooks/MasterThesis/MLP/img'
      path = os.path.join(path, name)
      print(path)
      plt.savefig(path)

  def plot_ROC_AUC(self, to_save= False):
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(np.arange(self.epochs), self.train_ROC_AUC)
    plt.plot(np.arange(self.epochs), self.valid_ROC_AUC)
    plt.legend(['train','valid'])
    plt.xlabel('epoch')
    plt.ylabel('ROC AUC')

    plt.title(self.name + ' Train/valid ROC AUC')

    if to_save:
      name = '_TrainValid_ROCAUC'
      name = self.name + name + '.png'
      path = '/content/drive/MyDrive/Colab Notebooks/MasterThesis/MLP/img'
      path = os.path.join(path, name)
      print(path)
      plt.savefig(path)

  def plot_error_from_chainer(self, to_save= False):
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(np.arange(self.epochs), self.train_error)
    plt.plot(np.arange(self.epochs), self.valid_error)
    plt.legend(['train','valid'])
    plt.xlabel('epoch')
    plt.ylabel('Errors')

    plt.title(self.name + ' Train/valid Errors from Chainer')

    if to_save:
      name = '_TrainValid_Errors'
      name = self.name + name + '.png'
      path = '/content/drive/MyDrive/Colab Notebooks/MasterThesis/MLP/img'
      path = os.path.join(path, name)
      print(path)
      plt.savefig(path)

  def _batch_train_error(self, output, target):
    # Prediction
    h   = torch.flatten(torch.sign(output))
    t   = torch.flatten(target)
    
    h   = h.detach().cpu().numpy()
    t   = t.cpu().numpy()

    # Number of positive/negative
    n_p = (t == 1).sum()
    n_n = (t == -1).sum()

    # True/False Positive/Negative
    t_p = ((h == 1) * (t == 1)).sum()
    t_n = ((h == -1) * (t == -1)).sum()
    f_p = n_n - t_n
    f_n = n_p - t_p

    return int(t_p), int(t_n), int(f_p), int(f_n)


# 经过推导可以验证 可以通过error_u来推导出大致的正确率
  def _train_error(self, summary):
    prior = self.prior
    t_p, t_u, f_p, f_u = summary
    n_p = t_p + f_u
    n_u = t_u + f_p
    error_p = 1 - t_p / n_p
    error_u = 1 - t_u / n_u
    train_error= 2 * prior * error_p + error_u - prior
    
    return train_error

  def _valid_error(self, output, target):
    # Prediction
    h = torch.flatten(torch.sign(output))
    t = torch.flatten(target)

    h   = h.detach().cpu().numpy()
    t   = t.cpu().numpy()
    
    result = (h != t).sum()

    return result


  def _save_checkpoint(self, save_dir=root_path+'checkpoints'):
      if not os.path.exists(save_dir):
        os.makedirs(save_dir)
      timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
      checkpoint_path = f"{save_dir}/checkpoint_{timestamp}.pth"
      
      checkpoint = {
          'model_state_dict': self.model.state_dict(),
          'optimizer_state_dict': self.optimizer.state_dict(),
          'epoch': self.epoch
      }
      # 使用 torch.save() 将checkpoint保存到文件
      torch.save(checkpoint, checkpoint_path)

  def _load_checkpoint(self, checkpoint_path):
      # 使用 torch.load() 加载checkpoint文件
      checkpoint = torch.load(checkpoint_path)
      
      # 从checkpoint中恢复模型和优化器的状态
      self.model.load_state_dict(checkpoint['model_state_dict'])
      self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      self.epoch = checkpoint['epoch']
      


if __name__ == '__main__':
  pass