import torch 
import torch.nn as nn 


class PULoss(nn.Module): 
  def __init__(self, prior, loss=(lambda x: torch.sigmoid(-x)), beta= 0,gamma= 1, nnPU= True):
    super(PULoss,self).__init__()
    if not 0 < prior < 1:
      raise NotImplementedError("The class prior should be in (0,1)")
    self.prior = prior
    self.beta  = beta
    self.gamma = gamma
    self.loss  = loss
    self.nnPU  = nnPU
    self.positive = 1
    self.negative = -1
    self.min_count = torch.tensor(1.)
    self.number_of_negative_loss = 0

  def forward(self, input, target, test= False):
    input, target = input.view(-1), target.view(-1)
    assert(input.shape == target.shape)
    positive = target == self.positive
    unlabeled = target == self.negative
    positive, unlabeled = positive.type(torch.float), unlabeled.type(torch.float) 
   
    if input.is_cuda:
      self.min_count = self.min_count.cuda()
      self.prior = self.prior.cuda()

    n_positive, n_unlabeled = torch.max(self.min_count, torch.sum(positive)), torch.max(self.min_count, torch.sum(unlabeled))

    # Loss function for positive and unlabeled
    ## All loss functions are unary, such that l(t,y) = l(z) with z = ty
    y_positive  = self.loss(input).view(-1)  # l(t, 1) = l(input, 1)  = l(input * 1)
    y_unlabeled = self.loss(-input).view(-1) # l(t,-1) = l(input, -1) = l(input * -1)
    
    # # # Risk computation
    positive_risk     = torch.sum(y_positive  * positive  / n_positive)
    positive_risk_neg = torch.sum(y_unlabeled * positive  / n_positive)
    unlabeled_risk    = torch.sum(y_unlabeled * unlabeled / n_unlabeled)
    negative_risk     = unlabeled_risk - self.prior * positive_risk_neg

    # Update Gradient 
    if negative_risk < -self.beta and self.nnPU:
      # Can't understand why they put minus self.beta
      output = self.prior * positive_risk - self.beta
      x_out  =  - self.gamma * negative_risk  
      self.number_of_negative_loss += 1
    else:
      # Rpu = pi_p * Rp + max{0, Rn} = pi_p * Rp + Rn
      output = self.prior * positive_risk + negative_risk
      x_out  = self.prior * positive_risk + negative_risk


    return output, x_out 

