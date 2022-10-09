import yaml
with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)
import torch.nn as nn
import torch.nn.functional as F
import torch

def one_hot(index, classes):
    #####################################################
    size = index.size()[:1] + (classes,) # jsut gets shape -> (P, C)
    view = index.size()[:1] + (1,) # get shapes -> (P, 1)
    #####################################################
    # makes a tensor of size (P, C) and fills it with zeros
    mask = torch.Tensor(size).fill_(0).to('cuda' if torch.cuda.is_available() else 'cpu') # (P, C)
    # reshapes the targets/index to (P, 1)
    index = index.view(view) 
    ones = 1.

    return mask.scatter_(1, index, ones) # places ones at those indexes (in mask) indicated by values in targets 


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7, size_average=True, one_hot=True, ignore=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.size_average = size_average
        self.one_hot = one_hot
        self.ignore = ignore

    def forward(self, input, target):
        '''
        only support ignore at 0
        '''
        B, C, H, W = input.size()
        input = input.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C => (P, C) 
        target = target.view(-1) # (P,)
        
        if self.ignore is not None:
            valid = (target != self.ignore)
            input = input[valid]
            target = target[valid]

        if self.one_hot:
            target = one_hot(target, C) # (P, C)

        probs = F.softmax(input, dim=1)
        probs = (probs * target).sum(1)
        probs = probs.clamp(self.eps, 1. - self.eps)

        log_p = probs.log()

        batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss