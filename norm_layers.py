#%%
import yaml
with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

import math
import torch
from functools import partial
from torch import nn
import torch.nn.functional as F
from sync_bn.nn.modules import SynchronizedBatchNorm2d

norm_layer = partial(SynchronizedBatchNorm2d, momentum=float(config['SyncBN_MOM']))

class myLayerNorm(nn.Module):
    def __init__(self, inChannels):
        super().__init__()
        self.norm == nn.LayerNorm(inChannels, eps=1e-5)

    def forward(self, x):
        # reshaping only to apply Layer Normalization layer
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1,2) # B*C*H*W -> B*C*HW -> B*HW*C
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # B*HW*C -> B*H*W*C -> B*C*H*W

        return x


class NormLayer(nn.Module):
    def __init__(self, inChannels, norm_type=config['norm_typ']):
        super().__init__()
        if norm_type == 'batch_norm':
            # print('Adding Batch Norm layer') # for testing
            self.norm = nn.BatchNorm2d(inChannels, eps=1e-5, momentum=float(config['BN_MOM']))
        elif norm_type == 'sync_bn':
            # print('Adding Sync-Batch Norm layer') # for testing
            self.norm = norm_layer(inChannels)
        elif norm_type == 'layer_norm':
            # print('Adding Layer Norm layer') # for testing
            self.norm == nn.myLayerNorm(inChannels)
        else:
            raise NotImplementedError

    def forward(self, x):

        x = self.norm(x)
        
        return x
    
