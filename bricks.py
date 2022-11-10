import yaml
with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

import torch
import torch.nn as nn
import torch.nn.functional as F
from sync_bn.nn.modules import SynchronizedBatchNorm2d
from functools import partial

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
        self.inChannels = inChannels
        self.norm_type = norm_type
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
    
    def __repr__(self):
        return f'{self.__class__.__name__}(dim={self.inChannels}, norm_type={self.norm_type})'

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class LayerScale(nn.Module):
    '''
    Layer scale module.
    References:
      - https://arxiv.org/abs/2103.17239
    '''
    def __init__(self, inChannels, init_value=1e-2):
        super().__init__()
        self.inChannels = inChannels
        self.init_value = init_value
        self.layer_scale = nn.Parameter(init_value * torch.ones((inChannels)), requires_grad=True)

    def forward(self, x):
        if self.init_value == 0.0:
            return x
        else:
            scale = self.layer_scale.unsqueeze(-1).unsqueeze(-1) # C, -> C,1,1
            return scale * x

    def __repr__(self):
        return f'{self.__class__.__name__}(dim={self.inChannels}, init_value={self.init_value})'

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def stochastic_depth(input: torch.Tensor, p: float,
                     mode: str, training: bool =  True):
    
    if not training or p == 0.0:
        # print(f'not adding stochastic depth of: {p}')
        return input
    
    survival_rate = 1.0 - p
    if mode == 'row':
        shape = [input.shape[0]] + [1] * (input.ndim - 1) # just converts BXCXHXW -> [B,1,1,1] list
    elif mode == 'batch':
        shape = [1] * input.ndim

    noise = torch.empty(shape, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
    # print(f'added sDepth of: {p}')
    return input * noise

class StochasticDepth(nn.Module):
    '''
    Stochastic Depth module.
    It performs ROW-wise dropping rather than sample-wise. 
    mode (str): ``"batch"`` or ``"row"``.
                ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
                randomly selected rows from the batch.
    References:
      - https://pytorch.org/vision/stable/_modules/torchvision/ops/stochastic_depth.html#stochastic_depth
    '''
    def __init__(self, p=0.5, mode='row'):
        super().__init__()
        self.p = p
        self.mode = mode
    
    def forward(self, input):
        return stochastic_depth(input, self.p, self.mode, self.training)
    
    def __repr__(self):
       s = f"{self.__class__.__name__}(p={self.p})"
       return s

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 
def resize(input,
           size=None,
           scale_factor=None,
           mode='bilinear',
           align_corners=None,
           warning=True):

    return F.interpolate(input, size, scale_factor, mode, align_corners)


class DownSample(nn.Module):
    def __init__(self, kernelSize=3, stride=2, in_channels=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=(kernelSize, kernelSize),
                              stride=stride, padding=(kernelSize//2, kernelSize//2))
        # stride 4 => 4x down sample
        # stride 2 => 2x down sample
    def forward(self, x):

        x = self.proj(x)
        B, C, H, W = x.size()
        # x = x.flatten(2).transpose(1,2)
        return x, H, W
    
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class DWConv3x3(nn.Module):
    '''Depth wise conv'''
    def __init__(self, dim=768):
        super(DWConv3x3, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

class ConvBNRelu(nn.Module):

    @classmethod
    def _same_paddings(cls, kernel):
        if kernel == 1:
            return 0
        elif kernel == 3:
            return 1

    def __init__(self, inChannels, outChannels, kernel=3, stride=1, padding='same',
                 dilation=1, groups=1):
        super().__init__()

        if padding == 'same':
            padding = self._same_paddings(kernel)
        
        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=kernel,
                              padding=padding, stride=stride, dilation=dilation,
                              groups=groups, bias=False)
        self.norm = NormLayer(outChannels, norm_type=config['norm_typ'])
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x

class SeprableConv2d(nn.Module):
    def __init__(self, inChannels, outChannels, kernal_size=3, bias=False):
        self.dwconv = nn.Conv2d(inChannels, inChannels, kernal_size=kernal_size,
                                groups=inChannels, bias=bias)
        self.pwconv = nn.Conv2d(inChannels, inChannels, kernal_size=1, bias=bias)

    def forward(self, x):

        x = self.dwconv(x)
        x = self.pwconv(x)
        
        return x

class ConvRelu(nn.Module):
    def __init__(self, inChannels, outChannels, kernel=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=kernel, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv(x)
        x = self.act(x)
        
        return x
