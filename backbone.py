#%%
from turtle import forward
from black import out
from cv2 import detail_SphericalProjector
from numpy import require
import torch
import torch.nn as nn
import math
import warnings
from torch.nn.modules.utils import _pair as to_2tuple

class DWConv3x3(nn.Module):
    '''Depth wise conv'''
    def __init__(self, dim=768):
        super(DWConv3x3, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

class StemConv(nn.Module):
    '''following ConvNext paper'''
    def __init__(self, in_channels, out_channels, bn_momentum=0.99):
        super(StemConv, self).__init__()

        self.proj = nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels//2,
                                                kernel_size=(3,3), stride=(2,2), padding=(1,1)),
                                    nn.BatchNorm2d(out_channels//2, eps=1e-5, momentum=bn_momentum),
                                    nn.GELU(),
                                    nn.Conv2d(out_channels//2, out_channels,
                                                kernel_size=(3,3), stride=(2,2), padding=(1,1)),
                                    nn.BatchNorm2d(out_channels, eps=1e-5, momentum=bn_momentum)
                                )
    
    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.size()
        # x = x.flatten(2).transpose(1,2) # B*C*H*W -> B*C*HW -> B*HW*C
        return x, H, W

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


class FFN(nn.Module):
    '''following ConvNext paper'''
    def __init__(self, in_channels, out_channels, hid_channels, dropout=0.):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, hid_channels, 1)
        self.dwconv = DWConv3x3(hid_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hid_channels, out_channels, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):

        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class BlockFFN(nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels, dropout=0.):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels, eps=1e-5, momentum=0.99)
        self.ffn = FFN(in_channels, out_channels, hid_channels, dropout)

    def forward(self, x):
        skip = x.clone()

        x = self.bn(x)
        x = self.ffn(x)

        op = skip + x
        return op




class MSCA(nn.Module):

    def __init__(self, dim):
        super(MSCA, self).__init__()
        # input
        self.conv55 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim) 
        # split into multipats of multiscale attention
        self.conv17_0 = nn.Conv2d(dim, dim, (1,7), padding=(0, 3), groups=dim)
        self.conv17_1 = nn.Conv2d(dim, dim, (7,1), padding=(3, 0), groups=dim)

        self.conv111_0 = nn.Conv2d(dim, dim, (1,11), padding=(0,5), groups=dim)
        self.conv111_1 = nn.Conv2d(dim, dim, (11,1), padding=(5,0), groups=dim)

        self.conv211_0 = nn.Conv2d(dim, dim, (1,21), padding=(0, 10), groups=dim)
        self.conv211_1 = nn.Conv2d(dim, dim, (21,1), padding=(10, 0), groups=dim)

        self.conv11 = nn.Conv2d(dim, dim, 1) # channel mixer

    def forward(self, x):
        
        skip = x.clone()

        c55 = self.conv55(x)
        c17 = self.conv17_0(x)
        c17 = self.conv17_1(c17)
        c111 = self.conv111_0(x)
        c111 = self.conv111_1(c111)
        c211 = self.conv211_0(x)
        c211 = self.conv211_1(c211)

        add = c55 + c17 + c111 + c211

        mixer = self.conv11(add)

        op = mixer * skip

        return op

class BlockMSCA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bn = nn.BatchNorm2d(dim, eps=1e-5, momentum=0.99)
        self.proj1 = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()
        self.msca = MSCA(dim)
        self.proj2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):

        skip = x.clone()

        x = self.bn(x)
        x = self.proj1(x)
        x = self.act(x)
        x = self.msca(x)
        x = self.proj2(x)

        out = x + skip

        return out


class StageMSCA(nn.Module):
    def __init__(self, dim, ffn_ratio=4., dropout=0., bn_momentum=0.99):
        super().__init__()
        self.msca_block = BlockMSCA(dim)

        ffn_hid_dim = int(dim * ffn_ratio)
        self.ffn_block = BlockFFN(in_channels=dim, out_channels=dim,
                                    hid_channels=ffn_hid_dim, dropout=dropout)

    def forward(self, x): # input coming form Stem
        # B, N, C = x.shape
        # x = x.permute()
        x = self.msca_block(x)
        x = self.ffn_block(x)

        return x

class MSCANet(nn.Module):
    def __init__(self, in_channnels=3, embed_dims=[32, 64, 460,256],
                 ffn_ratios=[4, 4, 4, 4], depths=[3,3,5,2], dropout=0.,
                 num_stages = 4):
        super(MSCANet, self).__init__()

        self.depths = depths
        self.num_stages = num_stages

        cur = 0

        for i in range(num_stages):
            if i == 0:
                input_embed = StemConv(in_channnels, embed_dims[0])
            else:
                input_embed = DownSample(in_channels=embed_dims[i-1], embed_dim=embed_dims[i])
            
            stage = nn.ModuleList([StageMSCA(dim=embed_dims[i], ffn_ratio=ffn_ratios[i], dropout=0.,)
                                for j in range(depths[i])])

            layer_norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f'input_embed{i+1}', input_embed)
            setattr(self, f'stage{i+1}', stage)
            setattr(self, f'layer_norm{i+1}', layer_norm)

        self.init_weights()

        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, val=1.0)
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, std=math.sqrt(2.0/fan_out), mean=0)
                # xavier_uniform_() tf default

    def forward(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            input_embed = getattr(self, f'input_embed{i+1}')
            stage = getattr(self, f'stage{i+1}')
            layer_norm = getattr(self, f'layer_norm{i+1}')
            
            x, H, W = input_embed(x)
            
            for stg in stage:
                x = stg(x)
            
            # reshaping only to apply Layer Normalization layer
            x = x.flatten(2).transpose(1,2) # B*C*H*W -> B*C*HW -> B*HW*C
            x = layer_norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # B*HW*C -> B*H*W*C -> B*C*H*W

            outs.append(x)

        return outs


#%%
# from torchsummary import summary
# model = MSCANet(in_channnels=3, embed_dims=[32, 64, 460,256],
#                  ffn_ratios=[4, 4, 4, 4], depths=[3,3,5,2], dropout=0.,
#                  num_stages = 4)
# # summary(model, (3,1024,2048))


# y = torch.randn((6,3,1024,2048))#.to('cuda' if torch.cuda.is_available() else 'cpu')
# x = model.forward(y)

# for i in range(4):
#     print(x[i].shape)
# %%
