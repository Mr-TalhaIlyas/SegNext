#%%
import yaml, math, os
with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)
import torch
import torch.nn.functional as F
import torch.nn as nn

from backbone import MSCANet
from decoder import DecoderHead


class UHDNext(nn.Module):
    def __init__(self, num_classes, in_channnels=3, embed_dims=[32, 64, 460, 256],
                 ffn_ratios=[4, 4, 4, 4], depths=[3, 3, 5, 2], dropout=0.,
                 num_stages=4, dec_outChannels=256, config=config):
        super().__init__()
        self.cls_conv = nn.Conv2d(
            dec_outChannels, num_classes, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.encoder = MSCANet(in_channnels=in_channnels, embed_dims=embed_dims,
                               ffn_ratios=ffn_ratios, depths=depths, dropout=dropout,
                               num_stages=num_stages)
        self.decoder = DecoderHead(
            outChannels=dec_outChannels, config=config, enc_embed_dims=embed_dims)
    
        self.init_weights()

        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1.0)
                nn.init.constant_(m.bias, val=0.0)
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, std=math.sqrt(2.0/fan_out), mean=0)
                # xavier_uniform_() tf default

    def forward(self, x):

        enc_feats = self.encoder(x)
        dec_out = self.decoder(enc_feats)
        out = self.cls_conv(dec_out)
        preds = self.softmax(out)

        return preds

# model = UHDNext(num_classes=34, in_channnels=3, embed_dims=[32, 64, 460, 256],
#                  ffn_ratios=[4, 4, 4, 4], depths=[3, 3, 5, 2], dropout=0.,
#                  num_stages=4, dec_outChannels=256, config=config)
# model = model.to('cuda')
# x = torch.randn((1,3,1024,2048)).to('cuda')
# y = model.forward(x)

# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print name, param.data




