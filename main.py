#%%
from pickletools import optimize
import yaml
with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus_to_use'];

if config['LOG_WANDB']:
    import wandb
    wandb.init(project=config['project_name'], name=config['experiment_name'])
    wandb.config = config

import torch
from torch.utils.data import DataLoader

import imgviz, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = config['DPI']
from gray2color import gray2color
g2c = lambda x : gray2color(x, use_pallet='cityscape')

from dataloader import GEN_DATA_LISTS, Cityscape
from data_utils import collate
from model import UHDNext, ModelUtils
from losses import FocalLoss
from metrics import ConfusionMatrix
from lr_scheduler import LR_Scheduler

import torch.nn.functional as F

data_lists = GEN_DATA_LISTS(config['data_dir'], config['sub_directories'])
train_paths, val_paths, test_paths = data_lists.get_splits()
classes = data_lists.get_classes()
data_lists.get_filecounts()

train_data = Cityscape(train_paths[0], train_paths[1], config['img_height'], config['img_width'],
                       config['Augment_data'], config['Normalize_data'])

train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True,
                          num_workers=config['num_workers'],
                          collate_fn=collate, pin_memory=config['pin_memory'],
                          prefetch_factor=2, persistent_workers=False)

batch = next(iter(train_loader))


s=1#255
img_ls = []
[img_ls.append((batch['img'][i]*s).astype(np.uint8)) for i in range(config['batch_size'])]
[img_ls.append(g2c(batch['lbl'][i])) for i in range(config['batch_size'])]
plt.title('Sample Batch')
plt.imshow(imgviz.tile(img_ls, shape=(2,config['batch_size']), border=(255,0,0)))
plt.axis('off')
#%%
model = UHDNext(num_classes=34, in_channnels=3, embed_dims=[32, 64, 460, 256],
                ffn_ratios=[4, 4, 4, 4], depths=[3, 3, 5, 2], dropout=0.,
                num_stages=4, dec_outChannels=256, config=config)
                
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

loss = FocalLoss(gamma=2)
criterion = lambda x,y: loss(x, y)

optimizer = torch.optim.Adam([{'params': model.module.parameters(),
                         'lr':config['learning_rate']}], weight_decay=config['WEIGHT_DECAY'])

scheduler = LR_Scheduler(config['lr_schedule'], config['learning_rate'], config['Epoch'],
                         iters_per_epoch=len(train_loader), warmup_epochs=config['warmup_epochs'])

metric = ConfusionMatrix(config['num_classes'])


mu = ModelUtils(config['num_classes'], config['checkpoint_path'], config['experiment_name'])
model.load_chkpt(model, optimizer)
