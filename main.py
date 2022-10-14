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
    wandb.init(dir=config['log_directory'],
               project=config['project_name'], name=config['experiment_name'])
    wandb.config = config

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import imgviz, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
mpl.rcParams['figure.dpi'] = 300


from dataloader import GEN_DATA_LISTS, Cityscape
from data_utils import collate, pallet_cityscape
from model import UHDNext
from losses import FocalLoss, CrossEntropyLoss2d
from metrics import ConfusionMatrix
from lr_scheduler import LR_Scheduler
from utils import Trainer, Evaluator, ModelUtils
import torch.nn.functional as F

from gray2color import gray2color
g2c = lambda x : gray2color(x, use_pallet='cityscape', custom_pallet=pallet_cityscape)

data_lists = GEN_DATA_LISTS(config['data_dir'], config['sub_directories'])
train_paths, val_paths, test_paths = data_lists.get_splits()
classes = data_lists.get_classes()
data_lists.get_filecounts()

train_data = Cityscape(train_paths[0], train_paths[1], config['img_height'], config['img_width'],
                       config['Augment_data'], config['Normalize_data'])

train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True,
                          num_workers=config['num_workers'],
                          collate_fn=collate, pin_memory=config['pin_memory'],
                          prefetch_factor=2, persistent_workers=True)

val_data = Cityscape(val_paths[0], val_paths[1], config['img_height'], config['img_width'],
                     False, config['Normalize_data'])

val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False,
                        num_workers=config['num_workers'],
                        collate_fn=collate, pin_memory=config['pin_memory'],
                        prefetch_factor=2, persistent_workers=True)

# DataLoader Sanity Checks
batch = next(iter(train_loader))
s=255
img_ls = []
[img_ls.append((batch['img'][i]*s).astype(np.uint8)) for i in range(config['batch_size'])]
[img_ls.append(g2c(batch['lbl'][i])) for i in range(config['batch_size'])]
plt.title('Sample Batch')
plt.imshow(imgviz.tile(img_ls, shape=(2,config['batch_size']), border=(255,0,0)))
plt.axis('off')
#%%
model = UHDNext(num_classes=config['num_classes'], in_channnels=3, embed_dims=[32, 64, 460, 256],
                ffn_ratios=[4, 4, 4, 4], depths=[3, 3, 5, 2], num_stages=4,
                dec_outChannels=256, ls_init_val=float(config['layer_scaling_val']), 
                drop_path=float(config['stochastic_drop_path']), config=config)
                
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
model= nn.DataParallel(model)

loss = FocalLoss()
# loss = CrossEntropyLoss2d()
criterion = lambda x,y: loss(x, y)

optimizer = torch.optim.AdamW([{'params': model.parameters(),
                               'lr':config['learning_rate']}], weight_decay=0.0005)

# optimizer = torch.optim.Adam([{'params': model.parameters(),
#                                'lr':config['learning_rate']}],
#                                 weight_decay=config['WEIGHT_DECAY'])

scheduler = LR_Scheduler(config['lr_schedule'], config['learning_rate'], config['epochs'],
                         iters_per_epoch=len(train_loader), warmup_epochs=config['warmup_epochs'])

metric = ConfusionMatrix(config['num_classes'])


mu = ModelUtils(config['num_classes'], config['checkpoint_path'], config['experiment_name'])
mu.load_chkpt(model, optimizer)

trainer = Trainer(model, config['batch_size'], optimizer, criterion, metric)
evaluator = Evaluator(model, metric)

# Initializing plots
if config['LOG_WANDB']:
    wandb.watch(model, criterion, log="all", log_freq=10)
    wandb.log({"epoch": 0, "val_mIOU": 0, "loss": 10,
                "mIOU": 0, "learning_rate": 0})

#%%
epoch, best_iou, curr_viou = 0, 0, 0
total_avg_viou = []
for epoch in range(config['epochs']):

    pbar = tqdm(train_loader)
    model.train()
    ta, tl = [], []
    for step, data_batch in enumerate(pbar):

        scheduler(optimizer, step, epoch)
        loss_value = trainer.training_step(data_batch)
        iou = trainer.get_scores()
        trainer.reset_metric()
        
        tl.append(loss_value)
        ta.append(iou['iou_mean'])
        pbar.set_description(f'Epoch {epoch+1}/{config["epochs"]} - t_loss {loss_value:.4f} - mIOU {iou["iou_mean"]:.4f}')
    print(f'=> Average loss: {np.nanmean(tl)}, Average IoU: {np.nanmean(ta)}')

    if (epoch + 1) % 2 == 0: # eval every 2 epoch
        model.eval()
        va = []
        vbar = tqdm(val_loader)
        for step, val_batch in enumerate(vbar):
            with torch.no_grad():
                evaluator.eval_step(val_batch)
                viou = evaluator.get_scores()
                evaluator.reset_metric()

            va.append(viou['iou_mean'])
            vbar.set_description(f'Validation - v_mIOU {viou["iou_mean"]:.4f}')

        img, gt, pred = evaluator.get_sample_prediction()
        tiled = imgviz.tile([img, g2c(gt), g2c(pred)], shape=(1,3), border=(255,0,0))
        # plt.imshow(tiled)
        avg_viou = np.nanmean(va)
        total_avg_viou.append(avg_viou)
        curr_viou = np.nanmax(total_avg_viou)
        print(f'=> Averaged Validation IoU: {avg_viou:.4f}')

        if config['LOG_WANDB']:
            wandb.log({"epoch": epoch+1, "val_mIOU": viou["iou_mean"]})
            wandb.log({'predictions': wandb.Image(tiled)})

    if config['LOG_WANDB']:
        wandb.log({"epoch": epoch+1, "loss": loss_value, "mIOU": iou["iou_mean"],
                    "learning_rate": optimizer.param_groups[0]['lr']})
    
    tl.append(loss_value)
    ta.append(iou["iou_mean"])
    
    if curr_viou > best_iou:
        best_iou = curr_viou
        mu.save_chkpt(model, optimizer, epoch, loss, iou['iou_mean'])

if config['LOG_WANDB']:
    wandb.run.finish()
#%%