import yaml
with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)
import torch
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np

torch.backends.cudnn.deterministic = True

transformer = transforms.Compose([
                                 # this transfrom converts BHWC -> BCHW and 
                                 # also divides the image by 255 by default if values are in range 0..255.
                                 transforms.ToTensor(),
                                ])
stride = config['output_stride']
torch_resizer = transforms.Compose([transforms.Resize(size=(config['img_height']//stride, config['img_width']//stride),
                                                interpolation=transforms.InterpolationMode.NEAREST)])
torch_imgresizer = transforms.Compose([transforms.Resize(size=(config['img_height']//stride, config['img_width']//stride),
                                                interpolation=transforms.InterpolationMode.BILINEAR)])
def collate(batch):
    '''
    custom Collat funciton for collating individual fetched data samples into batches.
    '''
    
    img = [ b['img'] for b in batch ] # w, h
    lbl = [ b['lbl'] for b in batch ]
   
    return {'img': img, 'lbl': lbl}

normalize = lambda x, alpha, beta : (((beta-alpha) * (x-np.min(x))) / (np.max(x)-np.min(x))) + alpha
standardize = lambda x : (x - np.mean(x)) / np.std(x)

def std_norm(img, norm=True, alpha=0, beta=1):
    '''
    Standardize and Normalizae data sample wise
    alpha -> -1 or 0 lower bound
    beta -> 1 upper bound
    '''
    img = standardize(img)
    if norm:
        img = normalize(img, alpha, beta)
        
    return img

def _mask_transform(mask):
    target = np.array(mask).astype('int32')
    return target

def masks_transform(masks, numpy=False):
    '''
    masks: list of PIL images
    '''
    targets = []
    for m in masks:
        targets.append(_mask_transform(m))
    targets = np.array(targets) 
    if numpy:
        return targets
    else:
        return torch.from_numpy(targets).long().to('cuda' if torch.cuda.is_available() else 'cpu')

def images_transform(images):
    '''
    images: list of PIL images
    '''
    inputs = []
    for img in images:
        inputs.append(transformer(img))
    inputs = torch.stack(inputs, dim=0).float().to('cuda' if torch.cuda.is_available() else 'cpu')
    return inputs

def encode_labels(mask):
    label_mask = np.zeros_like(mask)
    for k in mapping_20:
        label_mask[mask == k] = mapping_20[k]
    return label_mask

mapping_20 = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 1,
    8: 2,
    9: 0,
    10: 0,
    11: 3,
    12: 4,
    13: 5,
    14: 0,
    15: 0,
    16: 0,
    17: 6,
    18: 0,
    19: 7,
    20: 8,
    21: 9,
    22: 10,
    23: 11,
    24: 12,
    25: 13,
    26: 14,
    27: 15,
    28: 16,
    29: 0,
    30: 0,
    31: 17,
    32: 18,
    33: 19,
    -1: 0
}

cityscape_class_names = ['background', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                        'traffic light', 'traffic sign',
                        'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                        'truck', 'bus', 'train', 'motorcycle', 'bicycle']

pallet_cityscape = np.array([[[0,0,0],
                            [128, 64, 128],
                            [244, 35, 232],
                            [70, 70, 70],
                            [102, 102, 156],
                            [190, 153, 153],
                            [153, 153, 153],
                            [250, 170, 30],
                            [220, 220, 0],
                            [107, 142, 35],
                            [152, 251, 152],
                            [70, 130, 180],
                            [220, 20, 60],
                            [255, 0, 0],
                            [0, 0, 142],
                            [0, 0, 70],
                            [0, 60, 100],
                            [0, 80, 100],
                            [0, 0, 230],
                            [119, 11, 32]]], np.uint8) / 255