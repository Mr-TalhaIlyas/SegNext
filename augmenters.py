# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 04:05:58 2022

@author: talha
"""
#%%
import yaml
with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)
from imgaug import augmenters as iaa
import imgaug as ia
import numpy as np
import random, copy, cv2



def add_to_contrast(images, random_state, parents, hooks):
    '''
    A custom augmentation function for iaa.aug library
    The randorm_state, parents and hooks parameters come
    form the lamda iaa lib**
    '''
    
    images[0] = images[0].astype('float')
    img = images
    value = random_state.uniform(0.75, 1.25)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    ret = img[0] * value + mean * (1 - value)
    ret = np.clip(img, 0, 255)
    ret = ret.astype(np.uint8)
    return ret


# Define the Augmentor Sequence
# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
# SomeOf will only apply only specifiel augmnetations on the data like SomeOf(2, ...)
# will only apply the 2 augmentaiton form the seq. list.

sometimes = lambda aug: iaa.Sometimes(0.80, aug)  

# apply on images and masks
Geometric = iaa.Sequential(
    [
    # apply only 2 of the following
    iaa.SomeOf(2, [
    # apply only 1 of following
    # iaa.OneOf([
        sometimes(iaa.Fliplr(0.9)),
        sometimes(iaa.Flipud(0.9)),
        sometimes(iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, order=0, backend="cv2")),
        sometimes(iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                             order=0, backend="cv2")), #, cval=(0, 255)
        sometimes(iaa.Affine(rotate=(-25, 25), order=0, backend="cv2")),
        sometimes(iaa.Cutout(fill_mode="constant", cval=0, nb_iterations=(1,2),#(0, 255)
                             fill_per_channel=0.5)),
        sometimes(iaa.Affine(shear=(-8, 8), order=0, backend="cv2")),
        sometimes(iaa.KeepSizeByResize(
                              iaa.Crop(percent=(0.05, 0.25), keep_size=False),
                              interpolation='nearest')),
        ], random_order=True),
    ], random_order=True)

# only apply on images
Noise = iaa.Sequential(
            [
            iaa.OneOf(
                [   
                # Blur each image using a median over neihbourhoods that have a random size between 3x3 and 7x7
                sometimes(iaa.MedianBlur(k=(3, 7))),
                # blur images using gaussian kernels with random value (sigma) from the interval [a, b]
                sometimes(iaa.GaussianBlur(sigma=(0.0, 1.0))),
                sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)),
                sometimes(iaa.ChannelShuffle(p=0.9))
                ]
            ),
            iaa.Sequential(
                [
                sometimes(iaa.AddToHue((-8, 8))),
                sometimes(iaa.AddToSaturation((-20, 20))),
                sometimes(iaa.AddToBrightness((-26, 26))),
                sometimes(iaa.Lambda(func_images = add_to_contrast))
                ], random_order=True)
            ], random_order=True)

def geomatric_augs(img_patch, lbl_patch):
    #print('Geomatric AUG')
    geom_aug = Geometric._to_deterministic() #This line is very important because without this one each pair of image and mask will get augumented differently
    
    img_patch = geom_aug.augment_image(img_patch)
    lbl_patch = geom_aug.augment_image(lbl_patch)

    return img_patch, lbl_patch

def noise_augs(img_patch, lbl_patch):
    #print('Noise AUG')
    nois_aug = Noise._to_deterministic()
    img_patch = nois_aug.augment_image(img_patch)

    return img_patch, lbl_patch


def data_augmenter(img_patch, lbl_patch):
    
    func_args = [
                (geomatric_augs, (img_patch, lbl_patch)),
                (noise_augs,     (img_patch, lbl_patch)),
                ]
    #print('one iter done')
    (func, args), = random.choices(func_args, weights=[0.7, 0.3])
    img_patch, lbl_patch = func(*args)                

    return img_patch, lbl_patch


