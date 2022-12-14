<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white" />  [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FMr-TalhaIlyas%2FSegNext&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
# SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation

![alt text](https://github.com/Visual-Attention-Network/SegNeXt/blob/main/resources/flops.png)
SegNeXt, a simple convolutional network architecture for semantic segmentation. Recent transformer-based models have dominated the field of semantic segmentation due to the efficiency of self-attention in encoding spatial information. In this paper, we show that convolutional attention is a more efficient and effective way to encode contextual information than the self-attention mechanism in transformers. By re-examining the characteristics owned by successful segmentation models, we discover several key components leading to the performance improvement of segmentation models. This motivates us to design a novel convolutional attention network that uses cheap convolutional operations. Without bells and whistles, our SegNeXt significantly improves the performance of previous state-of-the-art methods on popular benchmarks, including ADE20K, Cityscapes, COCO-Stuff, Pascal VOC, Pascal Context, and iSAID. Notably, SegNeXt outperforms EfficientNet-L2 w/ NAS-FPN and achieves 90.6% mIoU on the Pascal VOC 2012 test leaderboard using only 1/10 parameters of it. On average, SegNeXt achieves about 2.0% mIoU improvements compared to the state-of-the-art methods on the ADE20K datasets with the same or fewer computations.

[Original Paper](https://arxiv.org/abs/2209.08575)

## SegNext-Tiny Results on Cityscapes Dataset (From Scratch)

### mIOU Validation

![alt text](https://github.com/Mr-TalhaIlyas/SegNext/blob/master/screens/iou.png)

### Prediction

[Checkpoint](https://drive.google.com/file/d/1HgwcXNt2JGtG_n6AGQG2FWw5kgMpN_Yu/view?usp=share_link), for default settings in `main.py` script.

`img : gt : pred`

![alt text](https://github.com/Mr-TalhaIlyas/SegNext/blob/master/screens/media.png)
