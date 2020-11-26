# pspnet-pytorch
PyTorch implementation of PSPNet segmentation network


### Original paper

 [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105)
 
### Details

This is a slightly different version - instead of direct 8x upsampling at the end I use three consequitive upsamplings for stability. 

### Feature extraction

### Usage 

To follow the training routine in train.py you need a DataLoader that yields the tuples of the following format:

(Bx3xHxW FloatTensor x, BxHxW LongTensor y, BxN LongTensor y\_cls) where

x - batch of input images,

y - batch of groung truth seg maps,

y\_cls - batch of image size


