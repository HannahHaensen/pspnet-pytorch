# pspnet-pytorch
PyTorch implementation of PSPNet segmentation network


### Original paper

 [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105)
 
### Details

- Note that a Dilated Resnet is used 
    - dilated convolutions are used to increase the receptive field of the higher layers, compensating for the reduction in receptive field induced by removing subsampling.
    - see @ [DRN](https://towardsdatascience.com/review-drn-dilated-residual-networks-image-classification-semantic-segmentation-d527e1a8fb5)

- Zoom Faktor in the end in original repo -> here interpolation back to original size

### Usage 




