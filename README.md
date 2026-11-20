# ConVIRT - Contrastive Learning Representations of Images and Text pairs

### Contrastive VIsual Representation Learning from Text

Deep neural networks learn from a large amount of data to obtain the correct parameters to perform a speciﬁc task. However, in practice, we often encounter a problem: **insuﬃcient amount of labeled data**. However, if your data contains pairs of images and text, you can solve the problem with contrastive learning.

Contrastive learning is a kind of self-supervised learning method. It does not require specialized labels, but rather a method to learn the correct parameters from the unlabeled data itself. It aims to learn an encoder that makes the encoding results of similar classes of data similar and makes the encoding results of diﬀerent classes of data as diﬀerent as possible. Typical contrast learning is done based on comparisons between two images. However, if we have paired image and text data, contrast learning can also be applied between images and text.

The repository is a PyTorch implementation of the architecture descibed in the ConVIRT paper: [_Contrastive Learning of Medical Visual Representations from Paired Images and Text_](https://arxiv.org/abs/2010.00747). The authors of paper are Yuhao Zhang, Hang Jiang, Yasuhide Miura, Christopher D. Manning, Curtis P. Langlotz.

This is a fork of [edreisMD/ConVIRT-pytorch](https://github.com/edreisMD/ConVIRT-pytorch), the non-official open source implementation by [Eduardo Reis](https://github.com/edreisMD), which was itself forked and modified from [sthalles/SimCLR](https://github.com/sthalles/SimCLR).

### My contributions

Merged upstream in [edreisMD/ConVIRT-pytorch#8](https://github.com/edreisMD/ConVIRT-pytorch/pull/8):

- Replaced NVIDIA Apex with native PyTorch automatic mixed precision (AMP)
- Fixed a half-precision type error, and replaced the deprecated `transforms.Scale` with `transforms.Resize`

Added in this fork, for pretraining on MIMIC-CXR chest X-rays and reports:

- The text encoder is now trained: pooling no longer runs under `torch.no_grad()`, so gradients reach BERT (its embedding layer stays frozen), and sentences are pooled with max pooling over the attention mask
- The image encoder starts from ImageNet-pretrained ResNet weights
- Report text is cleaned up and split into sentences without splitting after list numbers such as "1."
- Augmentations suited to X-rays: random affine, brightness and contrast jitter, and Gaussian blur, without random grayscale
- Learning rate drops on a validation-loss plateau, and training loss and learning rate are logged per epoch

References:

- Yuhao Zhang et al. Contrastive Learning of Medical Visual Representations from Paired Images and Text. https://arxiv.org/pdf/2010.00747.pdf
- Ting Chen et al. A Simple Framework for Contrastive Learning of Visual Representations. https://arxiv.org/abs/2002.05709
- https://github.com/sthalles/SimCLR
- https://github.com/google-research/simclr
- https://github.com/google-research/bert
- https://github.com/hanxiao/bert-as-service#q-what-are-the-available-pooling-strategies
