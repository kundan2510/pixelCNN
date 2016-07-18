# Generating images pixel by pixel
### This repository contains code for training an image generator using the pixelCNN architecture as described in [Conditional Image Generation with PixelCNN Decoders](https://arxiv.org/abs/1606.05328)

Most of the code is in core theano. 'keras' has been used for loading data. Optimizer implementation from 'lasagne' has been used.

Dependencies:

[theano](http://deeplearning.net/software/theano/install.html)

[lasagne](http://lasagne.readthedocs.io/en/latest/user/installation.html)

[keras](http://keras.io/#getting-started-30-seconds-to-keras)

Generated images

![Generated images](output/generated_only_images.jpg)

Training images

![Training images](output/training_images.jpg)













Salient features: No blind spots, efficient implemenattion of vertical stacks and horizontal stacks and good generation results :D

TODO: Implement gated activation and conditional generation.
