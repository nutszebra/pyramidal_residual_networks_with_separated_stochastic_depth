# What's this
Implementation of Pyramidal Residual Networks With Separated Stochastic Depth by chainer  

# Dependencies

    git clone https://github.com/nutszebra/pyramidal_residual_networks_with_separated_stochastic_depth.git
    cd pyramidal_residual_networks_with_separated_stochastic_depth
    git submodule init
    git submodule update

# How to run
    python main.py -g 0

# Details about my implementation
All hyperparameters and network architecture are the same as in [[1]][Paper] except for some parts.  

* Data augmentation  
Train: Pictures are randomly resized in the range of [32, 36], then 32x32 patches are extracted randomly and are normalized locally. Horizontal flipping is applied with 0.5 probability.  
Test: Pictures are resized to 32x32, then they are normalized locally. Single image test is used to calculate total accuracy.  

* Multi-model learning  
Single model is used.


# Cifar10 result
| network                                   | alpha  | depth  | total accuracy (%) |
|:------------------------------------------|--------|--------|-------------------:|
| [[1]][Paper]                              | 150    | 182    | 96.69              |
| my implementation                         | 150    | 182    | soon               |


<img src="https://github.com/nutszebra/pyramidal_residual_networks_with_separated_stochastic_depth/blob/master/loss.jpg" alt="loss" title="loss">
<img src="https://github.com/nutszebra/pyramidal_residual_networks_with_separated_stochastic_depth/blob/master/accuracy.jpg" alt="total accuracy" title="total accuracy">

# References
Deep Pyramidal Residual Networks with Separated Stochastic Depth [[1]][Paper]

[paper]: https://arxiv.org/abs/1612.01230 "Paper"
