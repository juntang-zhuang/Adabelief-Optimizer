<h1 align="center">AdaBelief Optimizer</h1>
<h4 align="center">NeurIPS 2020 Spotlight, trains fast as Adam, generalizes well as SGD, and is stable to train GANs.</h4>

## Table of Contents
- [External Links](#external-links)
- [Installation and usage](#Installation-and-usage)
- [Reproduce results in the paper ](#Reproduce-results-in-the-paper)
- [Discussions (Important, please read before using)](#Discussions)
- [Citation](#citation)

## External Links
Project page: https://juntang-zhuang.github.io/adabelief/ <br>
arXiv: <br>
reddit: <br>
twitter: <br>

## Installation and usage

### PyTorch implementations
#### AdaBelief
```
pip install adabelief-pytorch
```
```
from adabelief_pytorch import AdaBelief
optimizer = AdaBelief(model.parameters(), lr=1e-3, eps=1e-12, betas=(0.9,0.999))
```
#### Adabelief with Ranger optimizer
```
pip install ranger-adabelief
```
```
from ranger_adabelief import RangerAdaBelief
optimizer = RangerAdaBelief(model.parameters(), lr=1e-3, eps=1e-12, betas=(0.9,0.999))
```
### Tensorflow implementation
```
pip install adabelief-tf
```
```
from adabelief_tf impoty AdaBeliefOptimizer
optimizer = AdaBeliefOptimizer(learning_rate, epsilon=1e-12) 
```

## Reproduce results in the paper 
#### (Comparison with 8 other optimizers: SGD, Adam, AdaBound, RAdam, AdamW, Yogi, MSVAG, Fromage)
See folder ``PyTorch_Experiments``, for each subfolder, execute ```sh run.sh```
### Results on Image Recongnition 
<img src="./imgs/image_recog.png" width="70%"/> 

### Results on GAN training
<img src="./imgs/GAN.png" width="70%"/>

### Results on Toy Example
<img src="./imgs/Beale2.gif" width="70%"/>

## Discussions

#### Installation
Please instal the latest version from pip, old versions might suffer from bugs. Source code for up-to-date package is available in folder ```pypi_packages```. 
#### Details to reproduce results
* Results in the paper are generated using the PyTorch implementation in ```adabelief-pytorch``` package. This is the __ONLY__ package that I have extensively tested for now.
* We also provide a modification of ```ranger``` optimizer in ```ranger-adavelief``` which combines ```RAdam + LookAhead + Gradient Centralization + AdaBelief```, but this is not used in the paper and is not extensively tested. 
* The ```adabelief-tf``` is a naive implementation in Tensorflow. It lacks many features such as ```decoupled weight decay```, and is not extensively tested. Please contact me if you want to collaborate and improve it.
#### Discussion on algorithms
* __Weight Decay__: 
** Decoupling:
   Currently there are two ways to perform weight decay for adaptive optimizers, directly apply it to the gradient (Adam), or ```decouple``` weight decay from gradient descent (AdamW). This is 

## Citation
