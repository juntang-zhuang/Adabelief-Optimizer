<h1 align="center">AdaBelief Optimizer</h1>
<h4 align="center">NeurIPS 2020 Spotlight, trains fast as Adam, generalizes well as SGD, and is stable to train GANs.</h4>

## Table of Contents
- [Introduction](#Installation-and-usage)
- [Reproduce results in the paper ](#Reproduce-results-in-the-paper)
- [Citation](#citation)
Propose an optimizer that trains fast as Adam, generalizes well as SGD, and is stable to train GANs. 

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
<img src="imgs/image_recog.png" width="70%"/>

### Results on GAN training
<img src="imgs/GAN.png" width="70%"/>

### Results on Toy Example
![](imgs/Beale2.gif)

# Citation
