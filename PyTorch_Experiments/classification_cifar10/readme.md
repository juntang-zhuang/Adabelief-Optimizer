### AdaBelief opitmizer: Adapting stepsizes by the belief in observed gradients

This repository contains code to reproduce results for submission 46 to NeurIPS 2020, "AdaBelief opitmizer: Adapting stepsizes by the belief in observed gradients".

This repo heavily depends on the official implementation of AdaBound: https://github.com/Luolc/AdaBound
The main difference is that we provide the code for different optimizers, including: AdaBlief, SGD, AdaBound, Yogi, Adam, MSVAG, RAdam, AdamW, Fromage.
Code for optimizers in the literature are forked from public implementations, please see the comments in corresponding files in the folder "optimizers".



### Dependencies
python 3.7
pytorch 1.1.0
torchvision 0.3.0
jupyter notebook
AdaBound  (Please instal by "pip install adabound")



### Visualization of pre-trained curves
Please use the jupyter notebook "visualization.ipynb" to visualize the training and test curves of different optimizers. We provide logs for pre-trained models (9 optimizers x 3 models = 27 pre-trained curves) in the folder "curve".



### Training and evaluation code

(1) train network with
CUDA_VISIBLE_DEVICES=0 python main.py --optim adabelief --lr 1e-3 --eps 1e-8 --beta1 0.9 --beta2 0.999 --momentum 0.9

--optim: name of optimizers, choices include ['sgd', 'adam', 'adamw', 'adabelief', 'yogi', 'msvag', 'radam', 'fromage', 'adabound',]
--lr: learning rate
--eps: epsilon value used for optimizers. Note that Yogi uses a default of 1e-03, other optimizers typically uses 1e-08
--beta1, --beta2: beta values in adaptive optimizers
--momentum: momentum used for SGD.s

(2) visualize using the notebook "visualization.ipynb"



### Running time
On a single GTX 1080 GPU, training a ResNet typically takes 4~5 hours for a single optimzer. To run all experiments would take 4 hours x 9 optimizers x 3 models = 108 hours
