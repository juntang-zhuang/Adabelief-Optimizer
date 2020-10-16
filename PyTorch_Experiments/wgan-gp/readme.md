### AdaBelief opitmizer: Adapting stepsizes by the belief in observed gradients

This repository contains code to reproduce results for submission 46 to NeurIPS 2020, "AdaBelief opitmizer: Adapting stepsizes by the belief in observed gradients".


### Dependencies
python 3.7
pytorch 1.1.0
torchvision 0.3.0
AdaBound  (Please instal by "pip install adabound")


### Training and evaluation code

CUDA_VISIBLE_DEVICES=0 python main.py --optimizer adabelief --eps 1e-12 --Train --dataset cifar10 

--optim: name of optimizers, choices include ['sgd', 'adam', 'adamw', 'adabelief', 'yogi', 'msvag', 'radam', 'fromage', 'adabound', 'rmsprop']
--lr: learning rate
--eps: epsilon value used for optimizers

The code will automatically generate a folder containing model weights, a csv file containing the FID score, and a separate folder containing 64,000 fake images


### Running time
On a single GTX 1080 GPU, training a one round takes 1~2 hours for a single optimzer. To run all experiments would take 2 hours x 10 optimizers x 5 repeats = 100 hours
