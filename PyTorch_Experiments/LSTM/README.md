# LSTM and QRNN Language Model Toolkit

This repository contains the code used for two [Salesforce Research](https://einstein.ai/) papers:
+ [Regularizing and Optimizing LSTM Language Models](https://arxiv.org/abs/1708.02182)
+ [An Analysis of Neural Language Modeling at Multiple Scales](https://arxiv.org/abs/1803.08240)
This code was originally forked from the [PyTorch word level language modeling example](https://github.com/pytorch/examples/tree/master/word_language_model).

The model comes with instructions to train:
+ word level language models over the Penn Treebank (PTB), [WikiText-2](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset) (WT2), and [WikiText-103](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset) (WT103) datasets

+ character level language models over the Penn Treebank (PTBC) and Hutter Prize dataset (enwik8)

The model can be composed of an LSTM or a [Quasi-Recurrent Neural Network](https://github.com/salesforce/pytorch-qrnn/) (QRNN) which is two or more times faster than the cuDNN LSTM in this setup while achieving equivalent or better accuracy.

+ Install PyTorch >=0.4
+ Run `getdata.sh` to acquire the Penn Treebank and WikiText-2 datasets
+ Train the base model using `main.py`
+ (Optionally) Finetune the model using `finetune.py`
+ (Optionally) Apply the [continuous cache pointer](https://arxiv.org/abs/1612.04426) to the finetuned model using `pointer.py`

## Update (June/13/2018)

The codebase is now PyTorch 0.4 compatible for most use cases (a big shoutout to https://github.com/shawntan for a fairly comprehensive PR https://github.com/salesforce/awd-lstm-lm/pull/43). Mild readjustments to hyperparameters may be necessary to obtain quoted performance. If you desire exact reproducibility (or wish to run on PyTorch 0.3 or lower), we suggest using an older commit of this repository. We are still working on `pointer`, `finetune` and `generate` functionalities.

## Software Requirements

Python 3 and PyTorch>=0.4 are required for the current codebase.

Included below are hyper parameters to get equivalent or better results to those included in the original paper.

If you need to use an earlier version of the codebase, the original code and hyper parameters accessible at the [PyTorch==0.1.12](https://github.com/salesforce/awd-lstm-lm/tree/PyTorch%3D%3D0.1.12) release, with Python 3 and PyTorch 0.1.12 are required.
If you are using Anaconda, installation of PyTorch 0.1.12 can be achieved via:
`conda install pytorch=0.1.12 -c soumith`.

## Experiments

The codebase was modified during the writing of the paper, preventing exact reproduction due to minor differences in random seeds or similar.
We have also seen exact reproduction numbers change when changing underlying GPU.
The guide below produces results largely similar to the numbers reported.

For data setup, run `./getdata.sh`.
This script collects the Mikolov pre-processed Penn Treebank and the WikiText-2 datasets and places them in the `data` directory.

Next, decide whether to use the QRNN or the LSTM as the underlying recurrent neural network model.
The QRNN is many times faster than even Nvidia's cuDNN optimized LSTM (and dozens of times faster than a naive LSTM implementation) yet achieves similar or better results than the LSTM for many word level datasets.
At the time of writing, the QRNN models use the same number of parameters and are slightly deeper networks but are two to four times faster per epoch and require less epochs to converge.

The QRNN model uses a QRNN with convolutional size 2 for the first layer, allowing the model to view discrete natural language inputs (i.e. "New York"), while all other layers use a convolutional size of 1.

**Finetuning Note:** Fine-tuning modifies the original saved model `model.pt` file - if you wish to keep the original weights you must copy the file.

**Pointer note:** BPTT just changes the length of the sequence pushed onto the GPU but won't impact the final result.

