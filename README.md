<h1 align="center">AdaBelief Optimizer</h1>
<h3 align="center">NeurIPS 2020 Spotlight, trains fast as Adam, generalizes well as SGD, and is stable to train GANs.</h3>

## Table of Contents
- [External Links](#external-links)
- [Quick Guide](#quick-guide)
- [Update Plan](#update-plan)
- [Installation and usage](#Installation-and-usage)
- [A quick look at the algorithm](#a-quick-look-at-the-algorithm)
- [Detailed Discussions](#Discussions) (Important) Please read the discussion, important info on hyper-params there.
- [Reproduce results in the paper ](#Reproduce-results-in-the-paper)
- [Citation](#citation)

## External Links
<a href="https://juntang-zhuang.github.io/adabelief/"> Project Page</a>, <a href="https://arxiv.org/abs/2010.07468"> arXiv </a>, <a href="https://www.reddit.com/r/MachineLearning/comments/jc1fp2/r_neurips_2020_spotlight_adabelief_optimizer">Reddit </a>, <a href="https://twitter.com/JuntangZhuang/status/1316934184607354891">Twitter</a>

## Quick Guide
AdaBelief uses a different denominator from Adam, and is orthogonal to other techniques such as recification, decoupled weight decay, weight averaging et.al.
This implies when you use some techniques with Adam, to get a good result with AdaBelief you might still need those techniques.

* ```epsilon``` in AdaBelief plays a different role as in Adam, typically when you use ```epslison=x``` in Adam, using ```epsilon=x*x``` will give similar results in AdaBelief. The default value ```epsilon=1e-8``` is not a good option in many cases, will modify it later to 1e-12 or 1e-16 later.

* If you task needs a "non-adaptive" optimizer, which means SGD performs much better than Adam(W), such as on image recognition, you need to set a large ```epsilon```(e.g. 1e-8,1e-10) for AdaBelief to make it more ```non-adaptive```; if your task needs a really ```adaptive``` optimizer, which means Adam is much better than SGD, such as GAN, then the recommended ```epsilon``` for AdaBelief is small (1e-12, 1e-16 ...).

* If decoupled weight decay is very important for your task, which means AdamW is much better than Adam, then you need to set ```weight_decouple``` as True to turn on decoupled decay in AdaBelief. Note that many optimizers uses decoupled weight decay without specifying it as an options, e.g. RAdam, but we provide it as an option so users are aware of what technique is actually used.

* Don't use "gradient threshold" (clamp each element independently) in AdaBelief, it could result in division by 0 and explosion in update; but "gradient clip" (shrink amplitude of the gradient vector but keeps its direction) is fine, though from my limited experience sometimes the clip range needs to be the same or larger than Adam.

## Update Plan
### To do
* <del>Someone (under the wechat group Jiqizhixin) points out that the results on GAN is bad, this might be due to the choice of GAN model (We pick the simplest code example from PyTorch docs without adding more tricks), and we did not perform cherry-picking or worsen the baseline perfomance intentionally. We will update results on new GANs (e.g. SN-GAN) and release code later. </del> 
* <del> Upload code for LSTM experiments. </del>
* Merge Tensorflow version
* Compare the rectified update, currently the implementation is slightly different from ```RAdam``` implementation.

### Done
* Updated results on an SN-GAN is in https://github.com/juntang-zhuang/SNGAN-AdaBelief, AdaBelief achieves 12.36 FID (lower is better) on Cifar10, while Adam achieves 13.25 (number taken from the log of official repository ```PyTorch-studioGAN```).
* LSTM experiments uploaded to ```PyTorch_Experiments/LSTM```

## Installation and usage

### 1. PyTorch implementations
（ Results in the paper are all generated using the PyTorch implementation in ```adabelief-pytorch``` package, which is the __ONLY__ package that I have extensively tested for now.) <br>

#### AdaBelief
```
pip install adabelief-pytorch==0.0.5
```
```
from adabelief_pytorch import AdaBelief
optimizer = AdaBelief(model.parameters(), lr=1e-3, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = False)
```
#### Adabelief with Ranger optimizer
```
pip install ranger-adabelief==0.0.9
```
```
from ranger_adabelief import RangerAdaBelief
optimizer = RangerAdaBelief(model.parameters(), lr=1e-3, eps=1e-12, betas=(0.9,0.999))
```
### 2. Tensorflow implementation
```
pip install adabelief-tf==0.0.1
```
```
from adabelief_tf impoty AdaBeliefOptimizer
optimizer = AdaBeliefOptimizer(learning_rate, epsilon=1e-12) 
```

<h2>A quick look at the algorithm</h2>
        <p align='center'>
        <img src="imgs/adabelief_algo.png" width="80%"> </p>
        <div>
            Adam and AdaBelief are summarized in Algo.1 and Algo.2, where all operations are 
            element-wise, with differences marked in blue. Note that no extra parameters are introduced in AdaBelief. For simplicity,
             we omit the bias correction step. Specifically, in Adam, the update 
             direction is  <img src="https://render.githubusercontent.com/render/math?math=\frac{m_t}{\sqrt{v_t}}"> , where <img src="https://render.githubusercontent.com/render/math?math=v_t"> is the EMA (Exponential Moving Average) of <img src="https://render.githubusercontent.com/render/math?math=g_t^2">; in AdaBelief, the update direction is <img src="https://render.githubusercontent.com/render/math?math=\frac{m_t}{\sqrt{s_t}}">,
              where <img src="https://render.githubusercontent.com/render/math?math=s_t"> is the of <img src="https://render.githubusercontent.com/render/math?math=(g_t-m_t)^2">. Intuitively, viewing <img src="https://render.githubusercontent.com/render/math?math=m_t"> as the prediction of <img src="https://render.githubusercontent.com/render/math?math=g_t">, AdaBelief takes a 
              large step when observation <img src="https://render.githubusercontent.com/render/math?math=g_t"> is close to prediction <img src="https://render.githubusercontent.com/render/math?math=m_t">, and a small step when the observation greatly deviates
               from the prediction.
        </div>
        
## Reproduce results in the paper 
#### (Comparison with 8 other optimizers: SGD, Adam, AdaBound, RAdam, AdamW, Yogi, MSVAG, Fromage)
See folder ``PyTorch_Experiments``, for each subfolder, execute ```sh run.sh```. See  ```readme.txt``` in each subfolder for visualization, or
refer to jupyter notebook for visualization.

### Results on Image Recongnition 
<p align="center">
<img src="./imgs/image_recog.png" width="80%"/> 
</p>

### Results on GAN training
<p align="center">
<img src="./imgs/GAN.png" width="80%"/>
</p>

### Results on LSTM
<p align="center">
<img src="./imgs/lstm.png" width="80%"/>
</p>

### Results on Toy Example
<p align="center">
<img src="./imgs/Beale2.gif" width="80%"/>
</p>

## Discussions

#### Installation
Please instal the latest version from pip, old versions might suffer from bugs. Source code for up-to-date package is available in folder ```pypi_packages```. 
#### Discussion on algorithms
##### 1. Weight Decay: 
- Decoupling (argument ```weight_decouple (default:False)``` appears in ```AdaBelief``` and ```RangerAdaBelief```): <br>
   Currently there are two ways to perform weight decay for adaptive optimizers, directly apply it to the gradient (Adam), or ```decouple``` weight decay from gradient descent (AdamW). This is passed to the optimizer by argument ```weight_decouple (default: False)```.

- Fixed ratio (argument ```fixed_decay (default: False)``` appears in ```AdaBelief```): <br>
   (1) If ```weight_decouple == False```, then this argument does not affect optimization. <br>
   (2) If ```weight_decouple == True```: <br>
        <ul>  If ```fixed_decay == False```, the weight is multiplied by ``` 1 -lr x weight_decay``` </ul> 
        <ul>  If ```fixed_decay == True```, the weight is multiplied by ```1 - weight_decay```. This is implemented as an option but not used to produce results in the paper. </ul>

- What is the acutal weight-decay we are using? <br>
   This is seldom discussed in the literature, but personally I think it's very important. When we set ```weight_decay=1e-4``` for SGD, the weight is scaled by ```1 - lr x weight_decay```. Two points need to be emphasized: (1) ```lr``` in SGD is typically larger than Adam (0.1 vs 0.001), so the weight decay in Adam needs to be set as a larger number to compensate. (2) ```lr``` decays, this means typically we use a larger weight decay in early phases, and use a small weight decay in late phases.

##### 2. Epsilon:
AdaBelief seems to require a different ```epsilon``` from Adam. In CV tasks in this paper, ```epsilon``` is set as ```1e-8```. For GAN training and LSTM, it's set as ```1e-12```. We recommend try different ```epsilon``` values in practice, and sweep through a large region, e.g. ```1e-8, 1e-10, 1e-12, 1e-14, 1e-16, 1e-18```. Typically a smaller ```epsilon``` makes it more adaptive.

##### 3. Rectify (argument ```rectify (default: False)``` in ```AdaBelief```):
Whether to turn on the rectification as in RAdam. The recitification basically uses SGD in early phases for warmup, then switch to Adam. Rectification is implemented as an option, but is never used to produce results in the paper.

##### 4. AMSgrad (argument ```amsgrad (default: False)``` in ```AdaBelief```):
Whether to take the max (over history) of denominator, same as AMSGrad. It's set as False for all experiments.

##### 5. Details to reproduce results
* Results in the paper are generated using the PyTorch implementation in ```adabelief-pytorch``` package. This is the __ONLY__ package that I have extensively tested for now. <br>

|   Task   | beta1 | beta2 | epsilon | weight_decay | weight_decouple | fixed_decay | rectify | amsgrad |
|:--------:|-------|-------|---------|--------------|-----------------|-------------|---------|---------|
| Cifar    | 0.9   | 0.999 | 1e-8    | 5e-4         | False           | False       | False   | False   |
| ImageNet | 0.9   | 0.999 | 1e-8    | 1e-2         | True            | False       | False   | False   |
| GAN      | 0.5   | 0.999 | 1e-12   | 0            | False           | False       | False   | False   |

* We also provide a modification of ```ranger``` optimizer in ```ranger-adabelief``` which combines ```RAdam + LookAhead + Gradient Centralization + AdaBelief```, but this is not used in the paper and is not extensively tested. 
* The ```adabelief-tf``` is a naive implementation in Tensorflow. It lacks many features such as ```decoupled weight decay```, and is not extensively tested. Currently I don't have plans to improve it since I seldom use Tensorflow, please contact me if you want to collaborate and improve it.

##### 6. Learning rate schedule
The experiments on Cifar is the same as demo in AdaBound, with the only difference is the optimizer. The ImageNet experiment uses a different learning rate schedule, typically is decayed by 1/10 at epoch 30, 60, and ends at 90. For some reasons I have not extensively experimented, AdaBelief performs good when decayed at epoch 70, 80 and ends at 90, using the default lr schedule produces a slightly worse result. If you have any ideas on this please open an issue here or email me.

##### 7. Some experience with RNN
I got some feedbacks on RNN on reddit discussion, here are a few tips:
* The epsilon is suggested to set as a smaller value for RNN (e.g. 1e-12, 1e-14, 1e-16) though the default is 1e-8. Please try different epsilon values, it varies from task to task.
* I might confuse "gradient threshold" with "gradient clip" in previous readme, clarify below: <br>
  (1) By "gradient threshold" I refer to element-wise operation, which only takes values between a certain region [a,b]. Values outside this region will be set as a and b respectively.<br>
  (2) By "gradient clip" I refer to the operation on a vector or tensor. Suppose X is a tensor, if ||X|| > thres, then X <- X/||X|| * thres. Take X as a vector, "gradient clip" shrinks the amplitude but keeps the direction.<br>
  (3) "Gradient threshold" is incompatible with AdaBelief, because if gt is thresholded for a long time, then  |gt-mt|~=0, and the division will explode; however, "gradient clip" is fine for Adabelief, yet the clip range still needs tuning (perhaps AdaBelief needs a larger range than Adam).<br>

##### 8. Contact
Please contact me at ```j.zhuang@yale.edu``` or open an issue here if you would like to help improve it, especially the tensorflow version, or explore combination with other methods, some discussion on the theory part, or combination with other methods to create a better optimizer. Any thoughts are welcome!

## Citation
```
@article{zhuang2020adabelief,
  title={AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients},
  author={Zhuang, Juntang and Tang, Tommy and Tatikonda, Sekhar and and Dvornek, Nicha and Ding, Yifan and Papademetris, Xenophon and Duncan, James},
  journal={Conference on Neural Information Processing Systems},
  year={2020}
}
```
