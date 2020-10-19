"""
Copyright (C) 2020 Jeremy Bernstein, Arash Vahdat, Yisong Yue & Ming-Yu Liu.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/).
"""

import torch
import math
from torch.optim.optimizer import Optimizer, required


class Fromage(Optimizer):

    def __init__(self, params, lr=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)
        super(Fromage, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad.data
                d_p_norm = p.grad.norm()
                p_norm = p.norm()

                if p_norm > 0.0 and d_p_norm > 0.0:
                    p.data.add_(-group['lr'], d_p * (p_norm / d_p_norm))
                else:
                    p.data.add_(-group['lr'], d_p)
                p.data /= math.sqrt(1+group['lr']**2)

        return loss
