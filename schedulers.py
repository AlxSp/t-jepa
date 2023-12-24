# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in https://github.com/facebookresearch/ijepa
#
import math

import torch


class WarmupCosineSchedule:

    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        T_max,
        final_lr=0.,
        step = 0.
    ):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self.current_step = step

    def step(self):
        if self.current_step < self.warmup_steps:
            progress = float(self.current_step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # -- progress after warmup
            progress = float(self.current_step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(self.final_lr,
                         self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress)))

        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

        self.current_step += 1
        return new_lr


class CosineWDSchedule:

    def __init__(
        self,
        optimizer,
        ref_wd,
        T_max,
        final_wd=0.,
        step = 0.
    ):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self.current_step = step

    def step(self):
        progress = self.current_step / self.T_max
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1. + math.cos(math.pi * progress))

        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            if ('WD_exclude' not in group) or not group['WD_exclude']:
                group['weight_decay'] = new_wd
        
        self.current_step += 1
        return new_wd

class ExponentialMovingAverageSchedule:
    def __init__(self, momentum, T_max, step = 0.):

        self.momentum = momentum
        self.T_max = T_max
        self.current_step = step

    @torch.no_grad()
    def step(self, source_model, target_model):
        momentum = self.momentum + self.current_step / self.T_max * (1.0 - self.momentum) 

        for param_q, param_k in zip(source_model.parameters(), target_model.parameters()):
            param_k.data.mul_(momentum).add_((1.0 - momentum) * param_q.detach().data)

        self.current_step += 1
        return momentum