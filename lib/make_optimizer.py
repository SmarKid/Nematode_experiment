#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
from torch import optim


def make_optimizer(config, model):
    trainable = filter(lambda x: x.requires_grad, model.parameters())
    if config.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = config.optim_args[config.optimizer]
    elif config.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (config.beta1, config.beta2),
            'eps': config.epsilon,
            'amsgrad': config.amsgrad
        }
    elif config.optimizer == 'ADAMAX':
        optimizer_function = optim.Adamax
        kwargs = {
            'betas': (config.beta1, config.beta2),
            'eps': config.epsilon
        }
    elif config.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {
            'eps': config.epsilon,
            'momentum': config.momentum
        }
    elif config.optimizer == 'AdamW':
        optimizer_function = optim.AdamW
        kwargs = config.optim_args[config.optimizer]
    else:
        raise Exception()

    return optimizer_function(trainable, **kwargs)