#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 qingze <qingze@node39.com>
#
# Distributed under terms of the MIT license.

import tflearn

def get_weight(shape, trainable = True, name = 'W',
                weights_init = 'truncated_normal', regularizer = None, weight_decay = None, restore = 'True'):
    """get_weights
        weights_init: `str` (name) or `Tensor`. Weights initialization.
        regularizer: `str` (name) or `Tensor`. Add a regularizer to layer weights
    """

    weights_init = weights_init
    if isinstance(weights_init, str):
        W_init = tflearn.initializations.get(weights_init)()

    W_regul = None
    if regularizer:
        W_regul = lambda x: tflearn.losses.get(regularizer)(x, weight_decay)

    W = tflearn.variables.variable(name, shape = shape, regularizer = W_regul,
                initializer = W_init, trainable = trainable, restore = restore)

    return W

def get_bias(shape, bias_init = 'zeros', trainable = True, restore = True, name = 'b'):
    if isinstance(bias_init, str):
        bias_init = tflearn.initializations.get(bias_init)()
        b = tflearn.variables.variable(name, shape = shape, initializer = bias_init,
                    trainable = trainable, restore = restore)
    return b
