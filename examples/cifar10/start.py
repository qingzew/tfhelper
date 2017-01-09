#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 qingze <qingze@node39.com>
#
# Distributed under terms of the MIT license.

import cifar10
from tfhelper.trainer import Trainer
import tensorflow as tf

trainer = Trainer(checkpoint_path = '/tmp/tf_logs/')

# @trainer.register('build_model')
# def build_model_func():
#     global_step = tf.Variable(0, trainable = False)

#     images, labels = cifar10.distorted_inputs()

#     logits = cifar10.inference(images)

#     loss = cifar10.loss(logits, labels)

#     train_op = cifar10.train(loss, global_step)

#     return train_op, loss, global_step

@trainer.register('build_model')
def build_model_func():

    images, labels = cifar10.distorted_inputs()

    logits = cifar10.inference(images)

    loss = cifar10.loss(logits, labels)

    return loss


cifar10.maybe_download_and_extract()
trainer.train()
