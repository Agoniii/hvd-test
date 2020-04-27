from __future__ import absolute_import, division, print_function

import numpy as np

import tensorflow as tf
import horovod.tensorflow as hvd
from network import Network

batch_size = 64
num_iters = 10
hvd.init()

with tf.device('/gpu:0'):
    data = tf.get_variable("data", shape=[batch_size, 224, 224, 3], initializer=tf.random_normal_initializer(), trainable=False)
    target = tf.get_variable("target", shape=[batch_size, 1], initializer=tf.random_normal_initializer(), trainable=False)
    target = tf.cast(target, tf.int64)

    network = Network(batch_size)

    loss = network.cal_train_loss([data,target])
    params = tf.trainable_variables()
    grads = tf.gradients(loss, params)
    grads1 = [hvd.allreduce(grad, average=False) for grad in grads]
    gradvars = list(zip(grads1, params))

    opt = tf.train.GradientDescentOptimizer(0.001)
    train_opt = opt.apply_gradients(gradvars)

init = tf.compat.v1.global_variables_initializer()
bcast_op = hvd.broadcast_global_variables(0)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
with tf.Session(config=config) as session:
    session.run(init)
    session.run(bcast_op)
    for x in range(num_iters):
        session.run(train_opt)
