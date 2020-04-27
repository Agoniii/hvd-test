# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import numpy as np
import tensorflow as tf

from tensorflow.python.training import moving_averages

HParams = namedtuple('HParams',
                     'batch_size, num_classes,'
                     'resnet_size, use_bottleneck, weight_decay_rate,'
                     'relu_leakiness')
                     
class ResNet(object):
    def __init__(self, hps, images, is_training, data_format='NHWC'):
        print("this is resnet50")
        self.hps = hps
        self.images = images
        self.filters = [256, 512, 1024, 2048]
        resnet_size = self.hps.resnet_size
        if resnet_size == 50:            # resnet size paramters
            self.stages = [3,   4,   6,    3]
        elif resnet_size == 101:
            self.stages = [3,   4,   23,   3]
        elif resnet_size == 152:
            self.stages = [3,   8,   36,   3]
        else:
            raise ValueError('resnet_size %d Not implement:' % FLAGS.resnet_size)
        self._data_format = data_format
        self.is_training = is_training
        self._extra_train_ops = []

    def _stride_array(self, stride):
        return [1, 1, stride, stride] if self._data_format == 'NCHW' else [1, stride, stride, 1]

    def _ksize_array(self, ksize):
        return [1, 1, ksize, ksize] if self._data_format == 'NCHW' else [1, ksize, ksize, 1]
        
    def _conv(self, name, x, kernel_size, in_channels, out_channels, strides, padding='SAME'):
        with tf.variable_scope(name):
            initializer=tf.variance_scaling_initializer()
            kernel = tf.get_variable(
                'conv2d_kernel', [kernel_size, kernel_size, in_channels, out_channels],
                tf.float32, initializer=initializer, trainable=True)
            return tf.nn.conv2d(x, kernel, strides, padding=padding, data_format=self._data_format)

    def _batch_norm(self, name, x):
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]] if self._data_format == 'NHWC' else [x.get_shape()[1]]

            beta = tf.get_variable('beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32), trainable=True)
            gamma = tf.get_variable('gamma', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32), trainable=True)

            moving_mean = tf.get_variable(
                'moving_mean', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)
                
            moving_variance = tf.get_variable(
                'moving_variance', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32), trainable=False)

            if self.is_training :
                bn, batch_mean, batch_variance = tf.nn.fused_batch_norm(
                    x, gamma, beta, epsilon=1e-5,
                    data_format=self._data_format, is_training=True)
                mean_update = moving_averages.assign_moving_average(
                    moving_mean, batch_mean, decay=0.9, zero_debias=False)
                variance_update = moving_averages.assign_moving_average(
                    moving_variance, batch_variance, decay=0.9, zero_debias=False)
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, mean_update)
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, variance_update)
                self._extra_train_ops.append(mean_update)
                self._extra_train_ops.append(variance_update)

            else:
                bn, mean, variance = tf.nn.fused_batch_norm(x, gamma, beta, mean=moving_mean,
                                                            variance=moving_variance, epsilon=1e-5,
                                                            data_format=self._data_format, is_training=False)
            return bn

    def _relu(self, x, leakiness=0.0):
        if leakiness > 0.0:
            return tf.nn.leaky_relu(x, leakiness)
        else:
            return tf.nn.relu(x)

    def _mpool(self, name, x, ksize, strides):
        with tf.variable_scope(name):
            return tf.nn.max_pool(x, ksize, strides, padding='SAME', data_format=self._data_format)

    def spatial_mean(self, x, keep_dims=False):
        assert x.get_shape().ndims == 4
        axes = [2, 3] if self._data_format == 'NCHW' else [1, 2]
        return tf.reduce_mean(x, axes, keepdims=keep_dims)
        
    def _fully_connected(self, x, out_dim):
        x_last_dim = x.get_shape()[-1] if self._data_format == 'NHWC' else x.get_shape()[1]
        in_channels = x.get_shape().as_list()[-1] if self._data_format == 'NHWC' else x.get_shape().as_list()[1]
        x = tf.reshape(x, [self.hps.batch_size, -1])
        init_factor = 1.
        stddev = np.sqrt(init_factor / in_channels)
        print("stddev", stddev)
        w = tf.get_variable(
            'conv2d_kernel', [x_last_dim, out_dim],
            initializer=tf.truncated_normal_initializer(stddev=stddev))

        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer(0.0))
        return tf.nn.xw_plus_b(x, w, b)
        
    def _bottleneck_residual_v1(self, x, out_channels, stride, shortcut):
        # short cut
        orig_x = x
        in_channels = x.get_shape()[-1] if self._data_format == 'NHWC' else x.get_shape()[1]
        if shortcut=='conv':
          orig_x = self._conv('conv1_b1', orig_x, 1, in_channels, out_channels, self._stride_array(stride))
          orig_x = self._batch_norm('conv1_b1_bn', orig_x)

        # bottleneck_residual_block
        x = self._conv('conv1_b2', x, 1, in_channels, out_channels/4, self._stride_array(stride))
        x = self._batch_norm('conv1_b2_bn', x)
        x = self._relu(x)

        x = self._conv('conv2_b2', x, 3, out_channels/4, out_channels/4, self._stride_array(1))
        x = self._batch_norm('conv2_b2_bn', x)
        x = self._relu(x)
        
        x = self._conv('conv3_b2', x, 1, out_channels/4, out_channels, self._stride_array(1))
        x = self._batch_norm('conv3_b2_bn', x)

        # sum
        return tf.nn.relu(x + orig_x)
    
    
    def _build_model(self):
        # init
        with tf.variable_scope('init'):
          with tf.xla.experimental.jit_scope(True):
            x = self.images
            if self._data_format == 'NCHW':
                x = tf.transpose(x, [0, 3, 1, 2]) 
            x = self._conv('init_conv', x, 7, 3, 64, self._stride_array(2), 'VALID')
            x = self._batch_norm('init_bn', x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._mpool("init_mpool", x, self._ksize_array(3), self._stride_array(2))
            
        # 4 stages 
        for i in range(0, len(self.stages)):
            with tf.variable_scope('stages_%d_block_%d' % (i, 0)):
              with tf.xla.experimental.jit_scope(True):
                stride = 1 if i == 0 else 2
                x = self._bottleneck_residual_v1(
                    x, 
                    self.filters[i], 
                    stride, 
                    'conv')
            for j in range(1, self.stages[i]):
                with tf.variable_scope('stages_%d_block_%d' % (i,j)):
                  with tf.xla.experimental.jit_scope(True):
                    x = self._bottleneck_residual_v1(
                        x, 
                        self.filters[i], 
                        1,
                        'identity')
       
        # spatial mean
        with tf.variable_scope('spatial_mean'):
          with tf.xla.experimental.jit_scope(True):
            x = self.spatial_mean(x)
            
        # logits
        with tf.variable_scope('logits'):
          with tf.xla.experimental.jit_scope(True):
            logits = self._fully_connected(x, self.hps.num_classes)
         
        return logits
        
class Network(object):
    def __init__(self, batch_size):
        self.hps = HParams(batch_size=batch_size,
                      num_classes=1001,
                      resnet_size=50,
                      use_bottleneck=False,
                      weight_decay_rate=0.0001,
                      relu_leakiness=0.0)
                      
    def cal_train_loss(self, data_list):
        images = data_list[0]
        labels = tf.reshape(data_list[1], [-1])
        labels = tf.one_hot(labels, 1001)

        model = ResNet(self.hps, images, True)
        logits = model._build_model()

        with tf.variable_scope('costs'):
            xent = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=labels)
            cost = tf.reduce_mean(xent, name='xent')
            # L2 weight decay loss.
            costs = []
            for var in tf.trainable_variables():
                if var.op.name.find(r'conv2d_kernel') > 0:
                    costs.append(tf.nn.l2_loss(var))
            cost += tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))
        return cost
