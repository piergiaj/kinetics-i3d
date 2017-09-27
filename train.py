# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Loads a sample video and classifies using a trained Kinetics checkpoint."""

from __future__ import absolute_import
from __future__ import division
from datetime import datetime
import os.path
import re
import time
import collections

import numpy as np
import tensorflow as tf

import i3d
from hmdb_reader import get_two_stream_input, get_rgb_input, get_flow_input

_IMAGE_SIZE = 224
_NUM_CLASSES = 51

base_dir = '/ssd2/hmdb/'

max_steps = 5000

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
    'rgb_hmdb': 'data/checkpoints/rgb_hmdb/model.ckpt',
    'flow_hmdb': 'data/checkpoints/flow_hmdb/model.ckpt'
}

_LABEL_MAP_PATH = 'data/label_map.txt'

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('eval_type', 'joint', 'rgb, flow, or joint')
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')


def i3d_loss(scope, rgb, flow, label, rgb_model, flow_model, gpu):
  """
  Builds an I3D model and computes the loss
  """
  cr_rgb = None
  ce_flow = None
  if rgb_model is not None:
    rgb_logits = rgb_model(rgb, is_training=True, dropout_keep_prob=1.0)[0]
    rgb_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=label, logits=rgb_logits, name='cross_entropy_rgb')
    ce_rgb = tf.reduce_mean(rgb_loss, name='rgb_ce')
    tf.summary.scalar('rgb_%d' % gpu, ce_rgb)
    
  if flow_model is not None:
    flow_logits = flow_model(flow, is_training=True, dropout_keep_prob=1.0)[0]
    flow_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                      labels=label, logits=flow_logits, name='cross_entropy_flow')
    ce_flow = tf.reduce_mean(flow_loss, name='flow_ce')
    tf.summary.scalar('flow_%d' % gpu, ce_flow)

  return ce_rgb, ce_flow


def average_gradients(grads):
  """
  Averages all the gradients across the GPUs
  """
  average_grads = []
  for grad_and_vars in zip(*grads):
    gr = []
    #print grad_and_vars
    for g,_ in grad_and_vars:
      if g is None:
        continue
      exp_g = tf.expand_dims(g, 0)
      gr.append(exp_g)
    if len(gr) == 0:
      continue
    grad = tf.concat(axis=0, values=gr)
    grad = tf.reduce_mean(grad, 0)

    # remove redundant vars (because they are shared across all GPUs)
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def train(split=-1, batch_size=8, num_gpus=3, mode='rgb'):
  # mode defines if this does RGB, Flow or both
  # both uses more GPU memory though, and greatly reduces batch sizes

  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # count number of train calls
    #global_step = tf.get_variable('global_step', initializer=tf.constant(0), trainable=False)

    # create the networks
    if 'joint' in mode or 'rgb' in mode:
      with tf.variable_scope('RGB'):
        rgb_model = i3d.InceptionI3d(_NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
    if 'joint' in mode or 'flow' in mode:
      with tf.variable_scope('Flow'):
        flow_model = i3d.InceptionI3d(_NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')

    # initial learning rate
    lr = tf.Variable(0.01)

    # create optimizer
    opt = tf.train.MomentumOptimizer(lr, tf.constant(0.9))


    if 'joint' in mode:
      # get dataset
      rgbs, flows, labels = get_two_stream_input(base_dir, os.path.join(base_dir, 'splits/final_split1_train.txt'), batch_size)
      batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([rgbs, flows, labels], capacity=16*num_gpus)
    elif 'rgb' in mode:
      rgbs, labels = get_rgb_input(base_dir, os.path.join(base_dir, 'splits/final_split1_train.txt'), batch_size)
      batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([rgbs, labels], capacity=16*num_gpus)
    else:
      flows, labels = get_flow_input(base_dir, os.path.join(base_dir, 'splits/final_split1_train.txt'), batch_size)
      batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([flows, labels], capacity=16*num_gpus)      


    # create model on each GPU and get gradients for each
    all_rgb_grads = []
    all_flow_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
      # for each GPU
      for i in xrange(num_gpus):
        with tf.device('gpu:%d' % i):
          with tf.name_scope('%s_%d' % ('I3D', i)) as scope:

            # dequeue a batch
            if 'joint' in mode:
              rgb_batch, flow_batch, label_batch = batch_queue.dequeue()
              # construct I3D while sharing all variables
              # but compute the loss for each GPU
              rgb_loss, flow_loss = i3d_loss(scope, rgb_batch, flow_batch, label_batch, rgb_model, flow_model, i)
            elif 'rgb' in mode:
              rgb_batch, label_batch = batch_queue.dequeue()
              rgb_loss, flow_loss = i3d_loss(scope, rgb_batch, None, label_batch, rgb_model, None, i)
            else:
              flow_batch, label_batch = batch_queue.dequeue()
              rgb_loss, flow_loss = i3d_loss(scope, None, flow_batch, label_batch, None, flow_model, i)


            # reuse the variables on next GPU
            tf.get_variable_scope().reuse_variables()

            # retain summaries
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

            # calculate gradients for this tower
            # track all gradients
            if rgb_loss is not None:
              grads_rgb = opt.compute_gradients(rgb_loss)
              all_rgb_grads.append(grads_rgb)
              
            if flow_loss is not None:
              grads_flow = opt.compute_gradients(flow_loss)
              all_flow_grads.append(grads_flow)

    # sync and average grads
    if 'joint' in mode or 'rgb' in mode:
      rgb_grads = average_gradients(all_rgb_grads)
    if 'joint' in mode or 'flow' in mode:
      flow_grads = average_gradients(all_flow_grads)

    # track lr
    summaries.append(tf.summary.scalar('learning_rate', lr))

    # track grads:
    if 'joint' in mode or 'rgb' in mode:
      for grad, var in rgb_grads:
        if grad is not None:
          summaries.append(tf.summary.histogram(var.op.name + '/rgb_gradients', grad))
    # track grads:
    if 'joint' in mode or 'flow' in mode:
      for grad, var in flow_grads:
        if grad is not None:
          summaries.append(tf.summary.histogram(var.op.name + '/flow_gradients', grad))

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

    
    # init new layeres
    new_layers = [lr]
    
    # load pretrained weights
    if 'joint' in mode or 'rgb' in mode:
      rgb_variable_map = {}
      rgb_final_map = {}
      for variable in tf.global_variables():
        if variable.name.split('/')[0] == 'RGB':
          if 'Logits' not in variable.name:
            rgb_variable_map[variable.name.replace(':0', '')] = variable
          else:
            new_layers.append(variable)
          rgb_final_map[variable.name.replace(':0', '')] = variable
      rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
      rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
      rgb_saver = tf.train.Saver(var_list=rgb_final_map, reshape=True)

    if 'joint' in mode or 'flow' in mode:    
      flow_variable_map = {}
      flow_final_map = {}
      for variable in tf.global_variables():
        if variable.name.split('/')[0] == 'Flow':
          if 'Logits' not in variable.name:
            flow_variable_map[variable.name.replace(':0', '')] = variable
          else:
            new_layers.append(variable)
          flow_final_map[variable.name.replace(':0', '')] = variable
      flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)
      flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
      flow_saver = tf.train.Saver(var_list=flow_final_map, reshape=True)

    # apply gradients
    if 'joint' in mode or 'rgb' in mode:
      apply_rgb_grad_op = opt.apply_gradients(rgb_grads)#, global_step=global_step)
    if 'joint' in mode or 'flow' in mode:
      apply_flow_grad_op = opt.apply_gradients(flow_grads)#, global_step=global_step)

    for var in tf.trainable_variables():
      summaries.append(tf.summary.histogram(var.op.name, var))

    if 'joint' in mode:
      train_op = tf.group(apply_rgb_grad_op, apply_flow_grad_op)
    elif 'rgb' in mode:
      train_op = apply_rgb_grad_op
    else:
      train_op = apply_flow_grad_op

    summary_op = tf.summary.merge(summaries)

    # init all momentum vars
    for var in tf.global_variables():
      if 'Momentum' in var.name:
        new_layers.append(var)
    # init new layers
    init = tf.variables_initializer(new_layers)
    sess.run(init)

    
    # begin queues
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter('/home/ajpiergi/tmp', sess.graph)

    if 'joint' in mode:
      loss = rgb_loss + flow_loss
    elif 'rgb' in mode:
      loss = rgb_loss
    else:
      loss = flow_loss
      
    losses = collections.deque(maxlen=10)
    last_loss = None
    for step in xrange(max_steps):
      start_time = time.time()
      _, loss_value, lra = sess.run([train_op, loss, lr])
      duration = time.time() - start_time

      # update learning rate when loss saturates
      losses.append(loss_value)
      
      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0 and step > 0:
        # check if loss is saturated
        if last_loss is None:
          last_loss = sum(losses)/10
        else:
          diff = last_loss - sum(losses)/10
          print 'Diff:', diff
          last_loss = sum(losses)/10
          if diff < 0.001:
            lr /= 10.

        
        num_examples_per_step = batch_size * num_gpus
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / num_gpus

        format_str = ('Mode:%s %s: step %d, avg loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (mode, datetime.now(), step, sum(losses)/10,
                             examples_per_sec, sec_per_batch))
        print lra

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      if step % 1000 == 0 or (step + 1) == max_steps:
        if 'joint' in mode or 'rgb' in mode:
          rgb_saver.save(sess, _CHECKPOINT_PATHS['rgb_hmdb'], global_step=step)
        if 'joint' in mode or 'flow' in mode:
          flow_saver.save(sess, _CHECKPOINT_PATHS['flow_hmdb'], global_step=step)


if __name__ == '__main__':
  tf.app.run(train)
