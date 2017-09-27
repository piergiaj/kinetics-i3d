from __future__ import unicode_literals, division
import os
import tensorflow as tf


def fix_image_flip_shape(image, result):
  """Set the shape to 3 dimensional if we don't know anything else.
  Args:
    image: original image size
    result: flipped or transformed image
  Returns:
    An image whose shape is at least None,None,None.
  """
  from tensorflow.python.framework import tensor_shape

  image_shape = image.get_shape()
  if image_shape == tensor_shape.unknown_shape():
    result.set_shape([None, None, None, None])
  else:
    result.set_shape(image_shape)
  return result

def random_flip_left_right(image, seed=None):
  """Randomly flip an image horizontally (left to right).
  With a 1 in 2 chance, outputs the contents of `image` flipped along the
  second dimension, which is `width`.  Otherwise output the image as-is.
  Args:
    image: A 3-D tensor of shape `[height, width, channels].`
    seed: A Python integer. Used to create a random seed. See
      @{tf.set_random_seed}
      for behavior.
  Returns:
    A 3-D tensor of the same type and shape as `image`.
  Raises:
    ValueError: if the shape of `image` not supported.
  """
  from tensorflow.python.framework import ops
  from tensorflow.python.ops import random_ops
  from tensorflow.python.ops import math_ops
  from tensorflow.python.ops import control_flow_ops
  from tensorflow.python.ops import array_ops

  image = ops.convert_to_tensor(image, name='image')
  uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
  mirror_cond = math_ops.less(uniform_random, .5)
  result = control_flow_ops.cond(mirror_cond,
                                 lambda: array_ops.reverse(image, [2]), # frames X height X width x channel
                                 lambda: image)
  return fix_image_flip_shape(image, result)


def read_rgb(base, video_dir, start, num=64):
    imgs = []
    for i in range(num):
        # convert to file name in tf format
        path = tf.string_join([tf.constant(base), video_dir, tf.constant('/frame'), tf.as_string(i+start, width=6, fill='0'), tf.constant('.jpg')])
        img = tf.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        imgs.append(img)
    return tf.stack(imgs)

def read_flow(base, video_dir, start, num=64):
    imgs = []
    for i in range(num):
        flw = []
        for f in ['u', 'v']:
            # convert to file name in tf format
            path = tf.string_join([tf.constant(base), tf.constant(f+'/'), video_dir, tf.constant('/frame'), tf.as_string(i+start, width=6, fill='0'), tf.constant('.jpg')])
            img = tf.read_file(path)
            img = tf.image.decode_jpeg(img, channels=1)
            img = tf.image.convert_image_dtype(img, dtype=tf.float32)
            flw.append(img)
        imgs.append(tf.transpose(tf.squeeze(tf.stack(flw)), [1,2,0]))
    return tf.stack(imgs)
    

def read_video(queue, base, flow=False, rgb=False):
    val = queue.dequeue()

    rgb_name, flow_name, num_frames, label = tf.decode_csv(val, [[''],[''],[0],[-1]], field_delim=' ')
    
    start = tf.random_uniform([1], 1, num_frames-64, dtype='int32')[0]

    # read 64 frames of rgb
    if rgb:
      rgb_frames = read_rgb(base+'jpegs_final/', rgb_name, start)
      # take random crops
      rgb_frames = tf.random_crop(rgb_frames, size=[64, 224, 224, 3])
      # random left/right flip
      rgb_frames = random_flip_left_right(rgb_frames)
      # images are in [0,1) need to scale to [-1, 1]
      rgb_frames *= 2
      rgb_frames -= 1
      rgb_frames.set_shape([64, 224, 224, 3])

    if flow:
      flow_frames = read_flow(base+'tvl1_flow/', flow_name, start)
      
      flow_frames = tf.random_crop(flow_frames, size=[64, 224, 224, 2])
      
      flow_frames = random_flip_left_right(flow_frames)

      flow_frames *= 2
      flow_frames -= 1

      flow_frames.set_shape([64, 224, 224, 2])

    label = tf.cast(label, tf.int32)

    if flow and rgb:
      return rgb_frames, flow_frames, label
    elif flow:
      return None, flow_frames, label
    return rgb_frames, None, label
    

def generate_two_stream_batch(rgb, flow, label, batch_size):
    num_preprocess_threads = 16

    rgbs, flows, labels = tf.train.shuffle_batch([rgb, flow, label],
                                                 batch_size=batch_size,
                                                 num_threads=num_preprocess_threads,
                                                 capacity=256,
                                                 min_after_dequeue=32)
    return rgbs, flows, tf.reshape(labels, [batch_size])

def generate_one_stream_batch(stream, label, batch_size):
    num_preprocess_threads = 16

    stream,  labels = tf.train.shuffle_batch([stream, label],
                                           batch_size=batch_size,
                                           num_threads=num_preprocess_threads,
                                           capacity=256,
                                           min_after_dequeue=32)
    return stream, tf.reshape(labels, [batch_size])

def get_two_stream_input(data_dir, split_file, batch_size):
    """Returns images, flow, and labels of size [batch, 64, 224, 224, 3/2]"""

    with open(split_file, 'r') as f:
        files = f.read().splitlines()
    
    # create a queue to select videos to read
    file_queue = tf.train.string_input_producer(files)

    # read the video
    rgb, flow, label = read_video(file_queue, data_dir, flow=True, rgb=True)

    return generate_two_stream_batch(rgb, flow, label, batch_size)


def get_flow_input(data_dir, split_file, batch_size):
    """Returns flow, and labels of size [batch, 64, 224, 224, 2]"""

    with open(split_file, 'r') as f:
        files = f.read().splitlines()
    
    # create a queue to select videos to read
    file_queue = tf.train.string_input_producer(files, flow=True)

    # read the video
    rgb, flow, label = read_video(file_queue, data_dir)

    return generate_one_stream_batch(flow, label, batch_size)

def get_rgb_input(data_dir, split_file, batch_size):
    """Returns images, and labels of size [batch, 64, 224, 224, 3]"""

    with open(split_file, 'r') as f:
        files = f.read().splitlines()
    
    # create a queue to select videos to read
    file_queue = tf.train.string_input_producer(files)

    # read the video
    rgb, flow, label = read_video(file_queue, data_dir, rgb=True)

    return generate_one_stream_batch(rgb, label, batch_size)



# get_two_stream_input('/ssd2/hmdb/', '/ssd2/hmdb/splits/final_split1_train.txt', 6)
# split file is of form:
# RGB-name FLOW-name #Frames Label
