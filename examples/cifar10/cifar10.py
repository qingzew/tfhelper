from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import cifar10_input

from tfhelper import variables_ext

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


# def _activation_summary(x):
#   """Helper to create summaries for activations.

#   Creates a summary that provides a histogram of activations.
#   Creates a summary that measures the sparsity of activations.

#   Args:
#     x: Tensor
#   Returns:
#     nothing
#   """
#   # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
#   # session. This helps the clarity of presentation on tensorboard.
#   tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
#   tf.histogram_summary(tensor_name + '/activations', x)
#   tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def distorted_inputs():
    """Construct distorted input for CIFAR training using the Reader ops.

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.

    Raises:
        ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def inference(images):
    """Build the CIFAR-10 model.

    Args:
        images: Images returned from distorted_inputs() or inputs().

    Returns:
      Logits.
    """
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = variables_ext.get_weight(shape = [5, 5, 3, 64])
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variables_ext.get_bias(shape = [64])
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name = scope.name)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = variables_ext.get_weight(shape = [5, 5, 64, 64])
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variables_ext.get_bias(shape = [64])
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = variables_ext.get_weight(shape = [dim, 384])
        biases = variables_ext.get_bias(shape = [384])
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name = scope.name)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = variables_ext.get_weight(shape = [384, 192])
        biases = variables_ext.get_bias(shape = [192])
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name = scope.name)

    # softmax, i.e. softmax(WX + b)
    with tf.variable_scope('softmax_linear') as scope:
        weights = variables_ext.get_weight(shape = [192, NUM_CLASSES], restore = False)
        biases = variables_ext.get_bias(shape = [NUM_CLASSES], restore = False)
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name = scope.name)
    # with tf.variable_scope('softmax') as scope:
    #     weights = variables_ext.get_weight(shape = [192, 12], restore = False)
    #     biases = variables_ext.get_bias(shape = [12], restore = False)
    #     softmax_linear = tf.add(tf.matmul(local4, weights), biases, name = scope.name)

    return softmax_linear


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Args:
        logits: Logits from inference().
        labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

    Returns:
        Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                    float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
