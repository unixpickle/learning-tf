"""
Train a GAN to generate MNIST digits.

Uses a feature matching objective.
"""

from math import sqrt
import sys

from matplotlib import pyplot
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

SAVE_FILE = 'gan.ckpt'
GRID_SIZE = 5

def main():
    """
    Train or generate digits.
    """
    if len(sys.argv) < 2:
        print('Usage: mnist_gan <create | train | generate>')
        sys.exit()

    if sys.argv[1] == 'create':
        create()
    elif sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'generate':
        generate()
    else:
        print('Unknown command: ' + sys.argv[1])
        sys.exit()

def create():
    """
    Create a new model and save it.
    """
    GAN()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, SAVE_FILE)

def train():
    """
    Load and train a model.
    """
    gan = GAN()
    saver = tf.train.Saver()

    samples = Samples()

    gen_obj = gan.generator_objective(samples.noise, samples.images)
    gen_opt = tf.train.AdamOptimizer(learning_rate=1e-4)
    opt_gen = gen_opt.minimize(gen_obj, var_list=gan.generator_vars())

    disc_obj = gan.discriminator_objective(samples.noise, samples.images)
    disc_opt = tf.train.AdamOptimizer(learning_rate=1e-4)
    opt_disc = disc_opt.minimize(disc_obj, var_list=gan.discriminator_vars())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, SAVE_FILE)
        while True:
            batch = samples.sample_feed_dict()
            print('disc=%f gen=%f' % (sess.run(disc_obj, feed_dict=batch),
                                      sess.run(gen_obj, feed_dict=batch)))
            sess.run(opt_disc, feed_dict=batch)
            sess.run(opt_gen, feed_dict=batch)
            saver.save(sess, SAVE_FILE)

def generate():
    """
    Generate images from the model.
    """
    gan = GAN()
    saver = tf.train.Saver()
    noise = tf.Variable(tf.random_normal([GRID_SIZE**2, 100]))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, SAVE_FILE)
        out = np.array(sess.run(gan.generate(noise)))
        cols = [out[i*GRID_SIZE:(i+1)*GRID_SIZE].reshape([28*GRID_SIZE, 28])
                for i in range(0, GRID_SIZE)]
        grid = np.concatenate(cols, axis=1)
        pyplot.title('Generated digits')
        pyplot.imshow(grid, cmap='gray')
        pyplot.show()

class GAN:
    """
    A small Generative Adversarial Network for producing
    MNIST digits.
    """
    def __init__(self, noise_size=100):
        """
        Create a new GAN with random weights.
        """
        self.generator = [
            FC(noise_size, 14 * 14),
            FC(14 * 14, 14 * 14),
            FC(14 * 14, 14 * 14),
            Reshape([14, 14, 1]),
            Resize([28, 28]),
            Conv(1, 16),
            Conv(16, 32),
            Conv(32, 64),
            Conv(64, 64),
            Conv(64, 32),
            Conv(32, 1, activation=False)
        ]
        self.discriminator = [
            Conv(1, 16, strides=[2, 2]),
            Conv(16, 32),
            Conv(32, 32),
            Conv(32, 32),
            Conv(32, 32),
            Conv(32, 16, strides=[2, 2]),
            Reshape([7 * 7 * 16]),
            FC(784, 256),
            FC(256, 256),
        ]
        self.discriminator_final = [
            FC(256, 1, activation=False)
        ]

    def generate(self, noise):
        """
        Apply the generator to the batch of noise.
        """
        return tf.sigmoid(apply_network(self.generator, noise))

    def discriminate(self, samples):
        """
        Apply the discriminator and get a pre-sigmoid
        probability.
        """
        disc = self.discriminator + self.discriminator_final
        return apply_network(disc, samples)

    def generator_objective(self, noise, samples):
        """
        Get the objective (loss) for the generator.
        """
        out_1 = apply_network(self.discriminator, self.generate(noise))
        out_2 = apply_network(self.discriminator, samples)
        mean_1 = tf.reduce_mean(out_1, axis=0)
        mean_2 = tf.reduce_mean(out_2, axis=0)
        return tf.reduce_mean(tf.square(mean_1 - mean_2))

    def discriminator_objective(self, noise, samples):
        """
        Get the objective (loss) for the discriminator.
        """
        out_1 = self.discriminate(self.generate(noise))
        out_2 = self.discriminate(samples)
        labels_1 = tf.zeros(tf.shape(out_1))
        labels_2 = tf.ones(tf.shape(out_2))
        cost_1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_1,
                                                         logits=out_1)
        cost_2 = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_2,
                                                         logits=out_2)
        return tf.reduce_mean(cost_1) + tf.reduce_mean(cost_2)

    def generator_vars(self):
        """
        Get the generator variables.
        """
        return network_vars(self.generator)

    def discriminator_vars(self):
        """
        Get the discriminator variables.
        """
        return network_vars(self.discriminator + self.discriminator_final)

class Samples:
    """
    Manage random batches of MNIST samples and isotropic
    Gaussian noise.
    """
    def __init__(self, batch_size=128, noise_size=100):
        self.batch_size = batch_size
        self.noise_size = noise_size
        self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        self.images = tf.placeholder(tf.float32, shape=[batch_size, 28, 28, 1])
        self.noise = tf.placeholder(tf.float32, shape=[batch_size, noise_size])

    def sample_feed_dict(self):
        """
        Get a dictionary of arguments to fill in for the
        placeholders.
        """
        mnist_batch = self.mnist.train.next_batch(self.batch_size)[0]
        noise_batch = np.random.normal(size=(self.batch_size, self.noise_size))
        return {
            self.images: mnist_batch.reshape((self.batch_size, 28, 28, 1)),
            self.noise: noise_batch
        }

class FC:
    """
    A fully connected layer.
    """
    def __init__(self, in_count, out_count, activation=True):
        self.activation = activation
        stddev = 1 / sqrt(in_count)
        self.weights = tf.Variable(tf.random_normal([in_count, out_count],
                                                    stddev=stddev))
        self.biases = tf.Variable(tf.zeros([1, out_count]))

    def apply(self, inputs):
        """
        Apply the layer to the batch of inputs.
        """
        pre_activation = tf.matmul(inputs, self.weights) + self.biases
        if not self.activation:
            return pre_activation
        return selu(pre_activation)

    def vars(self):
        """
        Get the parameters of the layer.
        """
        return [self.weights, self.biases]

class Reshape:
    """
    A layer to reshape inputs.
    """
    def __init__(self, shape):
        self.shape = shape

    def apply(self, inputs):
        """
        Apply the layer to a batch of inputs.
        """
        out = tf.reshape(inputs, [tf.shape(inputs)[0]] + self.shape)
        return out

    def vars(self):
        """
        Return the parameters of the layer (i.e. []).
        """
        return []

class Resize:
    """
    A layer to resize image inputs.
    """
    def __init__(self, size):
        self.size = size

    def apply(self, inputs):
        """
        Apply the layer to a batch of inputs.
        """
        return tf.image.resize_images(inputs, self.size)

    def vars(self):
        """
        Return the parameters of the layer (i.e. []).
        """
        return []

class Conv:
    """
    A 3x3 convolutional layer.
    """
    def __init__(self, in_depth, out_depth, strides=None, activation=True):
        self.activation = activation
        self.strides = strides
        shape = [3, 3, in_depth, out_depth]
        stddev = 1 / sqrt(in_depth * 9)
        self.filters = tf.Variable(tf.random_normal(shape, stddev=stddev))
        self.biases = tf.Variable(tf.zeros([1, out_depth]))

    def apply(self, inputs):
        """
        Apply the layer to the batch of inputs.
        """
        conv_out = tf.nn.convolution(inputs, self.filters, 'SAME',
                                     strides=self.strides)
        pre_activation = conv_out + self.biases
        if not self.activation:
            return pre_activation
        return selu(pre_activation)

    def vars(self):
        """
        Get the parameters of the layer.
        """
        return [self.filters, self.biases]

def apply_network(network, inputs):
    """
    Apply a neural network (a list of layers).
    """
    sub_in = inputs
    for layer in network:
        sub_in = layer.apply(sub_in)
    return sub_in

def network_vars(network):
    """
    Concatenate the parameters of each layer in a list.
    """
    res = []
    for layer in network:
        res.extend(layer.vars())
    return res

def selu(x):
    """
    Self-normalizing ELU.

    See https://github.com/bioinf-jku/SNNs/blob/master/selu.py.
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))

main()
