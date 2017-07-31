"""
Train a Wasserstein GAN to generate MNIST digits.
"""

from math import sqrt
import sys

from matplotlib import pyplot
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

SAVE_FILE = 'gan.ckpt'

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
    Train the model.
    """
    gan = GAN()
    saver = tf.train.Saver()

    samples = Samples()

    gen_obj = gan.generator_objective(samples.noise)
    gen_adam = tf.train.AdamOptimizer()
    opt_gen = gen_adam.minimize(gen_obj, var_list=gan.generator_vars())

    disc_obj = gan.discriminator_objective(samples.noise, samples.images)
    disc_adam = tf.train.AdamOptimizer()
    opt_disc = disc_adam.minimize(disc_obj, var_list=gan.discriminator_vars())
    clip_disc = gan.clip_discriminator()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, SAVE_FILE)
        while True:
            losses = []
            for _ in range(0, 10):
                batch = samples.sample_feed_dict()
                losses.append(sess.run(disc_obj, feed_dict=batch))
                sess.run(opt_disc, feed_dict=batch)
                sess.run(clip_disc)
            batch = samples.sample_feed_dict()
            loss = sess.run(gen_obj, feed_dict=batch)
            print('disc=%f gen=%f' % (sum(losses)/len(losses), loss))
            sess.run(opt_gen, feed_dict=batch)
            saver.save(sess, SAVE_FILE)

def generate():
    """
    Generate images from the model.
    """
    gan = GAN()
    saver = tf.train.Saver()
    noise = tf.Variable(tf.random_normal([4, 100]))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, SAVE_FILE)
        out = np.array(sess.run(gan.generate(noise)))
        pyplot.title('Generated digits')
        pyplot.imshow(out.reshape([28*4, 28]), cmap='gray')
        pyplot.show()

class GAN:
    """
    GAN is a Generative Adversarial Network for producing
    MNIST digits.
    """
    def __init__(self, noise_size=100):
        """
        Create a new GAN.
        """
        self.gen_weights_1 = random_weights([noise_size, 14*14], noise_size)
        self.gen_fc_biases_1 = tf.Variable(tf.zeros([1, 14*14]))
        self.gen_weights_2 = random_weights([14*14, 14*14], 14*14)
        self.gen_fc_biases_2 = tf.Variable(tf.zeros([1, 14*14]))
        self.gen_filters_1 = random_weights([3, 3, 1, 16], 9)
        self.gen_biases_1 = tf.Variable(tf.zeros([1, 1, 16]))
        self.gen_filters_2 = random_weights([3, 3, 16, 1], 144)
        self.gen_biases_2 = tf.Variable(tf.zeros([1, 1, 1]))
        self.disc_filters_1 = random_weights([3, 3, 1, 16], 9)
        self.disc_biases_1 = tf.Variable(tf.zeros([1, 1, 16]))
        self.disc_filters_2 = random_weights([3, 3, 16, 16], 144)
        self.disc_biases_2 = tf.Variable(tf.zeros([1, 1, 16]))
        self.disc_weights_1 = random_weights([784, 256], 784)
        self.disc_fc_biases = tf.Variable(tf.zeros([1, 256]))
        self.disc_weights_2 = random_weights([256, 1], 256)

    def generate(self, noise):
        """
        Apply the generator to the batch of noise.
        """
        batch_size = tf.shape(noise)[0]
        fc_1 = tf.tanh(tf.matmul(noise, self.gen_weights_1) +
                       self.gen_fc_biases_1)
        fc_2 = tf.tanh(tf.matmul(fc_1, self.gen_weights_2) +
                       self.gen_fc_biases_2)
        small_images = tf.reshape(fc_2, [batch_size, 14, 14, 1])
        full_images = tf.image.resize_images(small_images, [28, 28])
        conv1 = tf.nn.convolution(full_images, self.gen_filters_1, 'SAME')
        out1 = tf.tanh(conv1 + self.gen_biases_1)
        conv2 = tf.nn.convolution(out1, self.gen_filters_2, 'SAME')
        return tf.sigmoid(conv2 + self.gen_biases_2)

    def discriminate(self, images):
        """
        Apply the discriminator to a batch of images.
        """
        batch_size = tf.shape(images)[0]
        conv1 = tf.nn.convolution(images, self.disc_filters_1, 'SAME',
                                  strides=[2, 2])
        out1 = tf.tanh(conv1 + self.disc_biases_1)
        conv2 = tf.nn.convolution(out1, self.disc_filters_2, 'SAME',
                                  strides=[2, 2])
        out2 = tf.tanh(conv2 + self.disc_biases_2)
        fc_in = tf.reshape(out2, [batch_size, 16*7*7])
        fc_out_1 = tf.tanh(tf.matmul(fc_in, self.disc_weights_1) +
                           self.disc_fc_biases)
        return tf.matmul(fc_out_1, self.disc_weights_2)

    def clip_discriminator(self, mag=0.1):
        """
        Return an op to clip the discriminator weights.
        Clips the absolute value in [-mag, mag].

        This is necessary for training a WGAN.
        """
        disc = self.discriminator_vars()
        ops = [tf.assign(x, tf.clip_by_value(x, -mag, mag)) for x in disc]
        return tf.group(*ops)

    def generator_objective(self, noise):
        """
        Get the objective (loss) for the generator.
        """
        gen_out = self.generate(noise)
        return -tf.reduce_mean(self.discriminate(gen_out))

    def discriminator_objective(self, noise, samples):
        """
        Get the objective (loss) for the discriminator.
        """
        gen_samples = self.generate(noise)
        return tf.reduce_mean(self.discriminate(gen_samples) -
                              self.discriminate(samples))

    def generator_vars(self):
        """
        Get the generator variables.
        """
        return [self.gen_filters_1, self.gen_filters_2,
                self.gen_biases_1, self.gen_biases_1,
                self.gen_weights_1, self.gen_weights_2,
                self.gen_fc_biases_1, self.gen_fc_biases_2]

    def discriminator_vars(self):
        """
        Get the discriminator variables.
        """
        return [self.disc_filters_1, self.disc_filters_2,
                self.disc_biases_1, self.disc_biases_1,
                self.disc_weights_1, self.disc_weights_2,
                self.disc_fc_biases]

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

def random_weights(shape, fan_in):
    """
    Create a random weight tensor which preserves the
    input standard deviation given the number of inputs.
    """
    return tf.Variable(tf.random_normal(shape, stddev=1/sqrt(fan_in)))

main()
