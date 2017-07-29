"""
Train an Echo State Network to predict characters in text.
"""

from math import sqrt

import numpy as np
import tensorflow as tf

NORMALIZING_COEFF = 1.5925374197228312

def main():
    print('Building graph...')
    esn = ESN(states=64)
    seqs = TextSequences(batch=4, length=16)
    outputs = esn.apply_text_sequences(seqs)
    loss = seqs.loss(outputs)

    print('Training...')
    batch = seqs.build_feed_dict([
        'Hello, world!',
        'Goodbye, world!',
        'Bonjour, world!',
        'Peace out!'
    ])
    opt = tf.train.AdamOptimizer()
    minimize = opt.minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1, 1000):
            sess.run(minimize, feed_dict=batch)
            if i % 10 == 0:
                print('iter %d: loss=%f' % (i, sess.run(loss, feed_dict=batch)))

        print('Sampling...')
        for i in range(0, 10):
            print(str(esn.sample_sequence(sess), 'utf-8'))

class ESN:
    """
    ESN is a single-layer Echo State Network block.
    """
    def __init__(self, states=512, ins=256, outs=256):
        self.num_states = states
        self.init_state = tf.Variable(tf.random_normal([1, states]))

        # Maintain a steady-state variance of 1.
        normal_mag = NORMALIZING_COEFF / sqrt(states + ins)
        self.state_trans = tf.Variable(tf.random_normal([states, states],
                                                        stddev=normal_mag))
        self.in_trans = tf.Variable(tf.random_normal([ins, states],
                                                     stddev=normal_mag))

        self.out_trans = tf.Variable(tf.random_normal([states, outs],
                                                      stddev=normal_mag))
        self.out_bias = tf.Variable(tf.random_normal([1, outs]))

    def apply(self, states, inputs):
        """
        Apply the block for a single timestep.
        Produces a (state, output) pair.
        """
        new_state = tf.tanh(tf.matmul(inputs, self.in_trans) +
                            tf.matmul(states, self.state_trans))
        out = tf.matmul(new_state, self.out_trans) + self.out_bias
        return new_state, out

    def apply_text_sequences(self, text_seqs):
        """
        Apply the block to a batch of text sequences.
        """
        state = tf.zeros([text_seqs.batch, self.num_states]) + self.init_state
        res = []
        for in_batch in text_seqs.inputs:
            state, out = self.apply(state, in_batch)
            res.append(out)
        return res

    def sample_sequence(self, sess):
        """
        Sample a sequence from the model.
        Returns the sequence as a bytearray.
        """
        chars = []
        state = sess.run(self.init_state)
        last_char = np.zeros(256)

        # start with \n
        last_char[10] = 1

        while True:
            state_in = tf.placeholder(tf.float32, shape=[1, self.num_states])
            char_in = tf.placeholder(tf.float32, shape=[1, 256])
            state_out, char_out = self.apply(state_in, char_in)
            args = {
                state_in: state,
                char_in: [last_char],
            }
            out_probs = sess.run(tf.nn.softmax(char_out), feed_dict=args)[0]
            state = sess.run(state_out, feed_dict=args)
            idx = np.random.choice(range(0, 256), p=out_probs)
            if idx == 0:
                break
            chars.append(idx)
            last_char = np.zeros(256)
            last_char[idx] = 1
        return bytearray(chars)


class TextSequences:
    """
    A batch of sequences of text which can be converted to
    a sequence of input tensors.
    """
    def __init__(self, length=128, batch=16):
        self.length = length
        self.batch = batch
        self.inputs = [tf.placeholder(tf.float32, shape=[batch, 256])
                       for _ in range(0, length)]
        self.outputs = [tf.placeholder(tf.float32, shape=[batch, 256])
                        for _ in range(0, length)]

    def build_feed_dict(self, strs):
        """
        Feed the strings into the TextSequences.

        Produces a feed_dict object.
        """
        if len(strs) != self.batch:
            raise Exception('bad batch size')
        byte_seqs = [bytearray('\n' + s, 'utf-8') for s in strs]
        feed_dict = {}
        for timestep in range(0, self.length+1):
            rows = []
            for seq in byte_seqs:
                row = np.zeros(256)
                if len(seq) > timestep:
                    row[seq[timestep]] = 1
                else:
                    row[0] = 1
                rows.append(row)
            if timestep < self.length:
                feed_dict[self.inputs[timestep]] = rows
            if timestep > 0:
                feed_dict[self.outputs[timestep-1]] = rows
        return feed_dict

    def loss(self, actual_out):
        """
        Compute the cross-entropy loss between the actual
        output and the desired targets.
        """
        cost_sum = None
        for timestep, actual_term in enumerate(actual_out):
            target_term = self.outputs[timestep]
            log_probs = tf.log(tf.nn.softmax(actual_term))
            loss = -tf.tensordot(log_probs, target_term, axes=2)
            if cost_sum is None:
                cost_sum = loss
            else:
                cost_sum += loss
        return cost_sum / (self.batch * self.length)

main()
