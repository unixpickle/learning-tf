"""
Train an linear policy on Knightower-v0 in µniverse.

Uses policy gradients with no discount factor.

This reaches an average score of 7.5 after a few hours.
Initial improvement is slow due to the learning rate.
"""

import math
import random
import threading

import muniverse
import numpy as np
import tensorflow as tf

def main():
    """
    Train an agent.
    """
    policy_in = tf.placeholder(tf.float32, [None, 120, 80, 1])
    policy_out = policy(policy_in)
    actions_in, rewards_in, objective = surrogate_objective(policy_out)

    optim = tf.train.GradientDescentOptimizer(1e-6)
    minimize = optim.minimize(-objective)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        roller = Roller(sess, policy_in, policy_out)
        while True:
            obses, actions, rewards, mean = roller.rollouts()
            mean_action = str(np.mean(np.array(actions), axis=0))
            print('mean_reward=%f, mean_action=%s' % (mean, mean_action))
            inputs = {
                policy_in: obses,
                actions_in: actions,
                rewards_in: rewards
            }
            sess.run(minimize, feed_dict=inputs)

def policy(in_frames):
    """
    Apply a linear policy to the input frame.
    The output is a distribution with two outcomes.
    """
    batch_size = tf.shape(in_frames)[0]
    fan_in = 80 * 120
    flat_in = tf.reshape(in_frames, [batch_size, fan_in]) / 255
    transformation = tf.Variable(tf.zeros([fan_in, 2]))
    return tf.nn.softmax(tf.matmul(flat_in, transformation))

def surrogate_objective(policy_out):
    """
    Create the surrogate objective for policy gradients.

    Returns actions, rewards, objective.
    """
    actions = tf.placeholder(tf.float32, [None, 2])
    rewards = tf.placeholder(tf.float32, [None, 1])
    objective = tf.tensordot(tf.log(policy_out), actions*rewards, axes=2)
    return actions, rewards, objective

class Roller:
    """
    Records and collect batches of environment rollouts.
    """
    def __init__(self, sess, policy_in, policy_out):
        self.sess = sess
        self.policy_in = policy_in
        self.policy_out = policy_out
        self.lock = threading.Lock()
        self.remaining_eps = 0
        self.rewards = []
        self.ep_lens = []
        self.obses = []
        self.actions = []

    def rollouts(self, num_eps=256, num_threads=8):
        """
        Collect the given number of rollouts.

        Returns observations, actions, goodnesses,
        and mean_reward.

        It is not safe to call this from multiple threads
        at a single time on one Roller.
        """
        self.remaining_eps = num_eps
        self.rewards = []
        self.ep_lens = []
        self.obses = []
        self.actions = []
        threads = []
        for _ in range(0, num_threads):
            thread = threading.Thread(target=self._run_thread)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        mean = sum(self.rewards) / len(self.rewards)
        centered = [x - mean for x in self.rewards]
        variance = sum([x*x for x in centered]) / len(centered)
        stddev = math.sqrt(variance)
        goodnesses = []
        for goodness, ep_len in zip(centered, self.ep_lens):
            goodnesses.extend([[goodness/stddev]] * ep_len)
        return self.obses, self.actions, goodnesses, mean

    def _run_thread(self):
        spec = muniverse.spec_for_name('Knightower-v0')
        env = muniverse.Env(spec)
        try:
            while True:
                self.lock.acquire()
                if self.remaining_eps == 0:
                    self.lock.release()
                    return
                self.remaining_eps -= 1
                self.lock.release()
                obses, actions, rewards = self._rollout(env)
                self.lock.acquire()
                self.rewards.append(sum(rewards))
                self.ep_lens.append(len(rewards))
                self.obses.extend(obses)
                self.actions.extend(actions)
                self.lock.release()
        finally:
            env.close()

    def _rollout(self, env):
        rewards = []
        obses = []
        actions = []
        env.reset()
        obs = downsample_obs(env.observe())
        while True:
            obses.append(obs)
            out_dist = self.sess.run(self.policy_out,
                                     feed_dict={self.policy_in: [obs]})
            action = [1, 0]
            key_action = muniverse.key_for_code('ArrowLeft')
            if random.random() < out_dist[0][1]:
                key_action = muniverse.key_for_code('ArrowRight')
                action = [0, 1]
            reward, done = env.step(0.1, key_action.with_event('keyDown'),
                                    key_action.with_event('keyUp'))
            rewards.append(reward)
            actions.append(action)
            obs = downsample_obs(env.observe())
            if done:
                break
        return obses, actions, rewards

def downsample_obs(obs):
    """
    Scale down a muniverse observation.
    """
    return obs[::4, ::4, 2:3]

main()
