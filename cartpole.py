"""
Train an agent on the CartPole environment in OpenAI Gym.

Uses policy gradients on a small neural network.
"""

import math
import random

import gym
import tensorflow as tf

def main():
    """
    Train a policy on the CartPole-v0 environment.
    """
    observations = tf.placeholder(tf.float32, shape=[None, 4])
    out_probs = tf.nn.softmax(policy(observations))

    # Selected actions (one-hot vectors) and cumulative
    # episode rewards for those actions.
    actions = tf.placeholder(tf.float32, shape=[None, 2])
    goodnesses = tf.placeholder(tf.float32, shape=[None, 1])

    loss = -tf.tensordot(tf.log(out_probs), actions*goodnesses, axes=2)
    loss /= tf.cast(tf.shape(actions)[0], tf.float32)
    opt = tf.train.AdamOptimizer(learning_rate=1e-2)
    minimize = opt.minimize(loss)

    env = gym.make('CartPole-v0')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while True:
            obs, acts, rews, mean_rew = rollouts(env, sess, observations,
                                                 out_probs, 10000)
            loss_args = {
                observations: obs,
                actions: acts,
                goodnesses: rews
            }
            print('mean_reward=%f' % (mean_rew,))
            sess.run(minimize, feed_dict=loss_args)

def policy(inputs):
    """
    Apply a randomized policy to the inputs.
    """
    weights1 = tf.Variable(tf.random_normal([4, 10]))
    biases1 = tf.Variable(tf.random_normal([1, 10]))
    weights2 = tf.Variable(tf.zeros([10, 2]))
    return tf.matmul(tf.tanh(tf.matmul(inputs, weights1) + biases1), weights2)

def rollouts(env, sess, inputs, out_probs, num_steps):
    """
    Rollouts runs the environment for up to num_steps
    timesteps and returns the observations, actions,
    action goodnesses, and mean reward.
    """
    observations = []
    actions = []
    ep_rewards = []
    ep_lens = []
    while len(observations) < num_steps:
        ep_len = 0
        ep_reward = 0
        obs = env.reset()
        while True:
            prob = sess.run(out_probs, feed_dict={inputs: [obs]})
            action = [1, 0]
            if random.random() < prob[0][1]:
                action = [0, 1]
            actions.append(action)
            observations.append(list(obs))
            obs, rew, done, _ = env.step(action[0])
            ep_reward += rew
            ep_len += 1
            if done:
                break
        ep_rewards.append(ep_reward)
        ep_lens.append(ep_len)
    goodnesses = []
    mean_rew = sum(ep_rewards) / len(ep_rewards)
    norm_rew = normalize_rewards(ep_rewards)
    for ep_len, ep_goodness in zip(ep_lens, norm_rew):
        goodnesses.extend([[ep_goodness] for _ in range(0, ep_len)])
    return observations, actions, goodnesses, mean_rew

def normalize_rewards(rewards):
    """
    Statistically normalize the rewards in the list.
    """
    mean = sum(rewards) / len(rewards)
    moment2 = sum([x*x for x in rewards]) / len(rewards)
    stddev = math.sqrt(moment2 - mean*mean)
    return [(x-mean)/stddev for x in rewards]

main()
