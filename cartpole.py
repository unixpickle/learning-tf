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
                                                 out_probs, 2000)
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
    Run at least num_steps in the environment and return
    the observations, actions, goodnesses, and reward.
    """
    observations = []
    actions = []
    rewards = []
    while len(observations) < num_steps:
        ep_obs, ep_acts, ep_rewards = rollout(env, sess, inputs, out_probs)
        observations.extend(ep_obs)
        actions.extend(ep_acts)
        rewards.append(ep_rewards)
    mean_rew = sum([sum(x) for x in rewards]) / len(rewards)
    goodnesses = action_goodnesses(rewards)
    return observations, actions, goodnesses, mean_rew

def rollout(env, sess, inputs, out_probs):
    """
    Run a single rollout and return the observations,
    actions, and reward.
    """
    obses = []
    acts = []
    rewards = []
    obs = env.reset()
    while True:
        prob = sess.run(out_probs, feed_dict={inputs: [obs]})
        action = [1, 0]
        if random.random() < prob[0][1]:
            action = [0, 1]
        acts.append(action)
        obses.append(list(obs))
        obs, rew, done, _ = env.step(action[0])
        rewards.append(rew)
        if done:
            break
    return obses, acts, rewards

def action_goodnesses(rewards):
    """
    Produce, for each action in each episode, a measure of
    how "good" that action was, as estimated by the total
    reward of the episode.
    """
    goodnesses = []
    ep_goodnesses = normalize_rewards([sum(x) for x in rewards])
    ep_lens = [len(x) for x in rewards]
    for ep_len, ep_goodness in zip(ep_lens, ep_goodnesses):
        goodnesses.extend([[ep_goodness]] * ep_len)
    return goodnesses

def normalize_rewards(rewards):
    """
    Statistically normalize the rewards in the list.
    """
    mean = sum(rewards) / len(rewards)
    moment2 = sum([x*x for x in rewards]) / len(rewards)
    stddev = math.sqrt(moment2 - mean*mean)
    return [(x-mean)/stddev for x in rewards]

main()
