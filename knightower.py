"""
Train an agent on Knightower-v0 in µniverse.

Uses policy gradients with a discount factor of 0.

Currently does not work very well.
"""

import math
import random

import muniverse
import tensorflow as tf

def main():
    """
    Train an agent.
    """
    policy_in = tf.placeholder(tf.float32, [None, 480, 320, 3])
    policy_out = policy(policy_in)
    actions_in, rewards_in, objective = surrogate_objective(policy_out)

    adam = tf.train.AdamOptimizer(learning_rate=1e-2)
    minimize = adam.minimize(-objective)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while True:
            obses, actions, rewards, totals = rollouts(sess, policy_in,
                                                       policy_out)
            print('mean_reward=%f' % (sum(totals)/len(totals),))
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
    fan_in = 320 * 480 * 3
    flat_in = tf.reshape(in_frames, [batch_size, fan_in])
    transformation = tf.Variable(tf.random_normal([fan_in, 2],
                                                  stddev=1/math.sqrt(fan_in)))
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

def rollouts(sess, policy_in, policy_out):
    """
    Run the policy through some rollouts.

    Return observations, actions, rewards, total_rewards.
    """
    spec = muniverse.spec_for_name('Knightower-v0')
    env = muniverse.Env(spec)
    obses = []
    actions = []
    rewards = []
    total_rewards = []
    try:
        for _ in range(0, 100):
            obs, act, rew = rollout(sess, env, policy_in, policy_out)
            obses.extend(obs)
            actions.extend(act)
            # Some amount of reward centering.
            rewards.extend([[x-0.5] for x in rew])
            total_rewards.append(sum(rew))
        return obses, actions, rewards, total_rewards
    finally:
        env.close()

def rollout(sess, env, policy_in, policy_out):
    """
    Run a single rollout.
    Return the rewards, inputs, and selected actions.
    """
    rewards = []
    obses = []
    actions = []
    env.reset()
    obs = env.observe()
    while True:
        obses.append(obs)
        out_dist = sess.run(policy_out, feed_dict={policy_in: [obs]})
        action = [1, 0]
        key_action = muniverse.key_for_code('ArrowLeft')
        if random.random() < out_dist[0][1]:
            key_action = muniverse.key_for_code('ArrowRight')
            action = [0, 1]
        reward, done = env.step(0.1, key_action.with_event('keyDown'),
                                key_action.with_event('keyUp'))
        rewards.append(reward)
        actions.append(action)
        obs = env.observe()
        if done:
            break
    return obses, actions, rewards

main()
