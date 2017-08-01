"""
Train an agent on Knightower-v0 in µniverse.

Uses policy gradients with no discount factor.

Currently does not work very well.
"""

import random
import threading

import muniverse
import tensorflow as tf

def main():
    """
    Train an agent.
    """
    policy_in = tf.placeholder(tf.float32, [None, 120, 80, 1])
    policy_out = policy(policy_in)
    actions_in, rewards_in, objective = surrogate_objective(policy_out)

    adam = tf.train.AdamOptimizer(learning_rate=1e-5)
    minimize = adam.minimize(-objective)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        roller = Roller(sess, policy_in, policy_out)
        while True:
            obses, actions, rewards, mean = roller.rollouts()
            print('mean_reward=%f' % (mean,))
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
        self.total_rewards = []
        self.rewards = []
        self.obses = []
        self.actions = []

    def rollouts(self, num_eps=128, num_threads=8):
        """
        Collect the given number of rollouts.

        Returns observations, actions, goodnesses,
        and mean_reward.

        It is not safe to call this from multiple threads
        at a single time on one Roller.
        """
        self.remaining_eps = num_eps
        self.total_rewards = []
        self.rewards = []
        self.obses = []
        self.actions = []
        threads = []
        for _ in range(0, num_threads):
            thread = threading.Thread(target=self._run_thread)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        mean = sum(self.total_rewards) / len(self.total_rewards)
        return self.obses, self.actions, self.rewards, mean

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
                self.total_rewards.append(sum(rewards))
                total = sum(rewards)
                self.rewards.extend([[total] for _ in rewards])
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
