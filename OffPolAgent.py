import argparse
import sys
import gym
import numpy as np

from collections import defaultdict
from gym import wrappers

from matplotlib import pyplot as plt

def linearize_state(state):
    return tuple(state.flatten())

def act_by_mu(state):
    aStar = np.argmax(Q[state])
    probabilities = [epsilon / num_actions] * num_actions
    probabilities[aStar] = 1 - epsilon + (epsilon / num_actions)
    a = np.random.choice(np.arange(num_actions), p=probabilities)
    return a

def act_by_pi(state):
    return np.argmax(Pi_sa[state])

def exp_by_pi(state):
    ret = 0
    for i in range(num_actions):
        ret += Pi_sa[state][i] * Q[state][i]
    return ret

def mu_probs(state):
    aStar = np.argmax(Q[state])
    probabilities = [epsilon / num_actions] * num_actions
    probabilities[aStar] = 1 - epsilon + (epsilon / num_actions) 
    return probabilities

def is_ratio(state, action):
    return Pi_sa[state][action] / mu_probs(state)[action]

def smoothing(in_list, window):
    cumsum, moving_aves = [0], []

    for i, x in enumerate(in_list, 1):
        cumsum.append(cumsum[i-1] + x)
        if i >= window:
            moving_ave = (cumsum[i] - cumsum[i - window]) / window
            moving_aves.append(moving_ave)

    return moving_aves

parser = argparse.ArgumentParser(description=None)
parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run')
args = parser.parse_args()

env = gym.make(args.env_id)
outdir = '/tmp/random-agent-results'
# env = wrappers.Monitor(env, directory=outdir, force=True)
env.seed(0)

num_actions = env.action_space.n

Q = defaultdict(lambda : [0] * num_actions)
Pi_sa = defaultdict(lambda : [0] * num_actions)

epsilon = 0.1
gamma = 0.9

episode_count = 500
reward = 0
done = False
episodes = []
rewards = []

for i in range(episode_count):
    ob = env.reset()
    ob_lin = linearize_state(ob)
    action = act_by_pi(ob_lin)
    coeff_prod = 1.0
    gamma_prod = 1.0
    total_reward = 0
    totals = [0] * num_actions

    length = 0

    while True:
        length += 1
        ob2, reward, done, _ = env.step(action)
        total_reward += reward
        ob2_lin = linearize_state(ob2)
        next_action = act_by_mu(ob2_lin)

        gamma_prod *= gamma

        delta = reward + gamma * exp_by_pi(ob2_lin) - Q[ob_lin][action]
        Q[ob2_lin][action] += gamma_prod * coeff_prod * is_ratio(ob_lin, action) * delta

        ob = ob2
        ob_lin = ob2_lin
        action = next_action

        if done:
            break

    for state in Q:
        mx = np.max(Q[state])
        mn = np.min(Q[state])
        denom = mx - mn

        if denom == 0:
            Pi_sa[state] = [1.0 / num_actions] * num_actions
        else:
            for act in range(num_actions):
                Pi_sa[state][act] = (Q[state][act] - mn) / denom

    rewards.append(length)

env.close()

# rewards = smoothing(rewards, 10)

x_axis = np.arange(0, len(rewards))
plt.plot(x_axis, rewards, "r")
plt.show()


