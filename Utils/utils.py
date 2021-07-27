import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions import Bernoulli

def generate_episode(env, controller, iters=100, render=False):
    episode = []
    state = env.reset()
    steps = 0
    total_reward = 0
    done = False
    while not done and steps != iters:
        # determine action
        action = controller.getControls(state)

        # step environment and save episode results
        next_state, reward = env.step(action)
        episode.append((state, action, reward))

        state = next_state
        steps += 1
        total_reward += reward
        done = env.done()

        if render:
            env.render()

    return episode, total_reward, steps

def train_RLalg(env, controller, episodes=1000, iters=100):
    # reset environment
    state = env.reset()

    reward_per_episode = []
    best_reward = 0
    for _ in range(episodes):
        if _ % 10 == 0:
            print("Training Episode: " + str(_) + " out of " + str(episodes))

        # generate episode
        episode, total_reward, steps = generate_episode(env, controller, iters=iters)

        # track reward per episode
        reward_per_episode.append(total_reward)

        # save the best policy
        if total_reward > best_reward:
            print("New best reward on episode " + str(_) + ": " + str(total_reward) + "! Saving policy!")
            best_reward = total_reward
            controller.save_policy()

        # update policy using the episode
        controller.update_policy(episode)

    return reward_per_episode
