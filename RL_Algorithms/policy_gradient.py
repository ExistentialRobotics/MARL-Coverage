import numpy as np
import torch
from torch.distributions.categorical import Categorical
from . Utils.utils import generate_episode

def policy_gradient(env, controller, episodes=1000, iters=100):
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

        # update policy using the episode
        controller.update_policy(episode)

        # save the best policy
        if total_reward > best_reward:
            best_reward = total_reward
            controller.save_policy()

    return reward_per_episode
