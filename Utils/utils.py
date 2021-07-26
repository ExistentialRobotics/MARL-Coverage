import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions import Bernoulli

def generate_episode(env, controller, iters=100):
    print("------------------------Generating Episode-------------------------")
    episode = []

    done = False
    state = env.reset()
    steps = 0
    while not done and steps != iters:
        # determine action
        action = controller.getControls(state)

        # step environment and save episode results
        next_state, reward = env.step(action)
        episode.append((state, action, reward))

        state = next_state
        steps += 1

    return episode, steps
