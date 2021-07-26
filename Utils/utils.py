import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions import Bernoulli

def generate_episode(env, controller, iters=100):
    episode = []

    done = False
    state = env.reset()
    i = 0
    while not done or i == iters:
        # determine action
        state = torch.from_numpy(state).float()
        probs = policy_net(state)
        m = Categorical(probs)
        action = m.sample()

        # step environment and save episode results
        next_state, reward, done, info = env.step(action.data.numpy().astype(int)[0])
        episode.append((state, action, reward))

        state = next_state
        steps += 1

    return episode, steps
