import numpy as np
import torch
from . base_policy import Base_Policy
from . replaybuffer import ReplayBuffer
from copy import deepcopy

class AlphaZero(Base_Policy):

    def __init__(self, env, net, num_actions, obs_dim, policy_config, model_config, model_path=None):
        self._env = env# environment to run on
        self._net = net # neural network for prediction

        self._iters = policy_config["iters"]
        self._epi = policy_config["episodes"]
        self._mcts_sims = policy_config["mcts_sims"]

        self._Q = {}
        self._P = {}
        self._R = {}


    def policy_iteration(self):
        for i in range(self._iters):
            # get data from playing data with current neural net
            train_data = []
            for j in range(self._epi):
                train_data += self.rollout()

            # update neural net
            self.update(train_data)


    def rollout(self):
        # reset environment
        env.reset()
        train_data = []

        # perform mcts
        for _ in range(self._mcts_sims):
            train_data += self.mcts()

        return train_data


    def mcts(self):
        pass

    def pi(self):
        pass

    def update(self, train_data):
        pass

    def reset(self):
        pass
