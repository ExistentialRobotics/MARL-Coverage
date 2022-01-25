import numpy as np
import torch
from . base_policy import Base_Policy
from . replaybuffer import ReplayBuffer
from copy import deepcopy

class AlphaZero(Base_Policy):

    def __init__(self, env, net, num_actions, obs_dim, policy_config, model_config, model_path=None):
        self._env = env # environment to run on
        self._net = net # neural network for prediction

        self._iters = policy_config["iters"]
        self._epi = policy_config["episodes"]
        self._mcts_sims = policy_config["mcts_sims"]

        self._Q = {} # state action dict
        self._P = {} # state action probabilities
        self._R = {} # rewards

        # cpu vs gpu code
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #moving net to gpu
        self._net.to(self._device)

        # loss and optimizer
        self._loss = torch.nn.CrossEntropyLoss()
        self._opt = torch.optim.Adam(self._net.parameters(),
                        lr=policy_config['lr'],
                        weight_decay=policy_config['weight_decay'])

        # performance metrics
        self._losses = []


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
        # I wrote this with the assumption that each datum is a tuple of the form:
        # (state, action probabilities, reward). I'll try to batch this as we move
        # along with the implementation

        # zero gradients
        self._optimizer.zero_grad()

        # calc loss for each data point
        losses = []
        for d in train_data:
            state, rollout_prob, reward = d
            action_prob = self.net()
            squ_e = (y-currq)**2
            cross_e = self.loss(input, target)
            loss = squ_e - cross_e
            loss.backward()

        # update parameters
        self._opt.step()

    def reset(self):
        pass
