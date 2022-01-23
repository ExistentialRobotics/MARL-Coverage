import numpy as np
import torch
from . base_policy import Base_Policy
from . replaybuffer import ReplayBuffer
from copy import deepcopy

class AlphaZero(Base_Policy):

    def __init__(self, network):
        self._network = network # neural network for prediction

    def mcts(self):

    def pi(self):

    def update(self):

    def reset(self):
