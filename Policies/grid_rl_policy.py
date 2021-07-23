import numpy as np
import torch
from . base_policy import Base_Policy
from . Networks.grid_rl_conv import Grid_RL_Conv
from torch.distributions.categorical import Categorical

class Grid_RL_Policy(Base_Policy):

    def __init__(self, num_output, action_space, learning_rate):
        super().__init__(num_output, action_space)

        # init policy network and optimizer
        self.policy_net = Grid_RL_Conv(num_output)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def step(self, state):
        probs = self.policy_net(torch.from_numpy(state).float())
        m = Categorical(probs)
        return m.sample()

    def update(self, trajectory):
        pass
