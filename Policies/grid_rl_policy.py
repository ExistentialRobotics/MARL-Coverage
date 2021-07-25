import numpy as np
import torch
from . base_policy import Base_Policy
from . Networks.grid_rl_conv import Grid_RL_Conv
from torch.distributions.categorical import Categorical

class Grid_RL_Policy(Base_Policy):

    def __init__(self, numrobot, action_space, learning_rate):
        super().__init__(numrobot, action_space)
        # init policy network and optimizer
        self.num_actions = action_space.num_actions
        self.policy_net = Grid_RL_Conv(numrobot * self.num_actions)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def step(self, state):
        probs = self.policy_net(torch.from_numpy(state).float())

        # sample from each set of probabilities to get the actions
        ulis = []
        for i in range(self.numrobot):
            m = Categorical(probs[i * self.num_actions: (i + 1) * self.num_actions])
            ulis.append(m.sample())
        return ulis

    def update(self, trajectory):
        pass
