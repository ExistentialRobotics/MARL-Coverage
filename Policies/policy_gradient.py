import numpy as np
import torch
from . base_policy import Base_Policy
from . Networks.grid_rl_conv import Grid_RL_Conv
from torch.distributions.categorical import Categorical

class PolicyGradient(Base_Policy):

    def __init__(self, numrobot, action_space, learning_rate, obs_dim,
                 conv_channels, conv_filters, conv_activation, hidden_sizes,
                 hidden_activation, output_activation, deterministic=False):
        super().__init__(numrobot, action_space)
        self.num_actions = action_space.num_actions
        self._deterministic = deterministic
        action_dim = numrobot * self.num_actions

        # init policy network and optimizer
        self.policy_net = Grid_RL_Conv(action_dim, obs_dim, conv_channels, conv_filters, conv_activation, hidden_sizes, hidden_activation, output_activation)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def step(self, state):
        probs = self.policy_net(torch.from_numpy(state).float())

        # sample from each set of probabilities to get the actions
        ulis = []
        for i in range(self.numrobot):
            if(not self._deterministic):
                m = Categorical(probs[i * self.num_actions: (i + 1) * self.num_actions])
                ulis.append(m.sample())
            else:
                u = torch.argmax(probs[i * self.num_actions: (i + 1) * self.num_actions])
                ulis.append(u)
        return ulis

    def calc_gradient(self, state, action, r_return):
        probs = self.policy_net(torch.from_numpy(state).float())

        # calculate gradient
        loss = 0
        for i in range(self.numrobot):
            m = Categorical(probs[i * self.num_actions: (i + 1) * self.num_actions])
            loss -= m.log_prob(action[i]) * r_return
        loss.backward()

    def set_train(self):
        self.policy_net.train()

    def set_eval(self):
        self.policy_net.eval()

    def print_weights(self):
        for name, param in self.policy_net.named_parameters():
            print(param.detach().numpy())

