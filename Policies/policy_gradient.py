import numpy as np
import torch
from . base_policy import Base_Policy
from . Networks.grid_rl_conv import Grid_RL_Conv
from torch.distributions.categorical import Categorical

class PolicyGradient(Base_Policy):

    def __init__(self, numrobot, action_space, learning_rate, obs_dim,
                 conv_channels, conv_filters, conv_activation, hidden_sizes,
                 hidden_activation, output_activation, gamma=0.9):
        super().__init__(numrobot, action_space)
        self.num_actions = action_space.num_actions
        action_dim = numrobot * self.num_actions

        # init policy network and optimizer
        self.policy_net = Grid_RL_Conv(action_dim, obs_dim, conv_channels, conv_filters, conv_activation, hidden_sizes, hidden_activation, output_activation)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        #reward discounting
        self._gamma = gamma

    def step(self, state):
        probs = self.policy_net(torch.from_numpy(state).float())

        # sample from each set of probabilities to get the actions
        ulis = []
        for i in range(self.numrobot):
            m = Categorical(probs[i * self.num_actions: (i + 1) * self.num_actions])
            ulis.append(m.sample())
        return ulis

    def update_policy(self, episode):
        self.optimizer.zero_grad()

        #converting all the rewards in the episode to be total rewards instead of per robot reward
        raw_rewards = []
        for i in range(len(episode)):
            raw_rewards.append(np.sum(episode[i][2]))

        #calculate all r_returns efficiently (with discounting)
        d_rewards = []
        d_rewards.append(raw_rewards[len(episode) - 1])
        for i in range(len(episode) - 1):
            d_rewards.insert(0, self._gamma*d_rewards[0]
                             + raw_rewards[len(episode) - 2 - i])

        #calculating gradients for each step of episode 
        for i in range(len(episode)):
            state = episode[i][0]
            action = episode[i][1]

            # increment gradient
            self.calc_gradient(state, action, d_rewards[i])

        # update parameters
        self.optimizer.step()

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

    def getnet(self):
        return self.policy_net


























