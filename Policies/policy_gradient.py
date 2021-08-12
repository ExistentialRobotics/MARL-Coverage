import numpy as np
import torch
from . base_policy import Base_Policy
from . Networks.grid_rl_conv import Grid_RL_Conv
from torch.distributions.categorical import Categorical

class PolicyGradient(Base_Policy):

    def __init__(self, numrobot, action_space, learning_rate, obs_dim,
                 conv_channels, conv_filters, conv_activation, hidden_sizes,
                 hidden_activation, output_activation, gamma=0.9, weight_decay=0.1,
                 model_path=None):
        super().__init__(numrobot, action_space)
        self.num_actions = action_space.num_actions
        action_dim = numrobot * self.num_actions

        # init policy network and optimizer
        self.policy_net = Grid_RL_Conv(action_dim, obs_dim, conv_channels, conv_filters, conv_activation, hidden_sizes, hidden_activation, output_activation)

        # init with saved weights if testing saved model
        if model_path is not None:
            self.policy_net.load_state_dict(torch.load(model_path))

        # init optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate, weight_decay=weight_decay)

        #reward discounting
        self._gamma = gamma

        #tracking loss
        self._lastloss = 0

    def step(self, state, testing):
        probs = self.policy_net(torch.from_numpy(state).float())

        # sample from each set of probabilities to get the actions
        ulis = []
        for i in range(self.numrobot):
            m = Categorical(logits=probs[i * self.num_actions: (i + 1) * self.num_actions])
            ulis.append(m.sample())
        return ulis

    def update_policy(self, episode):
        self.optimizer.zero_grad()

        #setting loss var to zero so we can increment it for each step of episode
        self._lastloss = 0

        #convert to tensors
        states = [e[0] for e in episode]
        actions = [e[1] for e in episode]
        rewards = [e[2] for e in episode]
        states = torch.tensor(states).float()
        actions = torch.tensor(actions).float()
        states = torch.squeeze(states, axis=1)
        rewards = torch.tensor(rewards).float()

        #calculate all discounted rewards (per robot) efficiently
        for i in range(len(episode) - 1):
            rewards[len(episode) - 2 - i] += self._gamma*rewards[len(episode) - 1 - i]

        # calculate network gradient
        self.calc_gradient(states, actions, rewards)

        # update parameters
        self.optimizer.step()

    def calc_gradient(self, states, actions, rewards):
        probs = self.policy_net(states)

        # calculate gradient
        loss = 0
        for i in range(self.numrobot):
            m = Categorical(logits=probs[:,i * self.num_actions: (i + 1) * self.num_actions])
            loss -= (m.log_prob(actions[:,i]) * rewards[:,i]).mean()
        loss.backward()
        self._lastloss += loss.item()

    def set_train(self):
        self.policy_net.train()

    def set_eval(self):
        self.policy_net.eval()

    def print_weights(self):
        for name, param in self.policy_net.named_parameters():
            print(param.detach().numpy())

    def getnet(self):
        return self.policy_net

    def printNumParams(self):
        pytorch_total_params = sum(p.numel() for p in self.policy_net.parameters() if p.requires_grad)
        print(str(pytorch_total_params) + " in the Policy Network")
