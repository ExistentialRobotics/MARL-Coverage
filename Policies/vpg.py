import numpy as np
import torch
from . base_policy import Base_Policy
from . Networks.grid_rl_conv import Grid_RL_Conv
from torch.distributions.categorical import Categorical

class VPG(Base_Policy):

    def __init__(self, actor, numrobot, action_space, learning_rate,
                 gamma=0.97, weight_decay=0.1, model_path=None):
        super().__init__()
        self.numrobot = numrobot
        self.num_actions = action_space.num_actions

        # init policy network and optimizer
        self.policy_net = actor

        # init with saved weights if testing saved model
        if model_path is not None:
            self.policy_net.load_state_dict(torch.load(model_path))

        # init optimizer
        self.a_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate, weight_decay=weight_decay)

        #reward discounting
        self._gamma = gamma

        #tracking loss
        self._lastloss = 0

    def pi(self, state):
        probs = self.policy_net(torch.from_numpy(state).float())

        # sample from each set of probabilities to get the actions
        m = Categorical(logits=probs)
        return m.sample()

    def update_policy(self, episode):
        # reset opt gradients
        self.a_optimizer.zero_grad()

        #setting loss var to zero so we can increment it for each step of episode
        self._lastloss = 0

        #converting episode data to tensors and gathering intermediate computations
        states, actions, rewards = self.processEpisode(episode)

        # calculate network gradient
        self.calc_gradient(states, actions, rewards)

        # update parameters
        self.a_optimizer.step()

    def calc_gradient(self, states, actions, rewards):
        probs = self.policy_net(states)

        # calc loss
        m = Categorical(logits=probs)
        # print(rewards)
        a_loss = -(m.log_prob(actions) * rewards).mean()

        # backprop
        a_loss.backward()
        self._lastloss += a_loss.item()

    def processEpisode(self, episode):
        '''
        Calculates the value function and discounted reward for an episode.
        '''
        #raw episode data
        episode_len = len(episode)
        states = [e[0] for e in episode]
        actions = [e[1] for e in episode]
        rewards = [e[2] for e in episode]
        next_states = [e[3] for e in episode]

        #state preprocessing
        states.append(next_states[-1])
        states = torch.tensor(states).float()
        states = torch.squeeze(states, axis=1)
        actions = torch.tensor(actions).float()
        rewards = torch.tensor(rewards).float()

        #calculate all discounted rewards efficiently
        for i in range(episode_len - 1):
            rewards[episode_len - 2 - i] += self._gamma*rewards[episode_len - 1 - i]

        #returning everything
        return states[:-1], actions, rewards

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
