import numpy as np
import torch
from . base_policy import Base_Policy
from . Networks.grid_rl_conv import Grid_RL_Conv
from torch.distributions.categorical import Categorical

class VPG(Base_Policy):

    def __init__(self, actor, critic, numrobot, action_space, learning_rate,
                 gamma=0.9, weight_decay=0.1, model_path=None):
        super().__init__()
        self.numrobot = numrobot
        self.num_actions = action_space.num_actions

        # init policy network and optimizer
        self.policy_net = actor
        self.critic = critic

        # init with saved weights if testing saved model
        if model_path is not None:
            self.policy_net.load_state_dict(torch.load(model_path))

        # init optimizer
        self.a_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.c_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate, weight_decay=weight_decay)

        #reward discounting
        self._gamma = gamma

        #tracking loss
        self._lastloss = 0

        #TODO fix hardcoding of horizon
        self.baseline = torch.zeros(100)
        #window for the exponentially moving average
        self.baselinecount = 100*self.numrobot

    def pi(self, state):
        probs = self.policy_net(torch.from_numpy(state).float())

        # sample from each set of probabilities to get the actions
        m = Categorical(logits=probs)
        return m.sample()

    def update_policy(self, episode):
        # reset opt gradients
        self.a_optimizer.zero_grad()
        self.c_optimizer.zero_grad()

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

        #combining all individual rewards
        rewards = torch.sum(rewards, 1)

        #updating the exponentially moving average
        self.baseline -= self.numrobot*self.baseline/self.baselinecount
        self.baseline += rewards/self.baselinecount

        #calculate all discounted rewards (per robot) efficiently
        for i in range(len(episode) - 1):
            rewards[len(episode) - 2 - i] += self._gamma*rewards[len(episode) - 1 - i]

        # calculate network gradient
        self.calc_gradient(states, actions, rewards)

        # update parameters
        self.a_optimizer.step()
        self.c_optimizer.step()

    def calc_gradient(self, states, actions, rewards):
        probs = self.policy_net(states)

        # calc advantage between actor and critic
        adv = rewards.detach() - self.critic(states)

        # calc loss
        m = Categorical(logits=probs)
        a_loss = (m.log_prob(actions) * adv.detach()).mean()
        c_loss = adv.pow(2).mean()

        # backprop
        a_loss.backward()
        c_loss.backward()
        self._lastloss += a_loss.item()

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
