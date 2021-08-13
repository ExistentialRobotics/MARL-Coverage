import numpy as np
import torch
from . base_policy import Base_Policy
from . Networks.grid_rl_conv import Grid_RL_Conv
from torch.distributions.categorical import Categorical

class VPG(Base_Policy):

    def __init__(self, actor, critic, numrobot, action_space, learning_rate,
                 gamma=0.97, lam = 0.95, weight_decay=0.1, model_path=None, gae=False):
        super().__init__()
        self.numrobot = numrobot
        self.num_actions = action_space.num_actions

        # init policy network and optimizer
        self.policy_net = actor
        self.critic = critic
        self.gae = gae

        # init with saved weights if testing saved model
        if model_path is not None:
            self.policy_net.load_state_dict(torch.load(model_path))

        # init optimizer
        self.a_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.c_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate*10, weight_decay=weight_decay)

        #reward discounting
        self._gamma = gamma
        self._lam = lam

        #tracking loss
        self._lastloss = 0

        #TODO fix hardcoding of horizon
        # self.baseline = torch.zeros(100)
        #window for the exponentially moving average
        # self.baselinecount = 100*self.numrobot

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

        #converting episode data to tensors and gathering intermediate computations
        states, actions, rewards, advantages, values = self.processEpisode(episode)

        # calculate network gradient
        self.calc_gradient(states, actions, rewards, advantages, values)

        # update parameters
        self.a_optimizer.step()
        self.c_optimizer.step()

    def calc_gradient(self, states, actions, rewards, advantages, values):
        probs = self.policy_net(states)

        # calc loss
        m = Categorical(logits=probs)
        if self.gae:
            a_loss = -(m.log_prob(actions) * advantages).mean()
            c_loss = (rewards - values).pow(2).mean()
        else:
            a_loss = -(m.log_prob(actions) * advantages.detach()).mean()
            c_loss = advantages.pow(2).mean()

        # backprop
        a_loss.backward()
        c_loss.backward()
        self._lastloss += a_loss.item()

    def processEpisode(self, episode):
        '''
        Calculates the value function, normalized advantages, and discounted reward for an episode.
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

        #combining all individual rewards
        rewards = torch.tensor(rewards).float()
        rewards = torch.sum(rewards, 1)

        #evaluate critic on states
        values = torch.squeeze(self.critic(states), 1)

        if self.gae:
            with torch.no_grad():
                #calculating the td-error at each timestep
                deltas = self._gamma*values[1:] + rewards - values[:-1]

                #calculating the discounted sum of td errors from each timestep onward
                for i in range(episode_len):
                    deltas[episode_len - 2 - i] += self._gamma*self._lam*deltas[episode_len - 1 - i]

                #normalizing the advantages
                mu = torch.mean(deltas, 0)
                std = torch.std(deltas)
                advantages = (deltas - mu)/std
            #confirming that advantages have mean 0 variance 1
            # print(torch.mean(advantages, 0))
            # print(torch.std(advantages))

        #calculate all discounted rewards efficiently
        for i in range(episode_len - 1):
            rewards[episode_len - 2 - i] += self._gamma*rewards[episode_len - 1 - i]

        if not self.gae:
            advantages = rewards.detach() - values[:rewards.shape[0]]

        #returning everything
        return states[:-1], actions, rewards, advantages, values[:-1]

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
