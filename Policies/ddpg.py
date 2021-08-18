import numpy as np
import torch
from . base_policy import Base_Policy
from . Networks.grid_rl_conv import Grid_RL_Conv
from torch.distributions.categorical import Categorical
from copy import deepcopy

class DDPG(Base_Policy):

    def __init__(self, actor, critic, buff, numrobot, num_actions, learning_rate, batch_size=100,
                 gamma=0.97, weight_decay=0.1, tau=0.9, model_path=None):
        super().__init__()
        self.numrobot = numrobot
        self.num_actions = num_actions

        #cpu vs gpu code
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # init nets
        self.actor = actor
        self.actor_target = deepcopy(actor)
        self.critic = critic
        self.critic_target = deepcopy(critic)

        #moving nets to gpu
        self.actor.to(self._device)
        self.actor_target.to(self._device)
        self.critic.to(self._device)
        self.critic_target.to(self._device)

        #replay buffer creation
        self.batch_size = batch_size
        self._buff = buff

        # init with saved weights if testing saved model
        if model_path is not None:
            self.actor.load_state_dict(torch.load(model_path))

        # init optimizers
        self.a_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.c_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate*10, weight_decay=weight_decay)

        #reward discounting
        self._gamma = gamma

        #polyak avg parameter
        self._tau = tau

        #tracking loss
        self._lastloss = 0

    def pi(self, state):
        state_tensor = (torch.from_numpy(state).float()).to(self._device)
        probs = self.actor(state_tensor)

        # sample from each set of probabilities to get the actions
        m = Categorical(logits=probs)
        return m.sample(), probs

    def update_policy(self):
        #zero gradients
        self.a_optimizer.zero_grad()
        self.c_optimizer.zero_grad()

        #sampling batch from buffer
        states, actions, rewards, next_states, done = self._buff.samplebatch(self.batch_size)

        # convert to tensors
        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).long()
        rewards = torch.from_numpy(rewards).float()
        next_states = torch.from_numpy(next_states).float()
        done = torch.from_numpy(done).long()

        #moving tensors to gpu
        states = states.to(self._device)
        actions = actions.to(self._device)
        rewards = rewards.to(self._device)
        next_states = next_states.to(self._device)
        done = done.to(self._device)

        # gradient calculation
        loss = self.calc_gradient(states, actions, rewards, next_states, done, self.batch_size)

        #tracking loss
        self._lastloss = loss.item()

        # update parameters
        self.a_optimizer.step()
        self.c_optimizer.step()

        #update target networks to soft follow main network
        with torch.no_grad():
            for c, c_targ in zip(self.critic.parameters(),
                                 self.critic_target.parameters()):
                c_targ.data.mul_(self._tau)
                c_targ.data.add_((1 - self._tau) * c.data)
            for a, a_targ in zip(self.actor.parameters(),
                                 self.actor_target.parameters()):
                a_targ.data.mul_(self._tau)
                a_targ.data.add_((1 - self._tau) * a.data)

        #paranoid about memory leak
        del states, next_states, rewards, actions

    def calc_gradient(self, states, actions, rewards, next_states, done, batch_size):
        # get q vals
        next_actions = self.actor_target(next_states)
        next_q = self.critic_target(next_states, next_actions)
        curr_q = self.critic(states, actions)

        # update critic
        y = rewards + self._gamma*(1 - done)*next_q
        c_loss = ((y-curr_q)**2).mean()
        c_loss.backward()

        # update actor
        a_loss = (-self.critic(states, self.actor(states))).mean()
        a_loss.backward()

        return c_loss

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
        states = torch.scueeze(states, axis=1)
        actions = torch.tensor(actions).float()
        rewards = torch.tensor(rewards).float()

        #calculate all discounted rewards efficiently
        for i in range(episode_len - 1):
            rewards[episode_len - 2 - i] += self._gamma*rewards[episode_len - 1 - i]

        #returning everything
        return states[:-1], actions, rewards

    def set_train(self):
        self.actor.train()

    def set_eval(self):
        self.actor.eval()

    def print_weights(self):
        for name, param in self.actor.named_parameters():
            print(param.detach().numpy())

    def getnet(self):
        return self.actor

    def printNumParams(self):
        pytorch_total_params = sum(p.numel() for p in self.actor.parameters() if p.requires_grad)
        print(str(pytorch_total_params) + " in the Policy Network")
