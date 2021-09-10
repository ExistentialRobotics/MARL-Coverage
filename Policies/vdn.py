import numpy as np
import torch
import itertools
from . base_policy import Base_Policy
from . Networks.grid_rl_conv import Grid_RL_Conv
from . Networks.gnn import GNN
from . replaybuffer import ReplayBuffer
from torch.distributions.categorical import Categorical
from copy import deepcopy


class VDN(Base_Policy):
    '''Multi-Agent Cooperative Policy with centralized training and decentralized
    execution. Takes in observations for all agents and outputs actions for
    each agent. Trains a Q-net: individual observation to action values for
    each action, and computes the joint action values as a sum of the
    individual action values. The same Q-net is used for all agents, and is
    trained by minimizing the TD-error from samples in the replay buffer.
    '''

    def __init__(self, q_net, num_actions, obs_dim, numrobot, policy_config,
                 model_path=None):
        super().__init__()

        #policy config parameters
        self._epsilon = 1
        self._e_decay = policy_config['epsilon_decay']
        self._min_epsilon = policy_config['min_epsilon']
        self._testing = False
        self._testing_epsilon = policy_config['testing_epsilon']
        buffer_maxsize = policy_config['buffer_size']
        self._buff = ReplayBuffer((numrobot,) + obs_dim, numrobot, buffer_maxsize)
        self.batch_size = policy_config['batch_size']
        self._gamma = policy_config['gamma']
        self._tau = policy_config['tau']

        #cpu vs gpu code
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # init q net
        self.q_net = q_net
        self.num_actions = num_actions

        # init with saved weights if testing saved model
        if model_path is not None:
            self.q_net.load_state_dict(torch.load(model_path))

        # init q target net
        self.target_net = deepcopy(self.q_net)

        #moving nets to gpu
        self.q_net.to(self._device)
        self.target_net.to(self._device)

        #setting requires gradient in target net to false for all params
        for p in self.target_net.parameters():
            p.requires_grad = False

        #optimizer
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                        lr=policy_config['lr'],
                        weight_decay=policy_config['weight_decay'])

    def pi(self, observations):
        '''
        This method takes observations as input and returns an array of actions,
        one for each agent. The observations should be stacked in a numpy array,
        along the first axis.
        '''
        obs, adj_m = observations

        # calc qvals for each agent, using no grad to avoid computing gradients
        with torch.no_grad():
            obs_tensor = (torch.from_numpy(obs).float()).to(self._device)
            qvals = self.q_net(obs_tensor)

        #epsilon greedy check
        s = np.random.uniform()

        #generating the action for each robot
        if(s > self._epsilon or (self._testing and s>self._testing_epsilon)):
            #greedy
            actions = torch.argmax(qvals, dim=1).cpu().numpy()
        else:
            #random
            actions = np.random.randint(self.num_actions)

        return actions

    def update_policy(self, episode):
        #adding new data to buffer
        self._buff.addepisode(episode)

        #decaying the epsilon
        if self._epsilon > self._min_epsilon:
            self._epsilon *= self._e_decay

        #zero gradients
        self.optimizer.zero_grad()

        #sampling episodes
        obs, actions, rewards, next_obs, done = self._buff.samplebatch(self.batch_size)

        # convert to tensors
        obs = torch.from_numpy(obs).float()
        actions = torch.from_numpy(actions).long()
        rewards = torch.from_numpy(rewards).float()
        next_obs = torch.from_numpy(next_obs).float()
        done = torch.from_numpy(done).long()

        #moving tensors to gpu
        obs = obs.to(self._device)
        actions = actions.to(self._device)
        rewards = rewards.to(self._device)
        next_obs = next_obs.to(self._device)
        done = done.to(self._device)

        # gradient calculation
        loss = self.calc_gradient(obs, actions, rewards, next_obs, done, self.batch_size)

        #tracking loss
        self._lastloss = loss.item()

        # update parameters
        self.optimizer.step()

        #update target network to soft follow main network
        with torch.no_grad():
            for q, q_targ in zip(self.q_net.parameters(),
                                 self.target_net.parameters()):
                q_targ.data.mul_(self._tau)
                q_targ.data.add_((1 - self._tau) * q.data)

        #paranoid about memory leak
        del obs, next_obs, rewards, actions

    def calc_gradient(self, obs, actions, rewards, next_obs, done, batch_size):
        # calc q and next q
        qvals = self.q_net(obs)
        next_qvals = self.target_net(next_obs)

        # calculate gradient for q function
        y = torch.zeros(batch_size).to(self._device)
        currq = torch.zeros(batch_size).to(self._device)

        #calculating the q values at next step (next_obs)
        with torch.no_grad():
            next_q = torch.max(next_qvals, dim=2).values
            next_q = torch.sum(next_q, dim=1)
            y = rewards + self._gamma*(1-done)*next_q

        #calculating the q values for current step and current action
        currq = torch.squeeze(qvals.gather(2, actions.unsqueeze(2)))
        currq = torch.sum(currq, dim=1)

        #calculating mean squared error
        loss = ((y-currq)**2).mean()
        loss.backward()

        return loss

    def set_train(self):
        '''
        Use this method to set the policy in a mode for training.
        '''
        self._testing = False
        self.q_net.train()

    def set_eval(self):
        '''
        Use this method to set the policy in a mode for testing.
        '''
        self._testing = True
        self.q_net.eval()

    def print_weights(self):
        for name, param in self.q_net.named_parameters():
            print(param.detach().numpy())

    def getnet(self):
        return self.q_net

    def printNumParams(self):
        pytorch_total_params = sum(p.numel() for p in self.q_net.parameters() if p.requires_grad)
        print(str(pytorch_total_params) + " in the Q Network")
