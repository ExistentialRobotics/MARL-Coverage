import numpy as np
import torch
import itertools
from . base_policy import Base_Policy
from . Networks.grid_rl_conv import Grid_RL_Conv
from . Networks.Qnet import Critic
from . replaybuffer import ReplayBuffer
from torch.distributions.categorical import Categorical
from copy import deepcopy


class VDN(Base_Policy):

    def __init__(self, q_net, buff, numrobot, num_actions, learning_rate, epsilon=0.999, min_epsilon=0.1,
                 batch_size=100, gamma=0.99, tau=0.9, weight_decay=0.1,
                 model_path=None):
        super().__init__()

        #cpu vs gpu code
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # num actions per robot, artifact of old code
        self.num_actions = num_actions
        self.numrobot = numrobot
        # init q net
        self.q_net = q_net

        # init with saved weights if testing saved model
        if model_path is not None:
            self.q_net.load_state_dict(torch.load(model_path))

        # init q net
        self.target_net = deepcopy(self.q_net)

        #moving nets to gpu
        self.q_net.to(self._device)
        self.target_net.to(self._device)

        #setting requires gradient in target net to false for all params
        for p in self.target_net.parameters():
            p.requires_grad = False

        #optimizer
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate, weight_decay=weight_decay)

        #epsilon-greedy parameters
        self._epsilon = 1
        self._e_decay = epsilon
        self._min_epsilon = min_epsilon

        #testing parameters
        self._testing_epsilon = 0.05
        self._testing = False

        #replay buffer creation
        self.batch_size = batch_size
        self._buff = buff

        #discount rate and q-net weighted average
        self._gamma = gamma
        self._tau = tau

    def pi(self, state):
        '''
        This method takes the state as input and returns a list of actions, one
        for each robot.
        '''
        # calc qvals, using no grad to avoid computing gradients
        with torch.no_grad():
            state_tensor = (torch.from_numpy(state).float()).to(self._device)
            qvals = self.q_net(state_tensor)

        #epsilon greedy check
        s = np.random.uniform()

        #generating the action for each robot
        ulis = []
        for i in range(self.numrobot):
            #if we are testing then we use a smaller testing epsilon
            if(s > self._epsilon or (self._testing and s>self._testing_epsilon)):
                #greedy
                u = torch.argmax(qvals[i * self.num_actions: (i + 1) * self.num_actions])
                u = u.item()
            else:
                #random
                u = np.random.randint(self.num_actions)

            #adding controls
            ulis.append(u)
        return ulis

    def update_policy(self, episode):
        #adding new data to buffer
        self._buff.addepisode(episode)

        #decaying the epsilon
        if self._epsilon > self._min_epsilon:
            self._epsilon *= self._e_decay

        #zero gradients
        self.optimizer.zero_grad()

        #sampling episodes
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
        self.optimizer.step()

        #update target network to soft follow main network
        with torch.no_grad():
            for q, q_targ in zip(self.q_net.parameters(),
                                 self.target_net.parameters()):
                q_targ.data.mul_(self._tau)
                q_targ.data.add_((1 - self._tau) * q.data)

        #paranoid about memory leak
        del states, next_states, rewards, actions

    def calc_gradient(self, states, actions, rewards, next_states, done, batch_size):
        # calc q and next q
        qvals = self.q_net(states)
        next_qvals = self.target_net(next_states)

        # calculate gradient for q function
        y = torch.zeros(batch_size).to(self._device)
        currq = torch.zeros(batch_size).to(self._device)
        for i in range(self.numrobot):
            with torch.no_grad():
                next_q = torch.max(next_qvals[:, i * self.num_actions: (i + 1) * self.num_actions], 1).values
                y += self._gamma*(1-done)*next_q
            q_temp = qvals[:, i * self.num_actions: (i + 1) * self.num_actions]

            #TODO vectorize this for loop
            for j in range(q_temp.shape[0]):
                currq[j] += q_temp[j, actions[j, i]]

        y += rewards
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
