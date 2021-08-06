import numpy as np
import torch
from . base_policy import Base_Policy
from . Networks.grid_rl_conv import Grid_RL_Conv
from torch.distributions.categorical import Categorical
from copy import deepcopy
from . replaybuffer import ReplayBuffer

class DQN(Base_Policy):

    def __init__(self, numrobot, action_space, learning_rate, obs_dim,
                 conv_channels, conv_filters, conv_activation, hidden_sizes,
                 hidden_activation, output_activation, epsilon=0.999,
                 min_epsilon=0.1, buffer_size = 1000, batch_size=None,
                 gamma=0.9, tau=0.9, weight_decay=0.1):
        super().__init__(numrobot, action_space)

        self.num_actions = action_space.num_actions

        # init q network and target-q-network
        action_dim = numrobot * self.num_actions
        self.q_net = Grid_RL_Conv(action_dim, obs_dim, conv_channels,
                            conv_filters, conv_activation, hidden_sizes,
                                hidden_activation, output_activation)
        self.target_net = deepcopy(self.q_net)

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

        #replay buffer creation
        self.batch_size = batch_size
        self._buff = ReplayBuffer(buffer_size)

        self._gamma = gamma
        self._tau = tau


    def step(self, state, testing):
        qvals = self.q_net(torch.from_numpy(state).float())

        #choosing the action with highest q-value or choose random with p(epsilon)
        ulis = []
        for i in range(self.numrobot):
            #epsilon greedy check
            s = np.random.uniform()

            #epsilon greedy policy
            if(s > self._epsilon or testing):
                #greedy
                u = torch.argmax(qvals[i * self.num_actions: (i + 1) * self.num_actions])
            else:
                #random
                u = np.random.randint(self.num_actions)

            #adding controls
            ulis.append(u)
        return ulis

    def update_policy_old(self, episode):
        #adding new data to buffer
        self._buff.addepisode(episode)

        #decaying the epsilon
        self._epsilon *= self._e_decay

        #zero gradients
        self.optimizer.zero_grad()

        #sampling episode(s)? from buffer and updating q-network
        N = len(episode)
        if self.batch_size is not None:
            N = self.batch_size
        state, action, reward, next_state = self._buff.samplebatch(N)

        for i in range(N):
            self.calc_gradient_old(state[i], action[i], reward[i], next_state[i], N)

        # update parameters
        self.optimizer.step()

        #update target network to soft follow main network
        with torch.no_grad():
            for q, q_targ in zip(self.q_net.parameters(),
                                 self.target_net.parameters()):
                q_targ.data.mul_(self._tau)
                q_targ.data.add_((1 - self._tau) * q.data)

    def calc_gradient_old(self, state, action, reward, next_state, batch_size):
        qvals = self.q_net(torch.from_numpy(state).float())
        next_qvals = self.target_net(torch.from_numpy(next_state).float())

        # calculate gradient for q function
        loss = 0
        for i in range(self.numrobot):
            with torch.no_grad():
                next_q = torch.max(next_qvals[i * self.num_actions: (i + 1) * self.num_actions])
                y = reward[i] + self._gamma*next_q
            currq = (qvals[i * self.num_actions: (i + 1) * self.num_actions])[action[i]]

            #calculating mean squared error
            loss += 1.0/batch_size* (y-currq)**2
        loss.backward()

    def update_policy(self, episode):
        #adding new data to buffer
        self._buff.addepisode(episode)

        #decaying the epsilon
        if self._epsilon > self._min_epsilon:
            self._epsilon *= self._e_decay

        #zero gradients
        self.optimizer.zero_grad()

        #sampling episode(s)? from buffer and updating q-network
        N = len(episode)
        if self.batch_size is not None:
            N = self.batch_size
        state, action, reward, next_state = self._buff.samplebatch(N)
        state = np.squeeze(state, axis=1)
        next_state = np.squeeze(next_state, axis=1)

        # calculate network gradient
        self.calc_gradient(state, action, reward, next_state, state.shape[0])

        # update parameters
        self.optimizer.step()

        #update target network to soft follow main network
        with torch.no_grad():
            for q, q_targ in zip(self.q_net.parameters(),
                                 self.target_net.parameters()):
                q_targ.data.mul_(self._tau)
                q_targ.data.add_((1 - self._tau) * q.data)

    def calc_gradient(self, state, action, reward, next_state, batch_size):
        # convert to tensors
        state = torch.from_numpy(state).float()
        next_state = torch.from_numpy(next_state).float()
        reward = torch.from_numpy(reward).float()
        action = torch.from_numpy(action).long()

        # calc q vals
        qvals = self.q_net(state)
        next_qvals = self.target_net(next_state)

        # calculate gradient for q function
        loss = 0
        for i in range(self.numrobot):
            with torch.no_grad():
                next_q = torch.max(next_qvals[:, i * self.num_actions: (i + 1) * self.num_actions], 1).values
                y = reward[:, i] + self._gamma*next_q
            q_temp = qvals[:, i * self.num_actions: (i + 1) * self.num_actions]

            #TODO vectorize this for loop
            currq = torch.zeros(batch_size)
            for j in range(q_temp.shape[0]):
                currq[j] = q_temp[j, action[:, i][j]]

            #calculating mean squared error
            loss += ((y-currq)**2).mean()
        loss.backward()

    def set_train(self):
        self.q_net.train()

    def set_eval(self):
        self.q_net.eval()

    def print_weights(self):
        for name, param in self.q_net.named_parameters():
            print(param.detach().numpy())

    def getnet(self):
        return self.q_net

    def printNumParams(self):
        pytorch_total_params = sum(p.numel() for p in self.q_net.parameters() if p.requires_grad)
        print(str(pytorch_total_params) + " in the Q Network")
