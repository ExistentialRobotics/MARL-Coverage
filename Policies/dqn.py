import numpy as np
import torch
import itertools
from . base_policy import Base_Policy
from . Networks.grid_rl_conv import Grid_RL_Conv
from . Networks.Qnet import Critic
from torch.distributions.categorical import Categorical
from copy import deepcopy
from . replaybuffer import ReplayBuffer

class DQN(Base_Policy):

    def __init__(self, numrobot, action_space, learning_rate, obs_dim,
                 conv_channels, conv_filters, conv_activation, hidden_sizes,
                 hidden_activation, epsilon=0.999,
                 min_epsilon=0.1, buffer_size = 1000, batch_size=None,
                 gamma=0.9, tau=0.9, weight_decay=0.1, ani=False):
        super().__init__(numrobot, action_space)

        self.ani = ani
        self.num_actions = action_space.num_actions

        # init q network and target-q-network
        if self.ani:
            # Generate action list using cartesian product. This is def not the
            # fastest way to do this, but I thought it would be the most simple
            # and easiest to understand given that we aren't gonna stick to DQN
            # in the long run. Plus it shouldn't affect runtime drastically
            # since it's just ran once in __init__.
            self.actions = [_ for _ in range(self.num_actions)]
            self.action_sets = np.array([p for p in itertools.product(self.actions, repeat=numrobot)])
            self.act_dim = self.num_actions * self.numrobot

            # create batched action sets
            self.batched_action_sets = np.stack([self.action_sets for _ in range(batch_size)], axis=0)

            # init q net
            self.q_net = Critic(obs_dim, self.act_dim, conv_channels, conv_filters,
                                conv_activation, hidden_sizes, hidden_activation)

            # init q val array
            self.q_vals = torch.zeros((self.action_sets.shape[0], 1))
            self.batch_q_vals = np.zeros((batch_size, self.action_sets.shape[0]))
            self.batch_next_q = torch.zeros((batch_size, self.action_sets.shape[0]))
        else:
            # use grid rl conv instead of qnet
            action_dim = numrobot * self.num_actions
            self.q_net = Grid_RL_Conv(action_dim, obs_dim, conv_channels,
                                conv_filters, conv_activation, hidden_sizes,
                                      hidden_activation, None)
        # init q net
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

    def one_hot(self, action_set, batch=False):
        if batch:
            oh = torch.zeros((self.batch_size, self.act_dim))
        else:
            oh = torch.zeros((self.act_dim,))

        for i in range(action_set.shape[0]):
            if batch:
                oh[:, i * self.num_actions: (i + 1) * self.num_actions][action_set[i]] = 1
            else:
                oh[i * self.num_actions: (i + 1) * self.num_actions][action_set[i]] = 1
        return oh


    def step(self, state, testing):
        if not self.ani:
            qvals = self.q_net(torch.from_numpy(state).float())

        #choosing the action with highest q-value or choose random with p(epsilon)
        if self.ani:
            # iterate over every possible set of actions
            for i in range(self.action_sets.shape[0]):
                a = torch.unsqueeze(self.one_hot(self.action_sets[i]), 0)
                self.q_vals[i] = self.q_net(torch.from_numpy(state).float(), a)

            # greedy
            ulis = self.action_sets[torch.argmax(self.q_vals)]

            #epsilon greedy check
            for i in range(ulis.shape[0]):
                s = np.random.uniform()

                #epsilon greedy policy
                if(s <= self._epsilon or testing):
                    ulis[i] = np.random.randint(self.num_actions)
        else:
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

        #setting loss to zero so we can increment
        self._lastloss = 0

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

        #incrementing the loss
        self._lastloss += loss.item()

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

        #converting all the rewards in the episode to be total rewards instead of per robot reward if ani
        if self.ani:
            for i in range(len(reward)):
                reward[i] = np.sum(reward[i])

        # convert to tensors
        state = torch.tensor(state).float()
        next_state = torch.tensor(next_state).float()
        reward = torch.tensor(reward).float()
        action = torch.tensor(action).long()
        state = torch.squeeze(state, axis=1)
        next_state = torch.squeeze(next_state, axis=1)

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

        #paranoid about memory leak
        del state, next_state, reward, action

    def calc_gradient(self, state, action, reward, next_state, batch_size):
        # calc q vals
        if self.ani:
            a = self.one_hot(action, batch=True)
            qvals = self.q_net(state, a)
        else:
            qvals = self.q_net(state)

        if not self.ani:
            next_qvals = self.target_net(next_state)

        if self.ani:
            with torch.no_grad():
                for i in range(self.action_sets.shape[0]):
                    a = self.one_hot(self.batched_action_sets[:, i], batch=True)
                    self.batch_next_q[:, i] = self.target_net(next_state.float(), a)
                next_q = torch.max(self.batch_next_q, 1).values
                y = reward + self._gamma*next_q

            #calculating mean squared error
            loss = ((y-qvals)**2).mean()
        else:
            # calculate gradient for q function
            loss = 0
            currq = torch.zeros(batch_size)
            for i in range(self.numrobot):
                with torch.no_grad():
                    next_q = torch.max(next_qvals[:, i * self.num_actions: (i + 1) * self.num_actions], 1).values
                    y = reward[:, i] + self._gamma*next_q
                q_temp = qvals[:, i * self.num_actions: (i + 1) * self.num_actions]

                #TODO vectorize this for loop
                for j in range(q_temp.shape[0]):
                    currq[j] = q_temp[j, action[j, i]]

                #calculating mean squared error
                loss += ((y-currq)**2).mean()
        loss.backward()

        #tracking loss
        self._lastloss = loss.item()
        print("Loss: " + str(self._lastloss))

        #paranoid about memory leak
        # del loss, next_qvals, qvals, currq, q_temp, y, next_q

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
