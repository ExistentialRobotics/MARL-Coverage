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
                 gamma=0.9, tau=0.9, weight_decay=0.1, ani=False, model_path=None):
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
            self.next_q = torch.zeros((self.action_sets.shape[0], 1))

            # self.batch_q_vals = np.zeros((batch_size, self.action_sets.shape[0]))
            # self.batch_next_q = torch.zeros((batch_size, self.action_sets.shape[0]))
        else:
            # use grid rl conv instead of qnet
            action_dim = numrobot * self.num_actions

            # init q net
            self.q_net = Grid_RL_Conv(action_dim, obs_dim, conv_channels,
                                conv_filters, conv_activation, hidden_sizes,
                                      hidden_activation, None)

        # init with saved weights if testing saved model
        if model_path is not None:
            self.q_net.load_state_dict(torch.load(model_path))

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
        temp = self._epsilon
        if testing:
            self._epsilon = self._min_epsilon

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
                if(s <= self._epsilon):
                    ulis[i] = np.random.randint(self.num_actions)
        else:
            ulis = []
            for i in range(self.numrobot):
                #epsilon greedy check
                s = np.random.uniform()

                #epsilon greedy policy
                if(s > self._epsilon):
                    #greedy
                    u = torch.argmax(qvals[i * self.num_actions: (i + 1) * self.num_actions])
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

        #sampling episode(s)? from buffer and updating q-network
        N = len(episode)
        if self.batch_size is not None:
            N = self.batch_size
        states, actions, rewards, next_states = self._buff.samplebatch(N)

        #converting all the rewards in the episode to be total rewards instead of per robot reward if ani
        if self.ani:
            for i in range(len(rewards)):
                rewards[i] = np.sum(rewards[i])

        # convert to tensors
        states = torch.tensor(states).float()
        next_states = torch.tensor(next_states).float()
        rewards = torch.tensor(rewards).float()
        actions = torch.tensor(actions).long()
        states = torch.squeeze(states, axis=1)
        next_states = torch.squeeze(next_states, axis=1)

        # calculate network gradient
        loss = 0
        if self.ani:
            for i in range(N):
                loss += self.calc_gradient(states[i], actions[i], rewards[i], next_states[i], N)
        else:
            loss = self.calc_gradient(states, actions, rewards, next_states, N)

        #tracking loss
        self._lastloss = loss.item()
        print("Loss: " + str(self._lastloss))

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

    def calc_gradient(self, states, actions, rewards, next_states, batch_size):
        # calc q vals
        if self.ani:
            a = torch.unsqueeze(self.one_hot(actions), 0)
            qval = self.q_net(torch.unsqueeze(states, 0), a)
        else:
            qvals = self.q_net(states)

        if not self.ani:
            next_qvals = self.target_net(next_states)

        if self.ani:
            with torch.no_grad():
                for i in range(self.action_sets.shape[0]):
                    a = torch.unsqueeze(self.one_hot(self.action_sets[i]), 0)
                    self.next_q[i] = self.target_net(torch.unsqueeze(next_states, 0), a)
                next_q = torch.max(self.next_q)
                y = rewards + self._gamma*next_q

            #calculating mean squared error
            loss = 1.0/batch_size * (y-qval)**2
        else:
            # calculate gradient for q function
            loss = 0
            currq = torch.zeros(batch_size)
            for i in range(self.numrobot):
                with torch.no_grad():
                    next_q = torch.max(next_qvals[:, i * self.num_actions: (i + 1) * self.num_actions], 1).values
                    y = rewards[:, i] + self._gamma*next_q
                q_temp = qvals[:, i * self.num_actions: (i + 1) * self.num_actions]

                #TODO vectorize this for loop
                for j in range(q_temp.shape[0]):
                    currq[j] = q_temp[j, actions[j, i]]

                #calculating mean squared error
                loss += ((y-currq)**2).mean()
        loss.backward()
        return loss

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
