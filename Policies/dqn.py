import numpy as np
import torch
import itertools
from . base_policy import Base_Policy
from . Networks.grid_rl_conv import Grid_RL_Conv
from . Networks.Qnet import Critic
from torch.distributions.categorical import Categorical
from copy import deepcopy

class ReplayBuffer(object):
    '''
    Stores state transitions (state, action, reward, next state)
    '''
    def __init__(self, maxsize):
        super().__init__()
        self.resetbuffer()
        self._maxsize = maxsize
        self._size = 0

    def resetbuffer(self):
        self._state = []
        self._action = []
        self._reward = []
        self._nextstate = []

    def addtransition(self, state, action, reward, next_state):
        #checking if buffer is full and removing first element
        if self._size == self._maxsize:
            self._state.pop(0)
            self._action.pop(0)
            self._reward.pop(0)
            self._nextstate.pop(0)

            self._size -= 1

        #adding new data
        self._state.append(state)
        self._action.append(action)
        self._reward.append(reward)
        self._nextstate.append(next_state)

        #incrementing size
        self._size += 1

    def addepisode(self, episode):
        for i in range(len(episode)):
            self.addtransition(episode[i][0], episode[i][1], episode[i][2], episode[i][3])

    def sampletransition(self):
        #returns a random transition in the replay buffer
        index = np.random.randint(self._size)
        return self._state[index], self._action[index], self._reward[index], self._nextstate[index]

    def samplebatch(self, N):
        states = []
        actions = []
        rewards = []
        next_states = []
        for i in range(N):
            state, action, reward, next_state = self.sampletransition()
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
        return states, actions, rewards, next_states
        

class DQN(Base_Policy):

    def __init__(self, numrobot, action_space, learning_rate, obs_dim,
                 conv_channels, conv_filters, conv_activation, hidden_sizes,
                 hidden_activation, epsilon=0.999,
                 min_epsilon=0.1, buffer_size = 1000, batch_size=None,
                 gamma=0.9, tau=0.9, weight_decay=0.1, model_path=None):
        super().__init__(numrobot, action_space)

        # save num actions as instance var
        self.num_actions = action_space.num_actions

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

    def step(self, state, testing):
        # calc qvals
        qvals = self.q_net(torch.from_numpy(state).float())

        # set eps
        temp = self._epsilon
        if testing:
            self._epsilon = self._min_epsilon

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

        # convert to tensors
        states = torch.tensor(states).float()
        next_states = torch.tensor(next_states).float()
        rewards = torch.tensor(rewards).float()
        actions = torch.tensor(actions).long()
        states = torch.squeeze(states, axis=1)
        next_states = torch.squeeze(next_states, axis=1)

        # gradient calculation
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
        # calc q and next q
        qvals = self.q_net(states)
        next_qvals = self.target_net(next_states)

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
