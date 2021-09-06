import numpy as np
import torch
from . base_policy import Base_Policy
from . replaybuffer import ReplayBuffer
from copy import deepcopy


class DRQN(Base_Policy):

    def __init__(self, q_net, buff, num_actions, policy_config, model_path=None):
        super().__init__()

        #policy config parameters
        self._epsilon = 1
        self._e_decay = policy_config['epsilon_decay']
        self._min_epsilon = policy_config['min_epsilon']
        self._testing = False
        self._testing_epsilon = policy_config['testing_epsilon']
        self._buff = buff
        self.batch_size = policy_config['batch_size']
        self._gamma = policy_config['gamma']
        self._tau = policy_config['tau']
        self.N = policy_config['steps']
        self.hidden_size = policy_config['hidden_size']
        self.curr_hidden = None

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

    def pi(self, state, start=False):
        '''
        This method takes the state as input and returns a list of actions, one
        for each robot.
        '''

        #epsilon greedy check
        s = np.random.uniform()

        # init hidden state if necessary
        if start:
            self.curr_hidden = (torch.zeros((1, 1, self.hidden_size)).to(self._device), torch.zeros((1, 1, self.hidden_size)).to(self._device))

        #epsilon greedy policy
        #if we are testing then we use a smaller testing epsilon
        if(s > self._epsilon or (self._testing and s > self._testing_epsilon)):
            # calc qvals, using no grad to avoid computing gradients
            with torch.no_grad():
                state_tensor = (torch.from_numpy(state).float()).to(self._device)
                qvals, hidden = self.q_net(state_tensor, self.curr_hidden)

                # set new hidden state
                self.curr_hidden = hidden
            #greedy
            u = torch.argmax(qvals)
        else:
            #random
            u = np.random.randint(self.num_actions)

        return u

    def update_policy(self, episode):
        #adding new data to buffer
        self._buff.addepisode(episode)

        #decaying epsilon
        self.decayEpsilon()

        #performing one gradient step per env step (like spinning up ddpg)
        self.update_policy_step()


    def update_policy_step(self):
        #zero gradients
        self.optimizer.zero_grad()

        #sampling batch from buffer
        # states, actions, rewards, next_states, done = self._buff.samplesequential(self.N)

        states, actions, rewards, next_states, done = self._buff.samplebatchsequential(self.batch_size, self.N)

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

        # print(states.shape)
        # print(actions.shape)
        # print(rewards.shape)

        # gradient calculation
        # loss = self.calc_gradient(states, actions, rewards, next_states, done, self.batch_size, self.N)
        loss = self.calc_batch_gradient(states, actions, rewards, next_states, done, self.batch_size, self.N)

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

    def calc_gradient(self, states, actions, rewards, next_states, done, batch_size, N):
        hidden = (torch.zeros((1, 1, self.hidden_size)).to(self._device), torch.zeros((1, 1, self.hidden_size)).to(self._device))
        hidden_t = (torch.zeros((1, 1, self.hidden_size)).to(self._device), torch.zeros((1, 1, self.hidden_size)).to(self._device))

        loss = 0
        for i in range(self.N):
            # calc q and next q
            qvals, hidden = self.q_net(states[i], hidden)
            next_qvals, hidden_t = self.target_net(next_states[i], hidden_t)

            # calculate gradient for q function
            with torch.no_grad():
                next_q = torch.max(next_qvals, 0).values
                y = rewards[i] + self._gamma*(1-done[i])*next_q

            #calculating mean squared error
            currq = qvals[actions[i]]
            loss += ((y-currq)**2).mean()
        loss = loss/N
        loss.backward()

        return loss

    def calc_batch_gradient(self, states, actions, rewards, next_states, done, batch_size, N):
        hidden = (torch.zeros((1, batch_size, self.hidden_size)).to(self._device),
                  torch.zeros((1, batch_size, self.hidden_size)).to(self._device))
        hidden_t = (torch.zeros((1, batch_size, self.hidden_size)).to(self._device),
                    torch.zeros((1, batch_size, self.hidden_size)).to(self._device))

        loss = 0
        for i in range(self.N):
            # calc q and next q
            qvals, hidden = self.q_net(states[i], hidden)
            next_qvals, hidden_t = self.target_net(next_states[i], hidden_t)

            # calculate gradient for q function
            with torch.no_grad():
                next_q = torch.max(next_qvals, 1).values
                y = rewards[i] + self._gamma*(1-done[i])*next_q

            #TODO vectorize this for loop
            currq = torch.zeros(batch_size).to(self._device)
            for j in range(batch_size):
                currq[j] = qvals[j, actions[i][j]]

            #calculating mean squared error
            loss += ((y-currq)**2).mean()
        loss = loss/N
        loss.backward()

        return loss

    def decayEpsilon(self):
        #decaying the epsilon
        if self._epsilon > self._min_epsilon:
            self._epsilon *= self._e_decay

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
