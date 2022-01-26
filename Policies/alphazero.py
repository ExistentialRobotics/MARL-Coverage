import numpy as np
import torch
from . base_policy import Base_Policy
from . replaybuffer import ReplayBuffer
from . mcts import MCTS, Node
from copy import deepcopy

class AlphaZero(Base_Policy):

    def __init__(self, env, sim, net, num_actions, obs_dim, policy_config, model_path=None):
        self._env = env # environment to run on
        self._net = net # neural network for prediction

        self._iters     = policy_config["train_iters"]
        self._epi       = policy_config["episodes"]
        self._testing   = False

        # cpu vs gpu code
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #moving net to gpu
        self._net.to(self._device)

        # loss and optimizer
        self._loss = torch.nn.CrossEntropyLoss()
        self._opt = torch.optim.Adam(self._net.parameters(),
                        lr=policy_config['lr'],
                        weight_decay=policy_config['weight_decay'])

        # instanciate MCTS
        self.tree = MCTS(sim, self._net, policy_config["mcts_sims"])

        # performance metrics
        self._losses = []


    def policy_iteration(self):
        for i in range(self._iters):
            # get data from playing data with current neural net
            train_data = []
            for j in range(self._epi):
                train_data += self.rollout()

            # update neural net
            self.update(train_data)


    def rollout(self, ignore_done=True):
        # reset environment
        state = env.reset()
        episode = []

        # play out an episode
        done = False
        while not done:
            # determine action
            probs = self.tree.pi(state)
            action = np.random.choice([0, 1, 2, 3], 1, p=probs)

            # step environment and save episode results
            next_state, reward = env.step(action)

            # determine if episode is completed
            done = env.done()

            #checking if done happened because we ran out of time and possibly ignoring it
            new_done = done
            if ignore_done and done and env._currstep == env._maxsteps:
                new_done = False

            #adding variables to episode
            episode.append((state, action, reward, next_state, new_done))
            state = next_state
            total_reward += reward

        return episode


    def pi(self, state):
        self.tree.pi(state)

    def update(self, train_data):
        # I wrote this with the assumption that each datum is a tuple of the form:
        # (state, action probabilities, reward). I'll try to batch this as we move
        # along with the implementation

        # zero gradients
        self._optimizer.zero_grad()

        # calc loss for each data point
        losses = []
        for d in train_data:
            state, rollout_prob, reward = d
            action_prob, _ = self.net(state)
            squ_e = (y-currq)**2
            cross_e = self.loss(input, target)
            loss = squ_e - cross_e
            loss.backward()

        # update parameters
        self._opt.step()

    def reset(self):
        pass

    def set_train(self):
        '''
        Use this method to set the policy in a mode for training.
        '''
        self._testing = False
        self._net.train()

    def set_eval(self):
        '''
        Use this method to set the policy in a mode for testing.
        '''
        self._testing = True
        self._net.eval()
