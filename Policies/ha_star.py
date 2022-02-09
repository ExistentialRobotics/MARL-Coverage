from . base_policy import Base_Policy
import numpy as np
import torch
from heapq import *
import copy
from collections import defaultdict

class Node(object):
    def __init__(self, state, currstep, action=None, previous=None):
        self.state = state
        self.currstep = currstep
        self.tstep = None
        self.children = []
        self.previous = previous
        self.previous_a = action
        self._prev_states = []

    def __lt__(self, other):
        return self.currstep < other.currstep

    def __eq__(self, other):
        return np.all(self.state==other.state)

    def __hash__(self):
        return hash(str(self.state) + str(self.currstep))


class HA_Star(Base_Policy):

    def __init__(self, env, sim, net, num_actions, obs_dim, logger, policy_config, model_path=None):
        self._logger = logger

        self._env = env
        self._sim_env = sim # sim_environment to run on
        self.num_actions = env._num_actions
        self._net = net # neural network for prediction

        # e greedy stuff
        self._epsilon = 0
        self._e_decay = policy_config['epsilon_decay']
        self._min_epsilon = policy_config['min_epsilon']
        self._testing = False
        self._testing_epsilon = policy_config['testing_epsilon']

        # number of nodes to explore every time an action is attempted
        self._num_explore = policy_config["num_explore"]

        # cpu vs gpu code
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #moving net to gpu
        self._net.to(self._device)

        # loss and optimizer
        self._loss = torch.nn.CrossEntropyLoss()
        self._opt = torch.optim.Adam(self._net.parameters(),
                        lr=policy_config['lr'],
                        weight_decay=policy_config['weight_decay'])
        self._avgloss = None

        # performance metrics
        self._losses = []

        # dict mapping states to nodes in the tree
        self._nodes = defaultdict(lambda: None)

    def sim_step(self, state, action):
        next_state, reward = self._sim_env.step(state, action)
        if str(next_state) not in self._prev_states:
            return False
        return True

    def pi(self, state):
        # available actions
        a = [0, 1, 2, 3]

        #epsilon greedy check
        s = np.random.uniform()

        #epsilon greedy policy
        #if we are testing then we use a smaller testing epsilon
        if(s > self._epsilon or (self._testing and s > self._testing_epsilon)):
            episode = self.rollout(state)
            if len(episode) == 0:
                u = np.random.randint(self.num_actions)
                print("0 episode length")
            else:
                u = episode[0][1]
        else:
            #random
            u = np.random.randint(self.num_actions)
        reset = self.sim_step(state, u)

        # check actions until one that isn't from a previous state is found
        while reset and len(a) > 0:
            if u in a:
                a.remove(u)
            u = np.random.randint(self.num_actions)
            reset = self.sim_step(state, u)

        return u


    def rollout(self, state):
        # obtain node from game tree if already constructed
        if self._nodes[str(state) + str(0)] is None:
            start_node = Node(state, 0)
            self._nodes[str(state) + str(0)] = start_node
        else:
            start_node = self._nodes[str(state) + str(0)]

        # dicts tracking the explored and frontier nodes
        self._explored = defaultdict(lambda: False)
        self._fdict_nodes = defaultdict(lambda: None)
        self._frontier = []

        # initialize frontier with start state
        heappush(self._frontier, (0, start_node))
        self._fdict_nodes[start_node] = (0, start_node)

        # play out an episode
        done = False
        fin_cost = -1
        goal_node = None
        for _ in range(self._num_explore):
            # get node with lowest (cost + heuristic) off the heap
            if (len(self._frontier) == 0):
                print("empty frontier!")
                break
            curr_cost, node = heappop(self._frontier)

            # reset dicts
            self._fdict_nodes[node] = None
            self._explored[node] = True

            # generate children if node hasn't been visited
            if len(node.children) == 0:
                for i in range(self._sim_env._num_actions):
                    state, reward = self._sim_env.step(copy.deepcopy(node.state), i)

                    # obtain node from game tree if already constructed
                    state_str = str(state) + str(node.currstep + 1)
                    if self._nodes[state_str] is None:
                        n = Node(state, node.currstep + 1)
                        self._nodes[state_str] = n
                    else:
                        n = self._nodes[state_str]
                    node.children.append(n)

            # iterate thru children
            done = False
            for i in range(len(node.children)):
                child = node.children[i]

                # determine if a node is in the frontier
                if not self._explored[child] and not self._fdict_nodes[child]:
                    # use neural net to give heuristic
                    state_tensor = (torch.from_numpy(child.state).float()).to(self._device)
                    heuristic = self._net(state_tensor)

                    # set parent
                    child.previous = node
                    child.previous_a = i

                    # if not done, add to heap
                    child_cost = child.currstep + heuristic.item()
                    if self._sim_env.isTerminal(child.state, child.currstep):
                        done = True
                        fin_steps = child.currstep
                        goal_node = child
                        break
                    else:
                        heappush(self._frontier, (child_cost, child))
                        self._fdict_nodes[child] = (child_cost, child)
                elif self._fdict_nodes[child] is not None:
                    # use neural net to give heuristic
                    state_tensor = (torch.from_numpy(child.state).float()).to(self._device)
                    heuristic = self._net(state_tensor)

                    # replace the state in the frontier if child has a lower cost
                    c_cost = child.currstep + heuristic.item()
                    f_cost, f_node = self._fdict_nodes[child]
                    if f_cost > c_cost:
                        self._frontier.remove((f_cost, f_node))
                        heappush(self._frontier, (c_cost, child))
                        self._fdict_nodes[child] = (c_cost, child)

                        # set parent
                        child.previous = node
                        child.previous_a = i

        # construct episode from optimal path
        episode = []

        # if goal hasn't been reached, set terminal node as last node explored
        if goal_node is None:
            goal_node = node
            total_steps = goal_node.currstep
        else:
            total_steps = goal_node.currstep

        cont = False
        if start_node == goal_node:
            cont = True

        c_node = goal_node
        p_node = goal_node.previous
        while cont or c_node != start_node:
            cont = False
            episode.append((p_node.state, c_node.previous_a, c_node.state, p_node.currstep, total_steps))
            temp = p_node.previous
            c_node = p_node
            p_node = temp
        episode.reverse()

        return episode

    def update_policy(self, train_data):
        # zero gradients
        self._opt.zero_grad()

        # calc loss for each data point
        total_steps = len(train_data)
        print("total_steps: " + str(total_steps))
        avg_loss = 0
        for i in range(total_steps):
            state, action, reward, next_state, done = train_data[i]
            state_tensor = (torch.from_numpy(state).float()).to(self._device)
            heuristic = self._net(state_tensor)
            loss = (total_steps - (heuristic + i))**2
            loss.backward()
            avg_loss += loss
        self._avgloss = avg_loss / total_steps
        print("Avg loss: " + str(self._avgloss.item()))

        # update parameters
        self._opt.step()

        #decaying epsilon
        self.decayEpsilon()

    def add_state(self, state):
        self._prev_states.append(state)

    def reset(self):
        self._prev_states = []

    def decayEpsilon(self):
        #decaying the epsilon
        if self._epsilon > self._min_epsilon:
            self._epsilon *= self._e_decay

    def getnet(self):
        return self._net

    def printNumParams(self):
        pytorch_total_params = sum(p.numel() for p in self._net.parameters()
                                   if p.requires_grad)
        print(str(pytorch_total_params) + " in the heuristic network")

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
