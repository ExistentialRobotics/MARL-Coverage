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

    def __lt__(self, other):
        return self.currstep < other.currstep

    def __eq__(self, other):
        return np.all(self.state==other.state) and self.currstep == other.currstep

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

    def simulate(self, episode, render=True, makevid=True):
        # print("----------In simulate----------")
        # iterate till episode completion
        for data in episode:
            # print(data)
            action = data[1]
            next_state, reward = self._env.step(action)

            # determine if episode is completed
            done = self._env.done()
            # print("action: " + str(action) + " done: " + str(done))

            # render if necessary
            if render:
                frame = self._env.render()
                if(makevid):
                    self._logger.addFrame(frame)

    def train(self):
        for i in range(self._iters):
            print("Training Iteration: " + str(i) + " out of " + str(self._iters))

            # get data with current neural net
            train_data = []
            for j in range(self._epi):
                print("Training Episode: " + str(j) + " out of " + str(self._epi))

                episode = self.rollout()
                train_data += episode

            # print(len(train_data))
            # print(train_data)

            # update neural net
            self.update_policy(train_data)


    def pi(self, state):
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

        return u


    def rollout(self, state):
        # obtain node from game tree if already constructed
        # print(str(state))
        if self._nodes[str(state) + str(0)] is None:
            # print("creating new start node")
            start_node = Node(state, 0)
            self._nodes[str(state) + str(0)] = start_node
        else:
            # print("start node already in tree")
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
            # print("----rollout loop----")
            # get node with lowest (cost + heuristic) off the heap
            if (len(self._frontier) == 0):
                print("empty frontier!")
                break
            curr_cost, node = heappop(self._frontier)
            # print("output of heappop: " + str(node.cost))
            # for jerome in self._frontier:
            #     print(jerome.cost)

            # print("frontier: " + str(len(self._frontier)) + " steps: " + str(node.currstep) + " num nodes in the tree: " + str(len(self._nodes)))

            # reset dicts
            self._fdict_nodes[node] = None
            self._explored[node] = True

            # print(str(len(self._explored)))

            # generate children if node hasn't been visited
            if len(node.children) == 0:
                for i in range(self._sim_env._num_actions):
                    state, reward = self._sim_env.step(copy.deepcopy(node.state), i)

                    # obtain node from game tree if already constructed
                    # print(str(state) + " " + str(self._nodes[str(state)]))
                    state_str = str(state) + str(node.currstep + 1)
                    if self._nodes[state_str] is None:
                        # print("creating new node")
                        n = Node(state, node.currstep + 1, i, node)
                        self._nodes[state_str] = n
                    else:
                        # print("node already in tree")
                        n = self._nodes[state_str]
                    node.children.append(n)

            # iterate thru children
            done = False
            for child in node.children:
                # print("child state: " + str(child.state))
                # for e in self._explored:
                #     print("explored node state: " + str(e.state))

                # determine if a node is in the frontier
                if not self._explored[child] and not self._fdict_nodes[child]:
                    # print("Child node not in frontier!")
                    # use neural net to give heuristic
                    state_tensor = (torch.from_numpy(child.state).float()).to(self._device)
                    # print("input to neural net: " + str(state_tensor))
                    heuristic = self._net(state_tensor)

                    # if not done, add to heap
                    child_cost = child.currstep + heuristic.item()
                    if self._sim_env.isTerminal(child.state, child.currstep):
                        done = True
                        fin_steps = child.currstep
                        goal_node = child
                        break
                    else:
                        # print("Adding child to frontier: " + str(child))
                        heappush(self._frontier, (child_cost, child))
                        self._fdict_nodes[child] = (child_cost, child)
                elif self._fdict_nodes[child] is not None:
                    # print("-----------Child node in frontier!-----------")
                    # use neural net to give heuristic
                    state_tensor = (torch.from_numpy(child.state).float()).to(self._device)
                    heuristic = self._net(state_tensor)

                    # replace the state in the frontier if child has a lower cost
                    c_cost = child.currstep + heuristic.item()
                    f_cost, f_node = self._fdict_nodes[child]
                    # print("node currstep: " + str(node.currstep))
                    # print("child currstep: " + str(child.currstep) + ", child cost: " + str(c_cost))
                    # print("frontier currstep: " + str(f_node.currstep) + ", frontier cost: " + str(f_cost))
                    if f_cost > c_cost:
                        # print("Replacing node in the frontier!")
                        self._frontier.remove((f_cost, f_node))
                        heappush(self._frontier, (c_cost, child))
                        self._fdict_nodes[child] = (c_cost, child)
                else:
                    pass
                    # print("Child has been explored: " + str((child in self._explored)))
            # print("Done: " + str(done))

        # construct episode from optimal path
        episode = []

        # if goal hasn't been reached, set terminal node as last node explored
        if goal_node is None:
            goal_node = node
            total_steps = goal_node.currstep
        else:
            total_steps = goal_node.currstep

        c_node = goal_node
        p_node = goal_node.previous
        # print("goal state: " + str(goal_node[1].state))
        while p_node:
            episode.append((p_node.state, c_node.previous_a, c_node.state, p_node.currstep, total_steps))
            temp = p_node.previous
            c_node = p_node
            p_node = temp
        episode.reverse()

        # print("length of path: " + str(len(episode)))
        # for i in range(len(episode)):
        #     print(str(i) + ": " + str(episode[i]))

        return episode

    def update_policy(self, train_data):
        # I wrote this with the assumption that each datum is a tuple of the form:
        # (state, action probabilities, reward). I'll try to batch this as we move
        # along with the implementation

        # zero gradients
        self._opt.zero_grad()

        # calc loss for each data point
        total_steps = len(train_data)
        avg_loss = 0
        for i in range(total_steps):
            state, action, reward, next_state, done = train_data[i]
            state_tensor = (torch.from_numpy(state).float()).to(self._device)
            heuristic = self._net(state_tensor)
            loss = (total_steps - (heuristic + i))**2
            loss.backward()
            avg_loss += loss
        print("Avg loss: " + str(avg_loss / total_steps))
        self._avgloss = (avg_loss / total_steps)

        # update parameters
        self._opt.step()

        #decaying epsilon
        self.decayEpsilon()

    def reset(self):
        pass

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
