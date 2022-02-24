from . base_policy import Base_Policy
import numpy as np
import torch
from heapq import *
import copy
import cv2
import sys
from collections import defaultdict
from queue import PriorityQueue

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
        self._max_len = (env._obs_dim[1] ** 2) * 2

        # e greedy stuff
        self._epsilon = 0
        self._e_decay = policy_config['epsilon_decay']
        self._min_epsilon = policy_config['min_epsilon']
        self._testing = False
        self._testing_epsilon = policy_config['testing_epsilon']

        # number of nodes to explore every time an action is attempted
        self._num_explore = policy_config["num_explore"]

        # which heurisitic to use
        self._learned = policy_config["learned"]

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

    def calc_heuristic(self, state):
        h = None
        if self._learned:
            state_tensor = (torch.from_numpy(state).float()).to(self._device)
            h = self._net(state_tensor)
        else:
            h = np.count_nonzero(state[2])
        return h

    def pi(self, state, phase_1=False):
        u = self.frontier_based(state)

        # available actions
        a = [0, 1, 2, 3]

        # epsilon greedy check
        s = np.random.uniform()

        # epsilon greedy policy
        if(s > self._epsilon or (self._testing and s > self._testing_epsilon)):
            if phase_1:
                # run frontier based with e probability for the first phase of training
                u = self.frontier_based(state)
            else:
                # run MDP A* with e probability for the first phase of training
                episode = self.rollout(state)
                if len(episode) == 0:
                    u = np.random.randint(self.num_actions)
                    print("0 episode length")
                else:
                    u = episode[0][1]
        else:
            if phase_1:
                #random
                u = np.random.randint(self.num_actions)
            else:
                u = self.frontier_based(state)
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
        start_node = Node(state, 0)

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
                    # get heuristic
                    heuristic = self.calc_heuristic(child.state)
                    if self._learned:
                        heuristic = heuristic.item()

                    # set parent
                    child.previous = node
                    child.previous_a = i

                    # if not done, add to heap
                    child_cost = child.currstep + heuristic
                    if self._sim_env.isTerminal(child.state, child.currstep):
                        done = True
                        fin_steps = child.currstep
                        goal_node = child
                        break
                    else:
                        heappush(self._frontier, (child_cost, child))
                        self._fdict_nodes[child] = (child_cost, child)
                elif self._fdict_nodes[child] is not None:
                    # get heuristic
                    heuristic = self.calc_heuristic(child.state)
                    if self._learned:
                        heuristic = heuristic.item()

                    # replace the state in the frontier if child has a lower cost
                    f_cost, f_node = self._fdict_nodes[child]
                    if f_node.currstep > child.currstep:
                        print("removing child from frontier")
                        c_cost = child.currstep + heuristic
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

    def frontier_based(self, state):
        pos_img, observed_obs, free, grid = state
        free_copy = copy.deepcopy(free)

        # get robot position
        pos = np.nonzero(pos_img)

        # flip 0s and 1s
        free_copy = np.bitwise_not(free_copy.astype('?')).astype(np.uint8)
        free_copy = free_copy.astype('int')

        # get obstacle positions
        obs_inds = np.argwhere(grid < 0)

        # mark obstacle positions as not free
        free_copy[obs_inds[:, 0], obs_inds[:, 1]] = -1

        # get dijkstra cost
        cost_map = self.dijkstra_path_map(free_copy, pos[0], pos[1])

        # determine action
        u = -1
        m = self._max_len
        for i in range(self.num_actions):
            if i == 0:
                p = (pos[0] - 1, pos[1])
            elif i == 1:
                p = (pos[0] + 1, pos[1])
            elif i == 2:
                p = (pos[0], pos[1] + 1)
            elif i == 3:
                p = (pos[0], pos[1] - 1)
            if self._env.isInBounds(p[0], p[1]) and cost_map[p] == 1:
                u = i
                break

        if u == -1:
            u = 0

        return u

    def update_policy(self, train_data):
        if self._learned:
            # zero gradients
            self._opt.zero_grad()

            # calc loss for each data point
            total_steps = len(train_data)
            print("total_steps: " + str(total_steps))
            avg_loss = 0
            for i in range(total_steps):
                state, action, reward, next_state, done = train_data[i]
                heuristic = self.calc_heuristic(state)
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

    def reset(self, grid, testing):
        self._testing = testing
        self._prev_states = []
        self._nodes = defaultdict(lambda: None)
        self._sim_env._grid = grid

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

    def in_bounds(self, x, y, grid):
        return x >= 0 and y >=0 and x < grid.shape[0] and y < grid.shape[1]

    def get_valid_neighbors(self, x, y, grid, visited):
        """
        Args:
           x : the x position on the grid
           y : the y position
           grid : an array representing the environment, 0 is explored,
                 1 is unexplored, and -1 is obstacle
           visited: an array representing whether we have visited a cell
                    or not, 1 is visited, 0 is not visited

        Returns:
           A list of valid neighbors and their coordinates
        """
        neighbors = []

        if self.in_bounds(x+1, y, grid) and visited[x+1][y] == 0 and grid[x+1][y] != -1:
            neighbors.append((x+1,y))

        if self.in_bounds(x-1, y, grid) and visited[x-1][y] == 0 and grid[x-1][y] != -1:
            neighbors.append((x-1,y))

        if self.in_bounds(x, y+1, grid) and visited[x][y+1] == 0 and grid[x][y+1] != -1:
            neighbors.append((x,y+1))

        if self.in_bounds(x, y-1, grid) and visited[x][y-1] == 0 and grid[x][y-1] != -1:
            neighbors.append((x,y-1))

        return neighbors


    def dijkstra_cost_map(self, grid):
        """
        Args:
           grid : an array representing the environment, 1 is explored,
                 0 is unexplored, and -1 is obstacle

        Returns:
           a cost array representing the cost from the closest unexplored node

        Notes:
        we will assume that the goals points are unexplored cells
        """
        open_set = PriorityQueue()
        visited = np.zeros(grid.shape)
        cost = -1*np.ones(grid.shape)

        #adding all unexplored cells to the open set
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):

                #checking if cell is unexplored
                if grid[i][j] == 0:
                    #adding unexplored cell to open set
                    open_set.put((0, (i,j)))

        #main dijkstra loop
        while not open_set.empty():
            cell = open_set.get()

            #check if cell has already been visited
            if visited[cell[1][0]][cell[1][1]] == 1:
                continue

            #mark as visited, finalize cost
            visited[cell[1][0]][cell[1][1]] = 1
            cost[cell[1][0]][cell[1][1]] = cell[0]

            #looping over all neighbors and updating their costs
            neighbors = get_valid_neighbors(cell[1][0], cell[1][1], grid, visited)

            for neighbor in neighbors:
                open_set.put((cell[0] + 1, neighbor))

        return cost

    def dijkstra_path_map(self, grid, start_x, start_y):
        """
        Args:
           grid : an array representing the environment, 1 is explored,
                 0 is unexplored, and -1 is obstacle
           start_x : starting x position
           start_y : starting y position

        Returns:
           an array showing the shortest path to an unexplored cell

        """
        open_set = PriorityQueue()
        visited = np.zeros(grid.shape)
        cost = -1*np.ones(grid.shape)

        #adding starting point to the open set
        open_set.put((0, (start_x[0], start_y[0])))

        end_point = None

        #main dijkstra loop
        # print(grid)
        while not open_set.empty():
            cell = open_set.get()

            #check if cell has already been visited
            # print("--------------")
            # print(str(cell[1][0]) + " " + str(cell[1][1]))
            # print(visited)
            if visited[cell[1][0]][cell[1][1]] == 1:
                continue

            #mark as visited, finalize cost
            visited[cell[1][0]][cell[1][1]] = 1
            cost[cell[1][0]][cell[1][1]] = cell[0]

            #checking if cell is unexplored
            if grid[cell[1][0]][cell[1][1]] == 0:
                end_point = cell[1]
                break

            #looping over all neighbors and updating their costs
            neighbors = self.get_valid_neighbors(cell[1][0], cell[1][1], grid, visited)

            for neighbor in neighbors:
                open_set.put((cell[0] + 1, neighbor))

        #using cost array to make optimal path
        path_array = np.zeros(grid.shape)

        #handling case where we can't reach any unexplored points
        # print(end_point)
        if end_point == None:
            return path_array

        curr = end_point
        curr_cost = cost[curr[0], curr[1]]
        path_array[curr[0], curr[1]] = 1

        #reset visited to not interfere with neighbor check
        visited = 1 - visited

        while curr[0] != start_x or curr[1] != start_y:
            neighbors = self.get_valid_neighbors(curr[0], curr[1], grid, visited)

            #finding the neighbor with the minimum cost
            for neighbor in neighbors:
                if cost[neighbor[0], neighbor[1]] == curr_cost - 1:
                    curr_cost -= 1
                    curr = neighbor
                    break

            #adding the current cell to the path
            path_array[curr[0], curr[1]] = 1

        assert path_array[start_x, start_y] == 1

        return path_array
