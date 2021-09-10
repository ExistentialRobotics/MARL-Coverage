import numpy as np
import matplotlib.pyplot as plt
from . environment import Environment
from queue import PriorityQueue
import pygame
import cv2
import time

class DecGridRL(object):
    """
    A Decentralized Multi-Agent Grid Environment with a discrete action
    space. The objective of the environment is to cover as much of the region
    as possible.
    """
    def __init__(self, gridlis, env_config):
        super().__init__()
        #list of grids to use in training
        self._gridlis = gridlis

        #environment config parameters
        self._numrobot = env_config['numrobot']
        self._maxsteps = env_config['maxsteps']
        self._collision_penalty = env_config['collision_penalty']
        self._senseradius = env_config['senseradius']
        self._free_penalty = env_config['free_penalty']
        self._done_thresh = env_config['done_thresh']
        self._done_incr = env_config['done_incr']
        self._terminal_reward = env_config['terminal_reward']

        #if ego_radius is 0, non-egocentric observation will be used
        self._egoradius = env_config['egoradius']
        self._mini_map_rad = env_config['mini_map_rad']
        self._comm_radius = env_config['comm_radius']
        self._allow_comm = env_config['allow_comm']

        #padding for map arrays
        self._pad = max(self._egoradius, self._mini_map_rad)

        #pick random map and generate robot positions
        self.reset()

        #observation and action dimensions for each agent
        self._obs_dim = self.get_egocentric_observations()[0].shape
        self._num_actions = 4

        #experimental pygame shite
        pygame.init()
        self._display = pygame.display.set_mode((1075, 1075))


    def step(self, action):
        #handling case where action is an integer that identifies the action
        if type(action) != np.ndarray:
            ulis = np.zeros((self._numrobot,))
            #conveting integer to base 4 and putting it in ulis
            for i in range(self._numrobot):
                ulis[i] = action % 4
                action = action // 4
        else:
            ulis = action

        #initialize reward for this step
        reward = 0

        # apply controls to each robot
        for i in range(self._numrobot):
            u = ulis[i]
            #right
            if(u == 0):
                reward += self.updateRobotPos(self._xinds[i] + 1, self._yinds[i], i)
            #up
            elif(u == 1):
                reward += self.updateRobotPos(self._xinds[i], self._yinds[i] + 1, i)
            #left
            elif(u == 2):
                reward += self.updateRobotPos(self._xinds[i] - 1, self._yinds[i], i)
            #down
            elif(u == 3):
                reward += self.updateRobotPos(self._xinds[i], self._yinds[i] - 1, i)

        # update communication graph
        self.updateCommmunicationGraph()

        #performing observation
        reward += self.observe()

        #getting observations
        if self._egoradius > 0:
            observations = self.get_egocentric_observations()
        else:
            observations = self.get_full_observations()

        #incrementing step count
        self._currstep += 1

        if min(self._done_thresh, 1) <= self.percent_covered():
            reward += self._terminal_reward

        if self._allow_comm:
            return [observations, self._adjacency_matrix], reward
        else:
            return observations, reward

    def updateRobotPos(self, x, y, i):
        """Updates the position of robot with robotindex to new coordinates
        if it doesn't result in a collision.

        Args:
           x : the new x position
           y : the new y position
           i : the index of the robot to update

        Returns: negative reward if collision would have occurred, zero
        otherwise
        """
        if(self.isInBounds(x,y) and not self.isOccupied(x,y)):
            #removing old robot positions from map
            self._robot_pos_map[self._xinds[i]][self._yinds[i]] = 0
            self._robot_pad[self._xinds[i]+self._pad][self._yinds[i]+self._pad]=0

            #updating tracked robot position
            self._xinds[i] = x
            self._yinds[i] = y

            #updating map with new position
            self._robot_pos_map[x][y] = 1
            self._robot_pad[x+self._pad][y+self._pad] = 1

            return 0
        else:
            return -self._collision_penalty

    def observe(self):
        '''
        Computes the observation from each of the robot positions and updates the shared map,
        which includes the observed cell and free cell layers
        '''
        obs_reward = 0
        #sense from all the current robot positions
        for i in range(self._numrobot):
            x = self._xinds[i]
            y = self._yinds[i]

            #looping over all grid cells to sense
            for j in range(x - self._senseradius, x + self._senseradius + 1):
                for k in range(y - self._senseradius, y + self._senseradius + 1):
                    #checking if cell is in bounds, not an obstacle
                    if(self.isInBounds(j,k) and self._grid[j][k]>=0):
                        # mark as not free
                        self._free[i][j][k] = 1
                        self._free_pad[i][j+self._pad][k+self._pad] = 1

                        #checking if visited
                        if not self._visited[j][k]:
                            # add reward
                            obs_reward += 1
                            self._numobserved += 1

                            #marking as visited
                            self._visited[j][k] = 1
                        else:
                            obs_reward -= self._free_penalty

                    elif(self.isInBounds(j,k) and self._grid[j][k]<0 and
                            self._observed_obstacles[i][j][k] == 0):
                            # track observed obstacles
                            self._observed_obstacles[i][j][k] = 1
                            self._obst_pad[i][j+self._pad][k+self._pad]=1
        return obs_reward

    def isInBounds(self, x, y):
        return x >= 0 and x < self._gridwidth and y >= 0 and y < self._gridlen

    def isOccupied(self, x, y):
        #checking if no obstacle in that spot and that no robots are there
        return self._grid[x][y] < 0 or self._robot_pos_map[x][y] == 1

    def get_egocentric_observations(self):
        '''
        Returns a list of observations, one for each agent, all from an egocentric
        perspective (centered at the agents' current positions). Draws data from the
        shared map.
        '''
        #creating the observation array
        if self._mini_map_rad > 0:
            numlayers = 5
        else:
            numlayers = 3

        z = np.zeros((self._numrobot, numlayers, 2*self._egoradius + 1, 2*self._egoradius + 1))

        #construct state for each robot
        for i in range(self._numrobot):
            x = self._xinds[i]
            y = self._yinds[i]

            #egocentric observation layers
            z[i][0] = self.arraySubset(self._robot_pad, x, y, self._egoradius)
            z[i][1] = self.arraySubset(self._free_pad[i], x, y, self._egoradius)
            z[i][2] = self.arraySubset(self._obst_pad[i], x, y,
                                    self._egoradius)
            #TODO I think this has been resolved, we don't need the copy
            # z[i][0][0][0] = 100
            # assert self._robot_pad[0][0] != 100

            # self._robot_pad[0][0] = 10
            # assert z[i][0][0][0] != 10

            #larger map view
            if self._mini_map_rad > 0:
                mini_free = self.arraySubset(self._free_pad[i], x, y, self._mini_map_rad)
                mini_obs = self.arraySubset(self._obst_pad[i], x, y,
                                            self._mini_map_rad)
                z[i][3] = cv2.resize(mini_free, dsize=(2*self._egoradius + 1, 2*self._egoradius + 1), interpolation=cv2.INTER_LINEAR)
                z[i][4] = cv2.resize(mini_obs, dsize=(2*self._egoradius + 1, 2*self._egoradius + 1), interpolation=cv2.INTER_LINEAR)

        return z

    def get_full_observations(self):
        """
        Return each agent's observations of the full map, showing
        the free cells and observed cells they have visited, and the
        robots currently visible to them in their communication radius.
        """
        #TODO
        pass


    def updateCommmunicationGraph(self):
        """
        Updates the communication graph based on the current
        robot positions and the communication radius.
        """
        #creating communication graph adjacency matrix
        self._adjacency_matrix = np.zeros((self._numrobot, self._numrobot))

        xinds = self._xinds
        yinds = self._yinds

        for i in range(xinds.shape[0]):
            for j in range(i+1, xinds.shape[0]):
                #chebyshev distance
                dist = max(abs(xinds[i] - xinds[j]), abs(yinds[i] - yinds[j]))
                if dist <= self._comm_radius:
                    self._adjacency_matrix[i][j] = 1
                    self._adjacency_matrix[j][i] = 1

    def arraySubset(self, array, x, y, radius):
        '''
        Returns a subset of the given array centered at (x,y) with the given
        radius (chebyshev metric). The array is assumed to be prepadded with
        '''
        width = array.shape[0]
        length = array.shape[1]

        lb = x - radius + self._pad
        rb = x + radius + 1 + self._pad

        #up and down boundaries
        ub = y + radius + 1 + self._pad
        db = y - radius + self._pad

        #raw observation
        obs = array[lb:rb, db:ub]

        #TODO do we need to copy the array, it adds a fair amount of overhead
        # return np.copy(obs)
        return obs

    def reset(self):
        #picking a map at random
        self._grid = self._gridlis[np.random.randint(len(self._gridlis))]

        #padding the grid with obstacles at the edges so the agent can sense walls
        self._grid = np.pad(self._grid, (1,), 'constant', constant_values=(-1,))

        #dimensions
        self._gridwidth = self._grid.shape[0]
        self._gridlen = self._grid.shape[1]

        #resetting step count
        self._currstep = 0

        #generating random robot positions
        self._xinds = np.zeros(self._numrobot, dtype=int)
        self._yinds = np.zeros(self._numrobot, dtype=int)

        #map of robot positions
        self._robot_pos_map = np.zeros((self._gridwidth, self._gridlen))
        self._robot_pad = np.zeros((self._gridwidth + 2*self._pad, self._gridlen + 2*self._pad))

        #repeatedly trying to insert robots
        count = 0
        while(count != self._numrobot):
            #generating random coordinate
            x = np.random.randint(self._gridwidth)
            y = np.random.randint(self._gridlen)

            #testing if coordinate is not an obstacle or other robot
            if(self._grid[x][y] >= 0 and self._robot_pos_map[x][y] == 0):
                self._robot_pos_map[x][y] = 1
                self._xinds[count] = x
                self._yinds[count] = y
                count += 1

        # history of observed obstacles
        self._observed_obstacles = np.zeros((self._numrobot, self._gridwidth, self._gridlen))
        self._obst_pad = np.zeros((self._numrobot, self._gridwidth + 2*self._pad, self._gridlen + 2*self._pad))

        # history of free cells, one layer per robot
        self._free = np.zeros((self._numrobot, self._gridwidth, self._gridlen))
        self._free_pad = np.zeros((self._numrobot, self._gridwidth + 2*self._pad, self._gridlen + 2*self._pad))

        #visited array to track all visitations
        self._visited = np.zeros((self._gridwidth, self._gridlen))

        #finding number of free cells
        self._numfree = np.count_nonzero(self._grid > 0)
        self._numobserved = 0

        #update communication graph
        self.updateCommmunicationGraph()

        #return first observation
        observations = self.get_egocentric_observations()

        #return observations
        if self._allow_comm:
            return [observations, self._adjacency_matrix]
        else:
            return observations

    def done(self):
        if min(self._done_thresh, 1) <= self.percent_covered():
            print("Full Environment Covered")
            self._done_thresh += self._done_incr
            return True
        if self._currstep == self._maxsteps:
            return True
        return False

    def percent_covered(self):
        return self._numobserved / self._numfree

    def render(self):
        #base image
        image = np.zeros((self._gridwidth, self._gridlen, 3))

        #adding observed obstacles to the base
        unionobst = np.clip(np.sum(self._observed_obstacles, axis=0), 0, 1)
        obslayer = np.stack([200*unionobst,
                             0*unionobst,
                             255*unionobst], -1)
        image += obslayer

        #adding observed free cells to the base
        freelayer = np.stack([0*self._visited, 225*self._visited, 255*self._visited], -1)
        image += freelayer

        #adding robot positions to the base
        freelayer = np.stack([255*self._robot_pos_map,
                              0*self._robot_pos_map,
                              0*self._robot_pos_map], -1)
        image += freelayer

        scaling = max(min(1024//self._gridwidth, 1024//self._gridlen), 1)
        image = cv2.resize(image, (0,0), fx=scaling, fy=scaling, interpolation=cv2.INTER_NEAREST)

        image = cv2.copyMakeBorder(image, 0, 1075 - image.shape[1], 0, 1075 - image.shape[0], cv2.BORDER_CONSTANT, value=[0,0,0])

        #graphing occupancy grid
        surf = pygame.surfarray.make_surface(image)

        #graphing adjacency matrix connections

        self._display.blit(surf, (0, 0))
        pygame.display.update()

        #returning the image that was used
        return image
