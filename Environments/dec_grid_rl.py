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

        #pick random map and generate robot positions
        self.reset()

        #observation and action dimensions
        #TODO fix this stuff for multiagent
        self._obs_dim = self.get_egocentric_observations()[0].shape
        self._num_actions = 4

        #experimental pygame shite
        pygame.init()
        self._display = pygame.display.set_mode((1075, 1075))


    def step(self, action):
        #handling case where action is an integer that identifies the action
        if type(action) != list:
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

            #left
            if(u == 0):
                x = self._xinds[i] - 1
                y = self._yinds[i]

                if(self.isInBounds(x,y) and not self.isOccupied(x,y)):
                    self._robot_pos_map[self._xinds[i]][self._yinds[i]] = 0
                    self._xinds[i] = x
                    self._robot_pos_map[x][y] = 1
                else:
                    reward -= self._collision_penalty
            #right
            elif(u == 1):
                x = self._xinds[i] + 1
                y = self._yinds[i]

                if(self.isInBounds(x,y) and not self.isOccupied(x,y)):
                    self._robot_pos_map[self._xinds[i]][self._yinds[i]] = 0
                    self._xinds[i] = x
                    self._robot_pos_map[x][y] = 1
                else:
                    reward -= self._collision_penalty
            #up
            elif(u == 2):
                x = self._xinds[i]
                y = self._yinds[i] + 1

                if(self.isInBounds(x,y) and not self.isOccupied(x,y)):
                    self._robot_pos_map[self._xinds[i]][self._yinds[i]] = 0
                    self._yinds[i] = y
                    self._robot_pos_map[x][y] = 1
                else:
                    reward -= self._collision_penalty
            #down
            elif(u == 3):
                x = self._xinds[i]
                y = self._yinds[i] - 1

                if(self.isInBounds(x,y) and not self.isOccupied(x,y)):
                    self._robot_pos_map[self._xinds[i]][self._yinds[i]] = 0
                    self._yinds[i]= y
                    self._robot_pos_map[x][y] = 1
                else:
                    reward -= self._collision_penalty

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

        return observations, self._adjacency_matrix, reward

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
                    #checking if cell is not visited, in bounds, not an obstacle
                    if(self.isInBounds(j,k) and self._grid[j][k]>=0 and
                        self._free[j][k] == 1):
                        # add reward
                        obs_reward += 1

                        self._numobserved += 1

                        # mark as not free
                        self._free[j][k] = 0

                    elif(self.isInBounds(j,k) and self._grid[j][k]>=0 and
                        self._free[j][k] == 0):
                        obs_reward -= self._free_penalty

                    elif(self.isInBounds(j,k) and self._grid[j][k]<0 and
                            self._observed_obstacles[j][k] == 0):
                            # track observed obstacles
                            self._observed_obstacles[j][k] = 1
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
        zlis = []
        #construct state for each robot
        for i in range(self._numrobot):
            x = self._xinds[i]
            y = self._yinds[i]

            if self._mini_map_rad > 0:
                numlayers = 5
            else:
                numlayers = 3

            z = np.zeros((numlayers, 2*self._egoradius + 1, 2*self._egoradius + 1))

            #egocentric observation layers
            z[0] = self.arraySubset(self._robot_pos_map, x, y, self._egoradius)
            z[1] = self.arraySubset(self._free, x, y, self._egoradius, pad=1)
            z[2] = self.arraySubset(self._observed_obstacles, x, y,
                                    self._egoradius)

            #larger map view
            if self._mini_map_rad > 0:
                mini_free = self.arraySubset(self._free, x, y, self._mini_map_rad, pad=1)
                mini_obs = self.arraySubset(self._observed_obstacles, x, y,
                                            self._mini_map_rad)
                z[3] = cv2.resize(mini_free, dsize=(2*self._egoradius + 1, 2*self._egoradius + 1), interpolation=cv2.INTER_LINEAR)
                z[4] = cv2.resize(mini_obs, dsize=(2*self._egoradius + 1, 2*self._egoradius + 1), interpolation=cv2.INTER_LINEAR)

            #adding observation
            zlis.append(z)
        return zlis

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

        for i in range(xinds.shape[0]):
            for j in range(i+1, xinds.shape[0]):
                #euclidean distance
                # dist = np.sqrt((xinds[i] - xinds[j])**2 + (yinds[i] - yinds[j])**2)
                #chebyshev distance
                dist = max(abs(xinds[i] - xinds[j]), abs(yinds[i] - yinds[j]))
                if dist <= self._commradius:
                    self._adjacency_matrix[i][j] = 1
                    self._adjacency_matrix[j][i] = 1

    def arraySubset(self, array, x, y, radius, pad=0):
        width = array.shape[0]
        length = array.shape[1]

        #right and left boundaries
        lb = max(0, x - radius)
        rb = min(width, x + radius + 1)

        #up and down boundaries
        ub = min(length, y + radius + 1)
        db = max(0, y - radius)

        #raw observation
        obs = array[lb:rb, db:ub]

        #adding padding in each direction to observation
        lpad = max(0, radius-x)
        rpad = max(0, x + radius + 1 - width)
        dpad = max(0, radius-y)
        upad = max(0, y + radius + 1 - length)
        obs = np.pad(obs, ((lpad, rpad), (dpad, upad)), 'constant', constant_values=(pad,))

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
        self._observed_obstacles = np.zeros((self._gridwidth, self._gridlen))

        # history of free cells
        self._free = np.ones((self._gridwidth, self._gridlen))

        #finding number of free cells
        self._numfree = np.count_nonzero(self._grid > 0)
        self._numobserved = 0

        # #performing observation
        # self.observe()

        #return observations
        return self.get_egocentric_observations()[0]

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
        obslayer = np.stack([200*self._observed_obstacles,
                             0*self._observed_obstacles,
                             255*self._observed_obstacles], -1)
        image += obslayer

        #adding observed free cells to the base
        inv_free = 1 - self._free
        freelayer = np.stack([0*inv_free, 225*inv_free, 255*inv_free], -1)
        image += freelayer

        #adding robot positions to the base
        freelayer = np.stack([255*self._robot_pos_map,
                              0*self._robot_pos_map,
                              0*self._robot_pos_map], -1)
        image += freelayer

        scaling = max(min(1024//self._gridwidth, 1024//self._gridlen), 1)
        image = cv2.resize(image, (0,0), fx=scaling, fy=scaling, interpolation=cv2.INTER_NEAREST)

        image = cv2.copyMakeBorder(image, 0, 1075 - image.shape[1], 0, 1075 - image.shape[0], cv2.BORDER_CONSTANT, value=[0,0,0])

        surf = pygame.surfarray.make_surface(image)
        self._display.blit(surf, (0, 0))
        pygame.display.update()

        #returning the image that was used
        return image
