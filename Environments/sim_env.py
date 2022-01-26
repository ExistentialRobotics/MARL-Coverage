import numpy as np
import matplotlib.pyplot as plt
from . environment import Environment
from queue import PriorityQueue
import cv2
import pygame

class SuperGrid_Sim(object):
    """
    A Centralized Multi-Agent Grid Environment with a discrete action
    space. The objective of the environment is to cover as much of the region
    as possible.
    """
    def __init__(self, grid, obs_dim, env_config):
        super().__init__()
        #list of grids to use in training
        self._grid = grid

        #environment config parameters
        self._numrobot = env_config['numrobot']
        self._maxsteps = env_config['maxsteps']
        self._collision_penalty = env_config['collision_penalty']
        self._senseradius = env_config['senseradius']
        self._free_penalty = env_config['free_penalty']
        self._done_thresh = env_config['done_thresh']
        self._done_incr = env_config['done_incr']
        self._terminal_reward = env_config['terminal_reward']
        self._dist_r = env_config['dist_reward']
        self._use_scanning = env_config['use_scanning']

        #observation and action dimensions
        self._obs_dim = obs_dim
        self._num_actions = 4**self._numrobot

        #experimental pygame shite
        pygame.init()
        self._display = pygame.display.set_mode((1075, 1075))

    def step(self, state, action):
        imgs, currstep = state

        # decompose state
        pos_img = imgs[0]
        observed_obstacles = imgs[1]
        free = imgs[2]
        distance_map = imgs[3]

        # extract x,y position
        pos = np.nonzero(pos_img)

        # dims of self._grid
        width = self._grid.shape[0]
        height = self._grid.shape[1]

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

        # calc distance from observed to free cells
        if self._dist_r:
            distance_map = self.get_distance_map(free)

        # apply controls
        u = action

        #left
        if(u == 0):
            x = pos[0] - 1
            y = pos[1]

            if(self.isInBounds(x,y,width,height) and not self.isOccupied(x,y)):
                pos = (x, y)
                if self._dist_r:
                    reward[i] += distance_map[x, y]
            else:
                reward[i] -= self._collision_penalty
        #right
        elif(u == 1):
            x = pos[0] + 1
            y = pos[1]

            if(self.isInBounds(x,y,width,height) and not self.isOccupied(x,y)):
                pos = (x, y)
                if self._dist_r:
                    reward[i] += distance_map[x, y]
            else:
                reward[i] -= self._collision_penalty
        #up
        elif(u == 2):
            x = pos[0]
            y = pos[1] + 1

            if(self.isInBounds(x,y,width,height) and not self.isOccupied(x,y)):
                pos = (x, y)
                if self._dist_r:
                    reward[i] += distance_map[x, y]
            else:
                reward[i] -= self._collision_penalty
        #down
        elif(u == 3):
            x = pos[0]
            y = pos[1] - 1

            if(self.isInBounds(x,y,width,height) and not self.isOccupied(x,y)):
                pos = (x, y)
                if self._dist_r:
                    reward[i] += distance_map[x, y]
            else:
                reward[i] -= self._collision_penalty

        #sense from the current robot position
        x = np.asscalar(pos[0])
        y = np.asscalar(pos[1])

        #looping over all self._grid cells to sense
        for j in range(x - self._senseradius, x + self._senseradius + 1):
            for k in range(y - self._senseradius, y + self._senseradius + 1):
                #checking if cell is not visited, in bounds, not an obstacle
                if(self.isInBounds(j,k,width,height) and self._grid[j][k]>=0 and
                    free[j][k] == 1):
                    # add reward
                    reward += self._grid[j][k]

                    # mark as not free
                    free[j][k] = 0

                elif(self.isInBounds(j,k,width,height) and self._grid[j][k]>=0 and
                    free[j][k] == 0):
                    reward -= self._free_penalty

                elif(self.isInBounds(j,k,width,height) and self._grid[j][k]<0 and
                        observed_obstacles[j][k] == 0):
                        # track observed obstacles
                        observed_obstacles[j][k] = 1

        #incrementing step count
        currstep += 1

        # create position image
        pos_img = np.zeros((pos_img.shape[0], pos_img.shape[1]))
        pos_img[pos] = 1

        # create state
        imgs = np.stack(np.array([self._grid, pos_img, observed_obstacles, free]), axis=0)
        state = (imgs, currstep)

        #check env is covered
        if min(self._done_thresh, 1) <= self.percent_covered(state):
            reward += self._terminal_reward

        return state, reward

    def isInBounds(self, x, y, width, length):
        return x >= 0 and x < width and y >= 0 and y < length

    def isOccupied(self, x, y):
        #checking if no obstacle in that spot
        if(self._grid[x, y] < 0):
            return True

        return False

    def get_distance_map(self, free):
        inv = np.bitwise_not(free.astype('?')).astype(np.uint8)
        distance_map = cv2.distanceTransform(inv, cv2.DIST_L1,
                                             cv2.DIST_MASK_PRECISE)

        # scale map values to be between 0 and 1
        if np.max(distance_map) > 0:
            distance_map = distance_map / np.max(distance_map)

        # invert the values
        return 1 - distance_map

    def isTerminal(self, state):
        imgs, steps = state
        if min(self._done_thresh, 1) <= self.percent_covered(state):
            self._done_thresh += self._done_incr
            return True
        if steps == self._maxsteps:
            return True
        return False

    def percent_covered(self, state):
        imgs, steps = state
        free = imgs[2]
        return np.count_nonzero(free < 1) / np.count_nonzero(self._grid > 0)
