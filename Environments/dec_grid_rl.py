import numpy as np
import matplotlib.pyplot as plt
from . environment import Environment
from queue import PriorityQueue
import pygame
import cv2

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
        self._egoradius = env_config['egoradius']
        self._free_penalty = env_config['free_penalty']
        self._done_thresh = env_config['done_thresh']
        self._done_incr = env_config['done_incr']
        self._terminal_reward = env_config['terminal_reward']

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

        #performing observation
        reward += self.observe()

        #getting observations
        observations = self.get_egocentric_observations()

        #incrementing step count
        self._currstep += 1

        if min(self._done_thresh, 1) <= self.percent_covered():
            reward += self._terminal_reward

        return observations[0], reward

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

            z = np.zeros((3, 2*self._egoradius + 1, 2*self._egoradius + 1))

            #looping over all grid cells to sense
            for j in range(x - self._egoradius, x + self._egoradius + 1):
                for k in range(y - self._egoradius, y + self._egoradius + 1):
                    #ego centric coordinates
                    a = j - (x - self._egoradius)
                    b = k - (y - self._egoradius)

                    #checking if in bounds
                    if(self.isInBounds(j,k)):
                       z[0][a][b] = self._robot_pos_map[j][k]
                       z[1][a][b] = self._free[j][k]
                       z[2][a][b] = self._observed_obstacles[j][k]

                    #cannot give any information about where grid ends (that would be cheating)
                    else:
                        z[1][a][b] = 1

            #adding observation
            zlis.append(z)
        return zlis

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
