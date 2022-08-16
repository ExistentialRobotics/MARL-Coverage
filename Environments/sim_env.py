"""
sim_env.py contains code for the SuperGrid_Sim simulation environment.

Author: Peter Stratton
Email: pstratto@ucsd.edu, pstratt@umich.edu, peterstratton121@gmail.com
Author: Shreyas Arora
Email: sharora@ucsd.edu
"""
import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
import cv2
import pygame
import copy

class SuperGrid_Sim(object):
    """
    A Centralized Multi-Agent Grid Environment with a discrete action
    space. The objective of the environment is to cover as much of the region
    as possible. This environment is used if a policy needs to simulate a few
    timesteps ahead. It contains no notion of state. SuperGrid_Sim should only
    be used in conjunction with environment SuperGridRL.
    """
    def __init__(self, obs_dim, env_config):
        """
        Constructor for class  SuperGrid_Sim inits assorted parameters

        Parameters:
            obs_dim    - dimension of observations
            env_config - config file containing assorted parameters of the
                         environment
        """
        super().__init__()
        #environment config parameters
        self._numrobot = env_config['numrobot']
        self._train_maxsteps = env_config['train_maxsteps']
        self._test_maxsteps = env_config['test_maxsteps']
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

        self.prev_a = None

    def step(self, state, action, grid):
        """
        Processes the an action according to the environment dynamics

        Parameters:
            state  - numpy array describing the state of the environment to
                     simulate
            action - int or array describing the action to execute
            grid   - numpy array representing the unaltered original environment

        Return:
            state   - stack of numpy arrays descirbing the state of the new
                      environment
            reward  - scalar reward signal calculated based on the action
        """
        # decompose state
        pos_img, observed_obstacles, free, distance_map = state[0]
        # currstep = state[1]

        # print("state inside sim_env step: " + str(state))

        # extract x,y position
        pos = np.nonzero(pos_img)
        # pos = copy.deepcopy(pos)

        # dims of self._grid
        width = grid.shape[0]
        height = grid.shape[1]

        #initialize reward for this step
        reward = 0

        # apply controls
        u = action

        # print("pos before alteration: " + str(pos) + " action: " + str(u))

        #left
        if(u == 0):
            x = pos[0] - 1
            y = pos[1]

            if(self.isInBounds(x, y, width, height) and not self.isOccupied(x,
                                                                      y, grid)):
                pos = (x, y)
                # print("new pos: " + str(pos))
                if self._dist_r:
                    reward += distance_map[x, y]
            else:
                reward -= self._collision_penalty
        #right
        elif(u == 1):
            x = pos[0] + 1
            y = pos[1]

            if(self.isInBounds(x, y, width, height) and not self.isOccupied(x,
                                                                      y, grid)):
                pos = (x, y)
                # print("new pos: " + str(pos))
                if self._dist_r:
                    reward += distance_map[x, y]
            else:
                reward -= self._collision_penalty
        #up
        elif(u == 2):
            x = pos[0]
            y = pos[1] + 1

            if(self.isInBounds(x, y, width, height) and not self.isOccupied(x,
                                                                      y, grid)):
                pos = (x, y)
                # print("new pos: " + str(pos))
                if self._dist_r:
                    reward += distance_map[x, y]
            else:
                reward -= self._collision_penalty
        #down
        elif(u == 3):
            x = pos[0]
            y = pos[1] - 1

            if(self.isInBounds(x, y, width, height) and not self.isOccupied(x,
                                                                      y, grid)):
                pos = (x, y)
                # print("new pos: " + str(pos))
                if self._dist_r:
                    reward += distance_map[x, y]
            else:
                reward -= self._collision_penalty

        #sense from the current robot position
        x_p = np.asscalar(pos[0])
        y_p = np.asscalar(pos[1])

        #looping over all self._grid cells to sense
        for j in range(x_p - self._senseradius, x_p + self._senseradius + 1):
            for k in range(y_p - self._senseradius,
                           y_p + self._senseradius + 1):
                #checking if cell is not visited, in bounds, not an obstacle
                if(self.isInBounds(j,k,width,height) and grid[j][k]>=0 and
                    free[j][k] == 1):
                    # add reward
                    reward += grid[j][k]

                    # mark as not free
                    free[j][k] = 0

                elif(self.isInBounds(j,k,width,height) and grid[j][k]>=0 and
                    free[j][k] == 0):
                    reward -= self._free_penalty

                elif(self.isInBounds(j,k,width,height) and grid[j][k]<0 and
                        observed_obstacles[j][k] == 0):
                        # track observed obstacles
                        observed_obstacles[j][k] = 1
        # increment step count
        # currstep += 1

        # create position image
        pos_img = np.zeros((pos_img.shape[0], pos_img.shape[1]))
        pos_img[pos] = 1

        # create state
        state = np.stack(np.array([pos_img, observed_obstacles, free,
                         distance_map]), axis=0)
        # state = (state, currstep)

        # print("state after sim_env step: " + str(state))

        #check env is covered
        if min(self._done_thresh, 1) <= self.percent_covered(state):
            reward += self._terminal_reward

        return state, reward


    def isInBounds(self, x, y, width, length):
        """
        Checks whether the position is inside the map

        Parameters:
            x - x position
            y - y position

        Return:
            - boolean describing whether or not the position is in bounds
        """
        return x >= 0 and x < width and y >= 0 and y < length

    def isOccupied(self, x, y, grid):
        """
        Checks whether the cell at the specificed position contains another
        robot or obstacle

        Parameters:
            x - x position
            y - y position

        Return:
            - boolean describing whether or not the position is occupied
        """
        #checking if no obstacle in that spot
        if(grid[x, y] < 0):
            return True

        return False

    def isTerminal(self, state):
        """
        Checks if robot reached a terminal state that ends the trajectory

        Parameters:
            state - stack of numpy arrays which represents the environment state

        Return:
            - boolean representing if the robot has reached the terminal state
        """
        if min(self._done_thresh, 1) <= self.percent_covered(state):
            self._done_thresh += self._done_incr
            return True
        return False

    def percent_covered(self, state, grid):
        """
        Parameters:
            state - numpy array describing the state of the environment
            grid  - unaltered grid map of the environment

        Return:
            - percent of free cells that have been sensed by the robot
        """
        pos_img, observed_obstacles, free, dist = state
        return np.count_nonzero(free < 1) / np.count_nonzero(grid > 0)
