import numpy as np
import matplotlib.pyplot as plt
from . environment import Environment
from queue import PriorityQueue
import cv2
import pygame

class SuperGridRL(object):
    """
    A Centralized Multi-Agent Grid Environment with a discrete action
    space. The objective of the environment is to cover as much of the region
    as possible.
    """
    def __init__(self, gridlis, env_config):
        super().__init__()
        #list of grids to use in training
        self._gridlis = gridlis

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
        self._prev_states = []

        #pick random map and generate robot positions
        self.reset()

        # # init graph data object
        # self._graph = Graph_Data(env_config['numfeatures'], self._xinds, self._yinds, env_config['commradius'])

        #observation and action dimensions
        state = self.get_state()
        self._obs_dim = state.shape
        self._num_actions = 4**self._numrobot

        #experimental pygame shite
        pygame.init()
        self._display = pygame.display.set_mode((1075, 1075))

    def step(self, action):
        # print("-----Inside step-----")

        #handling case where action is an integer that identifies the action
        if type(action) != list:
            ulis = np.zeros((self._numrobot,))
            #conveting integer to base 4 and putting it in ulis
            for i in range(self._numrobot):
                ulis[i] = action % 4
                action = action // 4
        else:
            ulis = action

        # print("robot position: " + str((self._xinds, self._yinds)))

        #initialize reward for this step
        reward = np.zeros((self._numrobot,))

        #making pq for sorting by minimum scan score
        pq = PriorityQueue()
        for i in range(self._numrobot):
            score = self._xinds[i] + self._yinds[i]*self._gridwidth
            pq.put((score, i))

        #robot 2 control, storing what robot got what control
        r2c = np.zeros((self._numrobot,), dtype=int)

        # calc distance from observed to free cells
        if self._dist_r:
            distance_map = self.get_distance_map()

        # apply controls to each robot
        for i in range(len(ulis)):
            u = ulis[i]

            #z is the robot index we are assigning controls to
            if self._use_scanning:
                z = pq.get()[1]
            else:
                z = i
            r2c[z] = i

            #left
            if(u == 0):
                x = self._xinds[z] - 1
                y = self._yinds[z]

                if(self.isInBounds(x,y) and not self.isOccupied(x,y)):
                    self._xinds[z] = x
                    if self._dist_r:
                        reward[i] += distance_map[x, y]
                else:
                    reward[i] -= self._collision_penalty
            #right
            elif(u == 1):
                x = self._xinds[z] + 1
                y = self._yinds[z]

                if(self.isInBounds(x,y) and not self.isOccupied(x,y)):
                    self._xinds[z] = x
                    if self._dist_r:
                        reward[i] += distance_map[x, y]
                else:
                    reward[i] -= self._collision_penalty
            #up
            elif(u == 2):
                x = self._xinds[z]
                y = self._yinds[z] + 1

                if(self.isInBounds(x,y) and not self.isOccupied(x,y)):
                    self._yinds[z] = y
                    if self._dist_r:
                        reward[i] += distance_map[x, y]
                else:
                    reward[i] -= self._collision_penalty
            #down
            elif(u == 3):
                x = self._xinds[z]
                y = self._yinds[z] - 1

                if(self.isInBounds(x,y) and not self.isOccupied(x,y)):
                    self._yinds[z]= y
                    if self._dist_r:
                        reward[i] += distance_map[x, y]
                else:
                    reward[i] -= self._collision_penalty

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
                        reward[r2c[i]] += self._grid[j][k]

                        # mark as not free
                        self._free[j][k] = 0

                    elif(self.isInBounds(j,k) and self._grid[j][k]>=0 and
                        self._free[j][k] == 0):
                        reward[r2c[i]] -= self._free_penalty

                    elif(self.isInBounds(j,k) and self._grid[j][k]<0 and
                            self._observed_obstacles[j][k] == 0):
                            # track observed obstacles
                            self._observed_obstacles[j][k] = 1

        # update graph
        # self._graph.set_data(self._xinds, self._yinds)

        #calculate current state
        state = self.get_state()
        # print(state)

        #incrementing step count
        self._currstep += 1

        reward = np.sum(reward)
        if min(self._done_thresh, 1) <= self.percent_covered():
            reward += self._terminal_reward

        return state, reward

    def isInBounds(self, x, y):
        return x >= 0 and x < self._gridwidth and y >= 0 and y < self._gridlen

    #TODO remove the for loop so this is actually fast
    def isOccupied(self, x, y):
        #checking if no obstacle in that spot
        if(self._grid[x][y] < 0):
            return True

        #checking if no other robots are there
        for a,b in zip(self._xinds, self._yinds):
            if(a == x and b == y):
                return True

        return False

    def get_distance_map(self):
        inv = np.bitwise_not(self._free.astype('?')).astype(np.uint8)
        distance_map = cv2.distanceTransform(inv, cv2.DIST_L1,
                                             cv2.DIST_MASK_PRECISE)

        # scale map values to be between 0 and 1
        if np.max(distance_map) > 0:
            distance_map = distance_map / np.max(distance_map)

        # invert the values
        return 1 - distance_map

    def get_state(self):
        # distance_map = self.get_distance_map()
        #
        # arrays = np.array(self.get_pos_image() + [self._observed_obstacles, self._free, distance_map])

        arrays = np.array(self.get_pos_image() + [self._observed_obstacles, self._free])

        state = np.stack(arrays, axis=0)
        return state

    def get_pos_image(self):
        """
        get_pos_image uses the list of robot positions to generate an image of
        the size of the overall map, where 1 denotes a space that a robot
        occupies and 0 denotes a free space. It uses only info determined from
        the current timestep.

        Return
        ------
        Image encoding the robot positions
        """
        if self._use_scanning:
            ret = np.zeros((self._gridwidth, self._gridlen))
            for i, j in zip(self._xinds, self._yinds):
                ret[i, j] = 1
            ret = [ret]
        else:
            ret = []
            for i, j in zip(self._xinds, self._yinds):
                robotlayer = np.zeros((self._gridwidth, self._gridlen))
                robotlayer[i, j] = 1
                ret.append(robotlayer)
        return ret

    def reset(self):
        #picking a map at random
        self._grid = self._gridlis[np.random.randint(len(self._gridlis))]

        #dimensions
        self._gridwidth = self._grid.shape[0]
        self._gridlen = self._grid.shape[1]

        #resetting step count
        self._currstep = 0

        #generating random robot positions
        self._xinds = np.zeros(self._numrobot, dtype=int)
        self._yinds = np.zeros(self._numrobot, dtype=int)

        #repeatedly trying to insert robots
        coord_dict = {}
        count = 0
        while(count != self._numrobot):
            #generating random coordinate
            x = np.random.randint(self._gridwidth)
            y = np.random.randint(self._gridlen)

            #testing if coordinate is not an obstacle or other robot
            if(self._grid[x][y] >= 0 and (x,y) not in coord_dict):
                coord_dict[(x,y)] = 1
                self._xinds[count] = x
                self._yinds[count] = y
                count += 1

        # history of observed obstacles
        self._observed_obstacles = np.zeros((self._gridwidth, self._gridlen))

        # history of free cells
        self._free = np.ones((self._gridwidth, self._gridlen))

        return self.get_state()

    def done(self):
        if min(self._done_thresh, 1) <= self.percent_covered():
            print("Full Environment Covered")
            self._done_thresh += self._done_incr
            return True
        # if self._currstep == self._maxsteps:
        #     return True
        return False

    def percent_covered(self):
        return np.count_nonzero(self._free < 1) / np.count_nonzero(self._grid > 0)

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

        robot_pos_map = self.get_pos_image()[0]
        #adding robot positions to the base
        freelayer = np.stack([255*robot_pos_map,
                              0*robot_pos_map,
                              0*robot_pos_map], -1)
        image += freelayer

        scaling = max(min(1075//self._gridwidth, 1075//self._gridlen), 1)
        image = cv2.resize(image, (0,0), fx=scaling, fy=scaling, interpolation=cv2.INTER_NEAREST)
        image = cv2.copyMakeBorder(image, 0, 1075 - image.shape[1], 0, 1075 - image.shape[0], cv2.BORDER_CONSTANT, value=[0,0,0])

        surf = pygame.surfarray.make_surface(image)
        self._display.blit(surf, (0, 0))
        pygame.display.update()

        #returning the image that was used
        return image
