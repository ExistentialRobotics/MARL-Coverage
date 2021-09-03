import numpy as np
import matplotlib.pyplot as plt
from . environment import Environment
from queue import PriorityQueue

class DecGridRL(object):
    """
    A Multi-Agent Grid Environment with a discrete action space for RL testing.
    """
    def __init__(self, numrobot, gridlis, maxsteps, collision_penalty=5,
                 senseradius=1, egoradius=12, seed=None, free_penalty=0,
                 done_thresh=1, done_incr=0, terminal_reward=0):
        super().__init__()
        self._numrobot = numrobot
        self._gridlis = gridlis
        self._collision_penalty = collision_penalty
        self._free_penalty = free_penalty

        #sensing radius using chess metric(like how a king moves) -> "Chebyshev distance"
        self._senseradius = senseradius
        self._egoradius = egoradius

        # create seed if user specifies it
        if seed is not None:
            np.random.seed(seed)

        #pick random map and generate robot positions
        self.reset()

        #observation and action dimensions
        #TODO fix this stuff for multiagent
        self._obs_dim = self.get_egocentric_observations()[0].shape
        self._num_actions = 4

        #maximum steps in an episode
        self._maxsteps = maxsteps

        #done threshold
        self._done_thresh = done_thresh
        self._done_incr = done_incr
        self._terminal_reward = terminal_reward


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
        return (np.count_nonzero(self._free < 1) / (self._gridlen * self._gridwidth))

    def render(self):
        #clear canvas
        plt.clf()

        #render all robots
        for i in range(self._numrobot):
            plt.scatter(self._xinds[i] + 0.5, self._yinds[i] + 0.5, s=50)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim([0, self._gridwidth])
        plt.ylim([0, self._gridlen])

        #setting gridlines to be every 1
        plt.gca().set_xticks(np.arange(0, self._gridwidth, 1))
        plt.gca().set_yticks(np.arange(0, self._gridlen, 1))
        plt.grid()

        #preprocessing grid to graph
        # obs = np.clip(self._grid, -1, 0)
        grid = 2*self._free - self._observed_obstacles

        plt.imshow(np.transpose(grid), extent=[0, self._gridwidth, self._gridlen, 0])

        #drawing everything
        plt.draw()
        plt.pause(0.02)
