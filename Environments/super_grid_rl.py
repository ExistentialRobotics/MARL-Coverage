import numpy as np
import matplotlib.pyplot as plt
from . environment import Environment

class SuperGridRL(object):
    """
    A Multi-Agent Grid Environment with a discrete action space for RL testing.
    """
    def __init__(self, controller, numrobot, gridlen, gridwidth, collision_penalty=10, sensesize=1, grid=None, seed=None):
        super().__init__()

        self._numrobot = numrobot
        self._gridlen = gridlen
        self._gridwidth = gridwidth
        self._controller = controller
        self._collision_penalty = collision_penalty

        #sensing radius using chess metric(like how a king moves) -> "Chebyshev distance"
        self._sensesize = sensesize

        #generating robot positions
        self.reset()

        #blank/uniform grid by default
        if grid is None:
            self._grid = np.ones((gridwidth, gridlen))
        else:
            self._grid = grid

        #visited array
        self._visited = np.full((gridwidth, gridlen), False)

        # create seed if user specifies it
        if seed is not None:
            np.random.seed(seed)

    def step(self):
        #initialize reward for this step
        reward = 0

        #sense from all the current robot positions
        for i in range(self._numrobot):
            x = self._xinds[i]
            y = self._yinds[i]

            #looping over all grid cells to sense
            for j in range(x - self._sensesize, x + self._sensesize + 1):
                for k in range(y - self._sensesize, y + self._sensesize + 1):

                    #checking if cell is not visited, in bounds, not an obstacle
                    if(self.isInBounds(j,k) and self._grid[j][k]>=0 and not
                       self._visited[j][k]):

                        #adding reward and marking as visited
                        reward += self._grid[j][k]
                        self._visited[j][k] = True


        #calculate current observation
        #TODO decide on observation format
        observation = None


        #calculate controls from observation
        ulis = self._controller.getControls(observation)

        #update robot positions using controls
        #TODO fix movement switching bugs(dependent on order)
        for i in range(len(ulis)):
            u = ulis[i]
            #left
            if(u == 'l'):
                x = self._xinds[i] - 1
                y = self._yinds[i]

                if(self.isInBounds(x,y) and not self.isOccupied(x,y)):
                    self._xinds[i] = x
                else:
                    reward -= self._collision_penalty
            #right
            elif(u == 'r'):
                x = self._xinds[i] + 1
                y = self._yinds[i]

                if(self.isInBounds(x,y) and not self.isOccupied(x,y)):
                    self._xinds[i] = x
                else:
                    reward -= self._collision_penalty
            #up
            elif(u == 'u'):
                x = self._xinds[i]
                y = self._yinds[i] + 1

                if(self.isInBounds(x,y) and not self.isOccupied(x,y)):
                    self._yinds[i] = y
                else:
                    reward -= self._collision_penalty
            #down
            elif(u == 'd'):
                x = self._xinds[i]
                y = self._yinds[i] - 1

                if(self.isInBounds(x,y) and not self.isOccupied(x,y)):
                    self._yinds[i] = y
                else:
                    reward -= self._collision_penalty

        return reward

    def isInBounds(self, x, y):
        return x >= 0 and x < self._gridwidth and y >= 0 and y < self._gridlen

    def isOccupied(self, x, y):
        #checking if no obstacle in that spot
        if(self._grid[x][y] < 0):
            return True

        #checking if no other robots are there
        for a,b in zip(self._xinds, self._yinds):
            if(a == x and b == y):
                return True

        return False

    def reset(self):
        #generating random robot positions
        #TODO make sure none are spawned on obstacles/other robots
        self._xinds = np.random.randint(self._gridwidth, size=self._numrobot)
        self._yinds = np.random.randint(self._gridlen, size=self._numrobot)

    def render(self):
        #clear canvas
        plt.clf()

        #render all robots
        for i in range(self._numrobot):
            plt.scatter(self._xinds[i] + 0.5, self._yinds[i] + 0.5, s=50)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim([0, self._gridwidth])
        plt.ylim([0, self._gridlen])

        #drawing everything
        plt.draw()
        plt.pause(0.02)

