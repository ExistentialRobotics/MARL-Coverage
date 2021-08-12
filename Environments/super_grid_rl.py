import numpy as np
import matplotlib.pyplot as plt
from . environment import Environment
from queue import PriorityQueue

class SuperGridRL(object):
    """
    A Multi-Agent Grid Environment with a discrete action space for RL testing.
    """
    def __init__(self, numrobot, gridlen, gridwidth, discrete_grid_values=2, collision_penalty=5, sensesize=1, grid=None, seed=None, free_penalty=0):
        super().__init__()

        self._numrobot = numrobot
        self._gridlen = gridlen
        self._gridwidth = gridwidth
        self._collision_penalty = collision_penalty
        self._free_penalty = free_penalty

        #sensing radius using chess metric(like how a king moves) -> "Chebyshev distance"
        self._sensesize = sensesize

        self._discrete_grid_values = discrete_grid_values

        #blank/uniform grid by default
        if grid is None:
            self._grid = np.ones((gridwidth, gridlen))
        else:
            self._grid = grid

        #normalizing grid into discrete sensing levels
        gridmax = np.amax(self._grid) + 1e-3
        self._grid *= 1.0/gridmax
        self._grid *= discrete_grid_values
        self._grid = self._grid.astype(int)

        # history of observed cells (their values)
        self._observed_cells = []
        #making one layer for each sensing level
        for i in range(discrete_grid_values - 1): #-1 is there because we don't want to include a grid for zero value
            self._observed_cells.append(np.zeros((gridwidth, gridlen)))

        # history of observed obstacles
        self._observed_obstacles = np.zeros((gridwidth, gridlen))

        # history of free cells
        self._free = np.ones((gridwidth, gridlen))

        #generating robot positions
        self.reset()

        # create seed if user specifies it
        if seed is not None:
            np.random.seed(seed)

    def step(self, ulis):
        #initialize reward for this step
        reward = np.zeros((self._numrobot,))

        #update robot positions using controls
        newx = self._xinds
        newy = self._yinds

        #making pq for sorting by minimum scan score
        pq = PriorityQueue()
        for i in range(self._numrobot):
            score = self._xinds[i] + self._yinds[i]*self._gridwidth
            pq.put((score, i))

        #robot 2 control, storing what robot got what control
        r2c = np.zeros((self._numrobot,), dtype=int)

        for i in range(len(ulis)):
            u = ulis[i]

            #z is the robot index we are assigning controls to
            z = pq.get()[1]
            r2c[z] = i

            #left
            if(u == 0):
                x = self._xinds[z] - 1
                y = self._yinds[z]

                if(self.isInBounds(x,y) and self._grid[x][y]>=0):
                    newx[z] = x
                else:
                    reward[i] -= self._collision_penalty
            #right
            elif(u == 1):
                x = self._xinds[z] + 1
                y = self._yinds[z]

                if(self.isInBounds(x,y) and self._grid[x][y]>=0):
                    newx[z] = x
                else:
                    reward[i] -= self._collision_penalty
            #up
            elif(u == 2):
                x = self._xinds[z]
                y = self._yinds[z] + 1

                if(self.isInBounds(x,y) and self._grid[x][y]>=0):
                    newy[z] = y
                else:
                    reward[i] -= self._collision_penalty
            #down
            elif(u == 3):
                x = self._xinds[z]
                y = self._yinds[z] - 1

                if(self.isInBounds(x,y) and self._grid[x][y]>=0):
                    newy[z]= y
                else:
                    reward[i] -= self._collision_penalty

        #checking if any robots are at same position
        coord_dict = {}
        for i in range(self._numrobot):
            if (newx[i], newy[i]) in coord_dict:
                coord_dict[(newx[i], newy[i])].append(i)
            else:
                coord_dict[(newx[i], newy[i])] = [i]

        #only updating positions of robots that ended in a unique position
        for coord in coord_dict:
            robots = coord_dict[coord]
            if(len(robots) == 1):
                self._xinds[robots[0]] = newx[robots[0]]
                self._yinds[robots[0]] = newy[robots[0]]
            else:
                # print(robots)
                #penalizing all robots that tried to end up in the same place
                for r in robots:
                    reward[r2c[r]] -= self._collision_penalty


        #sense from all the current robot positions
        for i in range(self._numrobot):
            x = self._xinds[i]
            y = self._yinds[i]

            #looping over all grid cells to sense
            for j in range(x - self._sensesize, x + self._sensesize + 1):
                for k in range(y - self._sensesize, y + self._sensesize + 1):
                    #checking if cell is not visited, in bounds, not an obstacle
                    if(self.isInBounds(j,k) and self._grid[j][k]>=0 and
                       self._free[j][k] == 1):
                        # add reward
                        reward[r2c[i]] += self._grid[j][k]

                        # record observation value
                        sensing_level = self._grid[j][k]
                        if(sensing_level > 0):
                            self._observed_cells[sensing_level - 1][j][k] = 1

                        # mark as not free
                        self._free[j][k] = 0

                    elif(self.isInBounds(j,k) and self._grid[j][k]>=0 and
                       self._free[j][k] == 0):
                        reward[r2c[i]] -= self._free_penalty

                    elif(self.isInBounds(j,k) and self._grid[j][k]<0 and
                         self._observed_obstacles[j][k] == 0):
                         # track observed obstacles
                         self._observed_obstacles[j][k] = 1

        #calculate current state
        state = self.get_state()

        return state, reward

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

    def get_state(self):
        arrays = np.array([self.get_pos_image(), self._observed_obstacles, self._free] + self._observed_cells)
        return np.expand_dims(np.stack(arrays, axis=0), axis=0)

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
        ret = np.zeros((self._gridwidth, self._gridlen))
        for i, j in zip(self._xinds, self._yinds):
            ret[i, j] = 1
        return ret

    def reset(self):
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

        # history of observed cells (their values)
        self._observed_cells = []
        #making one layer for each sensing level
        for i in range(self._discrete_grid_values - 1): #-1 is there because we don't want to include a grid for zero value
            self._observed_cells.append(np.zeros((self._gridwidth, self._gridlen)))

        # history of observed obstacles
        self._observed_obstacles = np.zeros((self._gridwidth, self._gridlen))

        # history of free cells
        self._free = np.ones((self._gridwidth, self._gridlen))

        return self.get_state()

    def done(self, thres=1):
        if thres <= self.percent_covered():
            print("Full Environment Covered")
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

        plt.imshow(np.transpose(self._free), extent=[0, self._gridwidth, self._gridlen, 0])

        #drawing everything
        plt.draw()
        plt.pause(0.02)
