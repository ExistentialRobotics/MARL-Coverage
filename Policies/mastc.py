import numpy as np
import math
from Utils.gridmaker import gridgen, gridload
from Environments.dec_grid_rl import DecGridRL
import time
from Logger.logger import Logger

"""
TODO problems to address ???
- multiple robots starting in same cell?
"""

class MultiAgentSpanningTreeCoveragePolicy(object):
    '''
    Online controller that takes incremental observations of the environment and
    can achieve optimal and full coverage in certain conditions (see Shreyas'
    notes).
    '''
    def __init__(self, numrobot, internal_grid_rad):
        super().__init__()

        #describes what subcell we are currently in
        self._internal_grid_rad = internal_grid_rad
        self._numrobot = numrobot

    def pi(self, obs):
        """
        Args:
            obs : an egocentric observation of radius 2 on the map of subcells,
                  obstacles take value 1, free takes zero
        Returns:
            Returns the controls based on the given observation.
        """
        assert obs.shape == (self._numrobot, 5,5), "wrong observation dummy"

        # controls cheatsheet
        # 0 - right, 1 - up, 2 - left, 3 - down
        ulis = np.zeros((self._numrobot,))

        #performing controls for each robot
        for i in range(self._numrobot):
            u = -1

            curr_pos = self.subcellpos(self._curr_x[i], self._curr_y[i])

            #checking whether the cell is fully explored
            if self.isCellVisited(self._curr_x[i], self._curr_y[i], i):
                print('cell fully visited, leaving')

                #means we got to leave the cell
                if curr_pos == "bl":
                    u = 3
                elif curr_pos == "br":
                    u = 0
                elif curr_pos == "ur":
                    u = 1
                elif curr_pos == "ul":
                    u = 2

            #exploring further if we can
            else:
                #different checks depending on which sub-cell we are in
                if curr_pos == "bl":
                    #check if cell below has any obstacles
                    o = obs[i,2:4,0:2]
                    free = not np.any(o == 1)

                    if free and not self.isAnySubcellVisited(self._curr_x[i], self._curr_y[i] - 1):
                        u = 3
                    else:
                        print("obstacle below")
                        u = 0

                elif curr_pos == "br":
                    #check if cell right has any obstacles
                    o = obs[i, 3:5,2:4]
                    free = not np.any(o == 1)

                    if free and not self.isAnySubcellVisited(self._curr_x[i] + 1, self._curr_y[i]):
                        u = 0
                    else:
                        print("obstacle right")
                        u = 1

                elif curr_pos == "ur":
                    #check if cell above has any obstacles
                    o = obs[i,1:3,3:5]
                    free = not np.any(o == 1)

                    if free and not self.isAnySubcellVisited(self._curr_x[i], self._curr_y[i] + 1):
                        u = 1
                    else:
                        u = 2
                elif curr_pos == "ul":
                    #check if cell above has any obstacles
                    o = obs[i,0:2,1:3]
                    free = not np.any(o == 1)

                    if free and not self.isAnySubcellVisited(self._curr_x[i] - 1, self._curr_y[i]):
                        u = 2
                    else:
                        u = 3

            #updating robot x, y based on controls
            if u == 0:
                self._curr_x[i] += 1
            elif u == 1:
                self._curr_y[i] += 1
            elif u == 2:
                self._curr_x[i] -= 1
            elif u == 3:
                self._curr_y[i] -= 1

            #adding controls to list
            ulis[i] = u

            #visiting the current cell
            self._visited[i][self._curr_x[i]][self._curr_y[i]] = 1

        return ulis

    def isCellVisited(self,x,y,i):
        '''
        returns true if all subcells in the enclosing cell of (x,y) are visited
        '''
        #making coordinates point to bottom left of cell
        x = x - x % 2
        y = y - y % 2

        cell = self._visited[i,x:x+2,y:y+2]
        return np.all(cell == 1)

    def isAnySubcellVisited(self, x, y):
        '''
        returns true if any subcell in the enclosing cell of (x,y) is visited
        '''
        #making coordinates point to bottom left of cell
        x = x - x % 2
        y = y - y % 2

        cell = self._visited[:,x:x+2,y:y+2]
        return np.any(cell == 1)

    def subcellpos(self, x, y):
        '''
        Tells us what subcell we are in based on the coordinates
        '''
        if x % 2 == 0:
            if y % 2 == 0:
                return "bl"
            else:
                return "ul"
        else:
            if y % 2 == 0:
                return "br"
            else:
                return "ur"

    def reset(self, x_coordinates, y_coordinates):
        """Resets the policy by creating a visited array and initializing
        start coordinates

        Args:
           x_coordinates : the initial x coordinates of the robots
           y_coordinates : the initial y coordinates of the robots
        """
        internal_grid_rad = self._internal_grid_rad

        #internal coordinate system for keeping track of where we have been
        self._visited = np.zeros((self._numrobot, 2*internal_grid_rad, 2*internal_grid_rad))

        #creating positions array
        self._curr_x = np.zeros(self._numrobot, dtype=int)
        self._curr_y = np.zeros(self._numrobot, dtype=int)

        #normalizing the given start coordinates
        x_coordinates = x_coordinates - x_coordinates[0]
        y_coordinates = y_coordinates - y_coordinates[0]

        #setting the starting x,y for the first robot
        index = int((internal_grid_rad + internal_grid_rad % 2)/2 - 1)

        #setting all robot coordinates
        for i in range(self._numrobot):
            self._curr_x[i] = 2*index + x_coordinates[i]
            self._curr_y[i] = 2*index + y_coordinates[i]

            #visiting the current cell
            self._visited[i][self._curr_x[i]][self._curr_y[i]] = 1

if __name__ == "__main__":
    #testing spanning tree coverage on dec_grid_rl environment
    env_config = {
        "numrobot": 10,
        "maxsteps": 60000,
        "collision_penalty": 5,
        "senseradius": 2,
        "egoradius": 2,
        "free_penalty": 0,
        "done_thresh": 1,
        "done_incr": 0,
        "terminal_reward": 30,
        "mini_map_rad" : 0,
        "comm_radius" : 0,
        "allow_comm" : 0,
        "map_sharing" : 0,
        "single_square_tool" : 1,
        "dist_reward" : 0
    }

    grid_config = {
        "grid_dir": "./Grids/bg2_100x100",
        "gridwidth": 200,
        "gridlen": 200,
        "numgrids": 30,
        "prob_obst": 0
    }

    '''Making the list of grids'''
    # gridlis = gridgen(grid_config)
    gridlis = gridload(grid_config)

    env = DecGridRL(gridlis, env_config)

    #logger stuff
    makevid = True
    exp_name = "stcEmptyGrid1"
    logger = Logger(exp_name, makevid)

    #testing stc
    stc_controller = MultiAgentSpanningTreeCoveragePolicy(10,210)

    state = (env.reset())[:,2,:,:]#getting only the obstacle layers
    stc_controller.reset(env._xinds, env._yinds)
    done = False
    render = True

    #simulating
    while not done:
        # determine action
        action = stc_controller.pi(state)

        # step environment and save episode results
        state, reward = env.step(action)
        state = state[:,2,:,:] #getting only the obstacle layers

        # determine if episode is completed
        done = env.done()

        # render if necessary
        if render:
            frame = env.render()
            if(makevid):
                logger.addFrame(frame)

