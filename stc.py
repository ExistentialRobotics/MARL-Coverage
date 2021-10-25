import numpy as np
import math
from Utils.gridmaker import gridgen, gridload
from Environments.dec_grid_rl import DecGridRL
import time

class SpanningTreeCoveragePolicy(object):
    '''
    Online controller that takes incremental observations of the environment and
    can achieve optimal and full coverage in certain conditions (see Shreyas'
    notes).
    '''
    def __init__(self, internal_grid_rad, startpos="bl"):
        super().__init__()

        #describes what subcell we are currently in
        self._curr_pos = startpos
        self._internal_grid_rad = internal_grid_rad

        #reset policy (creates visited array and curr_x, curr_y)
        self.reset()

    def pi(self, obs):
        """
        Args:
            obs : an egocentric observation of radius 2 on the map of subcells,
                  obstacles take value 1, free takes zero
        Returns:
            Returns the controls based on the given observation.
        """
        assert obs.shape == (5,5), "wrong observation dummy"
        print(self._curr_pos)

        # controls cheatsheet
        # 0 - right, 1 - up, 2 - left, 3 - down
        print(self._visited)

        #checking whether the cell is fully explored
        if self.isCellVisited(self._curr_x, self._curr_y):
            #means we got to leave the cell
            if self._curr_pos == "bl":
                u = 3
            elif self._curr_pos == "br":
                u = 0
            elif self._curr_pos == "ur":
                u = 1
            elif self._curr_pos == "ul":
                u = 2

        #exploring further if we can
        else:
            #different checks depending on which sub-cell we are in
            if self._curr_pos == "bl":
                #check if cell below has any obstacles
                obs = obs[2:4,0:2]
                free = not np.any(obs == 1)

                if free and not self.isAnySubcellVisited(self._curr_x, self._curr_y - 1):
                    u = 3
                    self._curr_pos = "ul"
                else:
                    u = 0
                    self._curr_pos = "br"
            elif self._curr_pos == "br":
                #check if cell right has any obstacles
                obs = obs[3:5,2:4]
                free = not np.any(obs == 1)

                if free and not self.isAnySubcellVisited(self._curr_x + 1, self._curr_y):
                    u = 0
                    self._curr_pos = "bl"
                else:
                    u = 1
                    self._curr_pos = "ur"

            elif self._curr_pos == "ur":
                #check if cell above has any obstacles
                obs = obs[1:3,3:5]
                free = not np.any(obs == 1)

                if free and not self.isAnySubcellVisited(self._curr_x, self._curr_y + 1):
                    u = 1
                    self._curr_pos = "br"
                else:
                    u = 2
                    self._curr_pos = "ul"
            elif self._curr_pos == "ul":
                #check if cell above has any obstacles
                obs = obs[0:2,1:3]
                free = not np.any(obs == 1)

                if free and not self.isAnySubcellVisited(self._curr_x - 1, self._curr_y):
                    u = 2
                    self._curr_pos = "ur"
                else:
                    u = 3
                    self._curr_pos = "bl"

        #updating robot x, y based on controls
        if u == 0:
            self._curr_x += 1
        elif u == 1:
            self._curr_y += 1
        elif u == 2:
            self._curr_x -= 1
        elif u == 3:
            self._curr_y -= 1

        #visiting the current cell
        self._visited[self._curr_x][self._curr_y] = 1

        return u

    def isCellVisited(self,x,y):
        '''
        returns true if all subcells in the enclosing cell of (x,y) are visited
        '''
        #making coordinates point to bottom left of cell
        x = x - x % 2
        y = y - y % 2

        cell = self._visited[x:x+2,y:y+2]
        return np.all(cell == 1)

    def isAnySubcellVisited(self, x, y):
        '''
        returns true if any subcell in the enclosing cell of (x,y) is visited
        '''
        #making coordinates point to bottom left of cell
        x = x - x % 2
        y = y - y % 2

        cell = self._visited[x:x+2,y:y+2]
        return np.any(cell == 1)

    def reset(self):
        '''
        resets the policy to run again on a different environment
        '''
        internal_grid_rad = self._internal_grid_rad

        #internal coordinate system for keeping track of where we have been
        self._visited = np.zeros((2*internal_grid_rad, 2*internal_grid_rad))

        #setting the starting x,y
        index = int((internal_grid_rad + internal_grid_rad % 2)/2 - 1)
        xinc = 0
        yinc = 0
        if self._curr_pos == "br":
            xinc = 1
            yinc = 0
        elif self._curr_pos == "ur":
            xinc = 1
            yinc = 1
        elif self._curr_pos == "ul":
            xinc = 0
            yinc = 1
        self._curr_x = 2*index + xinc
        self._curr_y = 2*index + yinc

        #visiting the current cell
        self._visited[self._curr_x][self._curr_y] = 1

if __name__ == "__main__":
    #testing spanning tree coverage on dec_grid_rl environment
    env_config = {
        "numrobot": 1,
        "maxsteps": 60,
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
        "single_square_tool" : 1
    }

    grid_config = {
        "gridwidth": 20,
        "gridlen": 20,
        "numgrids": 30,
        "prob_obst": 0
    }

    '''Making the list of grids'''
    gridlis = gridgen(grid_config)

    env = DecGridRL(gridlis, env_config)

    #testing stc
    stc_controller = SpanningTreeCoveragePolicy(10)

    state = np.squeeze(env.reset())[2] #getting only the obstacle layer
    done = False
    render = True

    #simulating
    while not done:
        # determine action
        action = stc_controller.pi(state)
        # print(action)

        # step environment and save episode results
        state, reward = env.step(action)
        state = np.squeeze(state)[2] #getting only the obstacle layer

        # determine if episode is completed
        done = env.done()

        # render if necessary
        if render:
            frame = env.render()
            time.sleep(2)


