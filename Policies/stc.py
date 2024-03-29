import numpy as np
import math
from Utils.gridmaker import gridgen, gridload
from Environments.dec_grid_rl import DecGridRL
import time
from Logger.logger import Logger
from . base_policy import Base_Policy


class STC(Base_Policy):
    '''
    Online controller that takes incremental observations of the environment and
    can achieve optimal and full coverage in certain conditions (see Shreyas'
    notes).
    '''

    def __init__(self, internal_grid_rad, startpos="bl"):
        super().__init__()

        # describes what subcell we are currently in
        self._curr_pos = startpos
        self._internal_grid_rad = internal_grid_rad

        # reset policy (creates visited array and curr_x, curr_y)
        self.reset(False, None)

    def pi(self, obs):
        """
        Args:
            obs : an egocentric observation of radius 2 on the map of subcells,
                  obstacles take value 1, free takes zero
        Returns:
            Returns the controls based on the given observation.
        """
        obs = np.squeeze(obs)[2]
        assert obs.shape == (5, 5), "wrong observation dummy"

        # controls cheatsheet
        # 0 - right, 1 - up, 2 - left, 3 - down

        u = -1
        # checking whether the cell is fully explored
        if self.isCellVisited(self._curr_x, self._curr_y):
            # print('cell fully visited, leaving')

            # means we got to leave the cell
            if self._curr_pos == "bl":
                u = 3
            elif self._curr_pos == "br":
                u = 0
            elif self._curr_pos == "ur":
                u = 1
            elif self._curr_pos == "ul":
                u = 2

        # exploring further if we can
        else:
            # different checks depending on which sub-cell we are in
            if self._curr_pos == "bl":
                # check if cell below has any obstacles
                obs = obs[2:4, 0:2]
                free = not np.any(obs == 1)

                if free and not self.isAnySubcellVisited(self._curr_x, self._curr_y - 1):
                    u = 3
                else:
                    # print("obstacle below")
                    u = 0
            elif self._curr_pos == "br":
                # check if cell right has any obstacles
                obs = obs[3:5, 2:4]
                free = not np.any(obs == 1)

                if free and not self.isAnySubcellVisited(self._curr_x + 1, self._curr_y):
                    u = 0
                else:
                    # print("obstacle right")
                    u = 1

            elif self._curr_pos == "ur":
                # check if cell above has any obstacles
                obs = obs[1:3, 3:5]
                free = not np.any(obs == 1)

                if free and not self.isAnySubcellVisited(self._curr_x, self._curr_y + 1):
                    u = 1
                else:
                    # print("obstacle above")
                    u = 2

            elif self._curr_pos == "ul":
                # check if cell above has any obstacles
                obs = obs[0:2, 1:3]
                free = not np.any(obs == 1)

                if free and not self.isAnySubcellVisited(self._curr_x - 1, self._curr_y):
                    u = 2
                else:
                    # print("obstacle left")
                    u = 3

        # updating robot x, y based on controls
        if u == 0:
            self._curr_x += 1
        elif u == 1:
            self._curr_y += 1
        elif u == 2:
            self._curr_x -= 1
        elif u == 3:
            self._curr_y -= 1

        # updating the subcell pos
        self._curr_pos = self.subcellpos(self._curr_x, self._curr_y)

        # visiting the current cell
        if self._visited[self._curr_x][self._curr_y] == 1:
            u = -1
        else:
            self._visited[self._curr_x][self._curr_y] = 1

        return u

    def isCellVisited(self, x, y):
        '''
        returns true if all subcells in the enclosing cell of (x,y) are visited
        '''
        # making coordinates point to bottom left of cell
        x = x - x % 2
        y = y - y % 2

        cell = self._visited[x:x+2, y:y+2]
        return np.all(cell == 1)

    def isAnySubcellVisited(self, x, y):
        '''
        returns true if any subcell in the enclosing cell of (x,y) is visited
        '''
        # making coordinates point to bottom left of cell
        x = x - x % 2
        y = y - y % 2

        cell = self._visited[x:x+2, y:y+2]
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

    def reset(self, testing, grid):
        '''
        resets the policy to run again on a different environment
        '''
        internal_grid_rad = self._internal_grid_rad

        # internal coordinate system for keeping track of where we have been
        self._visited = np.zeros((2*internal_grid_rad, 2*internal_grid_rad))

        # setting the starting x,y
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

        # visiting the current cell
        self._visited[self._curr_x][self._curr_y] = 1


if __name__ == "__main__":
    # testing spanning tree coverage on dec_grid_rl environment
    env_config = {
        "numrobot": 1,
        "maxsteps": 200,
        "collision_penalty": 5,
        "egoradius": 2,
        "done_thresh": 1,
        "done_incr": 0,
        "terminal_reward": 30,
        "mini_map_rad": 0,
        "comm_radius": 0,
        "allow_comm": 0,
        "map_sharing": 0,
        "single_square_tool": 1,
        "dist_reward": 0,
        "dijkstra_input": 0,
        "sensor_type": "square_sensor",
        "sensor_config": {
            "range": 2
            }

    }

    grid_config = {
        "grid_dir": "./Grids/bg2_100x100",
        "gridwidth": 30,
        "gridlen": 30,
        "numgrids": 30,
        "prob_obst": 0
    }

    '''Making the list of grids'''
    train_set, test_set = gridload(grid_config)

    env = DecGridRL(train_set, env_config, test_set=test_set)

    #logger stuff
    makevid = True
    exp_name = "stcEmptyGrid1"
    logger = Logger(exp_name, makevid)

    #testing stc
    stc_controller = STC(105)

    # getting only the obstacle layer
    state = np.squeeze(env.reset(False, None))[2]
    done = False
    render = True

    #simulating
    while not done:
        # determine action
        action = stc_controller.pi(state)

        # step environment and save episode results
        state, reward = env.step(action)
        state = np.squeeze(state)[2]  # getting only the obstacle layer

        # determine if episode is completed
        done = env.done()

        # render if necessary
        if render:
            frame = env.render()
            if(makevid):
                logger.addFrame(frame)
