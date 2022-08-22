import numpy as np
from Utils.gridmaker import gridload
from queue import PriorityQueue
from Environments.dec_grid_rl import DecGridRL
from Logger.logger import Logger
import copy


def coord_to_rc(x, y):
    return (y, x)


class BSA(object):
    '''
    Online controller that takes incremental observations of the environment and
    can achieve optimal and full coverage in certain conditions (see Shreyas'
    notes).
    '''

    def __init__(self, internal_grid_rad):
        super().__init__()

        # describes what subcell we are currently in
        self._internal_grid_rad = internal_grid_rad

        self.num_actions = 4

        # reset policy (creates visited array and curr_x, curr_y)
        self.reset()

    def get_obs_vis(self, state):
        # getting only the obstacle layer
        obs = copy.deepcopy(np.squeeze(state)[2])
        vis = copy.deepcopy(
            self._visited[self._curr_x - 1:self._curr_x + 2, self._curr_y - 1:self._curr_y + 2])

        if self._curr_a == 0:
            r = 1
        elif self._curr_a == 90:
            r = 0
        elif self._curr_a == 180:
            r = 3
        else:
            r = 2

        for i in range(r):
            vis = np.rot90(vis)

        if self._curr_a == 0:
            r = 1
        elif self._curr_a == 90:
            r = 0
        elif self._curr_a == 180:
            r = 3
        else:
            r = 2

        for i in range(r):
            obs = np.rot90(obs)

        return obs, vis

    def pi(self, state):
        # getting only the obstacle layer
        obs, vis = self.get_obs_vis(state)

        if obs[0, 1] != 1 and vis[0, 1] != 1:
            self.turn_left()
        elif obs[1, 2] != 1 and vis[1, 2] != 1:
            pass
        elif obs[2, 1] != 1 and vis[2, 1] != 1:
            self.turn_right()
        else:
            print("Running frontier based!")
            self._prev_a = self.frontier_based(state)

            if self._prev_a == 0:
                self._curr_a = 0
            elif self._prev_a == 1:
                self._curr_a = 90
            elif self._prev_a == 2:
                self._curr_a = 180
            else:
                self._curr_a = 270

        u = self._prev_a

        # updating robot x, y based on controls
        if u == 0:
            self._curr_x += 1
        elif u == 1:
            self._curr_y += 1
        elif u == 2:
            self._curr_x -= 1
        elif u == 3:
            self._curr_y -= 1

        # visiting the current cell
        self._visited[self._curr_x][self._curr_y] = 1

        return u

    def turn_left(self):
        self._curr_a = (self._curr_a + 90) % 360
        self._prev_a = (self._prev_a + 1) % 4

    def turn_right(self):
        self._curr_a -= 90
        if self._curr_a < 0:
            self._curr_a = 270
        self._prev_a -= 1
        if self._prev_a < 0:
            self._prev_a = 3

    def frontier_based(self, state):
        pos_img, observed_obs, free, path_map = state[0]

        # get robot position
        pos = np.nonzero(pos_img)

        # determine action
        u = -1
        for i in range(self.num_actions):
            if i == 0:
                p = (pos[0] + 1, pos[1])
            elif i == 1:
                p = (pos[0], pos[1] + 1)
            elif i == 2:
                p = (pos[0] - 1, pos[1])
            elif i == 3:
                p = (pos[0], pos[1] - 1)
            if path_map[p] == 1:
                u = i
                break

        if u == -1:
            u = 0

        return u

    def reset(self):
        '''
        resets the policy to run again on a different environment
        '''
        internal_grid_rad = self._internal_grid_rad

        # internal coordinate system for keeping track of where we have been
        self._visited = np.zeros((2*internal_grid_rad, 2*internal_grid_rad))

        # setting the starting x,y
        index = int((internal_grid_rad + internal_grid_rad % 2)/2 - 1)

        self._curr_x = 2*index
        self._curr_y = 2*index

        self._curr_a = 0
        self._prev_a = 0

        # visiting the current cell
        self._visited[self._curr_x][self._curr_y] = 1


if __name__ == "__main__":
    # testing spanning tree coverage on dec_grid_rl environment
    env_config = {
        "numrobot": 1,
        "maxsteps": 10000,
        "collision_penalty": 5,
        "egoradius": 1,
        "done_thresh": 1,
        "done_incr": 0,
        "terminal_reward": 30,
        "mini_map_rad": 0,
        "comm_radius": 0,
        "allow_comm": 0,
        "map_sharing": 0,
        "single_square_tool": 1,
        "dist_reward": 0,
        "dijkstra_input": 1,
        "sensor_type": "square_sensor",
        "sensor_config": {
            "range": 1
            }
    }

    grid_config = {
        "grid_dir": "./Grids/bg2_100x100",
        "gridwidth": 100,
        "gridlen": 100,
        "numgrids": 30,
        "prob_obst": 0
    }

    '''Making the list of grids'''
    gridlis = gridload(grid_config)
    # train_set, test_set = gridload(grid_config)

    env = DecGridRL(gridlis, env_config)

    # logger stuff
    makevid = True
    exp_name = "stcEmptyGrid1"
    logger = Logger(exp_name, makevid)

    # testing bsa
    bsa_controller = BSACoveragePolicy(155)

    state = env.reset()  # getting only the obstacle layer
    done = False
    render = True

    # simulating
    while not done:
        # determine action
        action = bsa_controller.pi(state)

        # step environment and save episode results
        state, reward = env.step(action)

        # determine if episode is completed
        done = env.done()

        # render if necessary
        if render:
            frame = env.render()
            if(makevid):
                logger.addFrame(frame)
