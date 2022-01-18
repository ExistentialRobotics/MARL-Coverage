import numpy as np
import math
from Utils.gridmaker import gridgen, gridload
from Environments.dec_grid_rl import DecGridRL
import time
from Logger.logger import Logger

class DijkstraFrontier(object):
    '''
    Online controller that takes incremental observations of the environment
    and greedily explores by going to the closest unexplored point.
    '''
    def pi(self, obs):
        """
        Args:
            obs : an egocentric observation of radius 1 on the map of subcells,
                  obstacles take value 1, free takes zero, includes dijkstra path layer
        Returns:
            Returns the controls based on the given observation.
        """
        #taking shortest path
        obs = obs[3]

        #right
        if obs[2,1] == 1:
            u = 0
        #up
        elif obs[1,2] == 1:
            u = 1
        #left
        elif obs[0,1] == 1:
            u = 2
        #down
        elif obs[1,0] == 1:
            u = 3
        else:
            print("Map is full covered")
            u = 0
        return u


if __name__ == "__main__":
    #testing frontier coverage on dec_grid_rl environment
    env_config = {
        "numrobot": 1,
        "maxsteps": 60000,
        "collision_penalty": 5,
        "egoradius": 1,
        "done_thresh": 1,
        "done_incr": 0,
        "terminal_reward": 30,
        "mini_map_rad" : 0,
        "comm_radius" : 0,
        "allow_comm" : 0,
        "map_sharing" : 0,
        "single_square_tool" : 0,
        "dist_reward" : 0,
        "dijkstra_input" : 1,
        "sensor_type" : "lidar",
        "sensor_config" : {
            "num_lasers" : 21,
            "range" : 10
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
    # gridlis = gridgen(grid_config)
    gridlis = gridload(grid_config)

    env = DecGridRL(gridlis, env_config)

    #logger stuff
    makevid = True
    exp_name = "dijkstrafrontier1"
    logger = Logger(exp_name, makevid)

    #testing stc
    frontier_controller = DijkstraFrontier()

    state = np.squeeze(env.reset())
    print(state.shape)
    done = False
    render = True

    #simulating
    while not done:
        # determine action
        action = frontier_controller.pi(state)

        # step environment and save episode results
        state, reward = env.step(action)
        state = np.squeeze(state)

        # determine if episode is completed
        done = env.done()

        # render if necessary
        if render:
            frame = env.render()
            if(makevid):
                logger.addFrame(frame)

