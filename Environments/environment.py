import numpy as np
import matplotlib.pyplot as plt

class Environment(object):
    """
    Class Environment represents a 2d multi agent environment where
    a number of agents have to maximize their cumulative sensing reward.
    """
    def __init__(self, agents, numobstacles, dt, seed=None):
        #TODO incorporate sensing function in environment
        super().__init__()

        self.agents = agents 
        self._numobstacles = numobstacles
        self._dt = dt

        # create seed if user specifies it
        if seed is not None:
            np.random.seed(seed)

        # setting random initial agent and obstacle positions
        self.reset()

    def step(self):
        for agent in self.agents:
            agent.odom_command()

        #euler integrating the controls forward
        # for i in range(len(U)):
            #TODO handle obstacle 'collisions'
            # self._agent_coordinates[i] += U[i] * self._dt

        #returning current agent and obstacle positions
        #TODO add reward (negative coverage cost)
        # return self._agent_coordinates, self._obst_coordinates

    def reset(self):
        for agent in self.agents:
            agent.reset()

        #populating agent position list
        # self._agent_coordinates = []
        # for i in range(self._numagent):
        #     xcoor = np.random.random_sample()
        #     ycoor = np.random.random_sample()
        #     self._agent_coordinates.append(np.array([[xcoor], [ycoor]]))

        #populating obstacle position list
        # self._obst_coordinates = []
        # for i in range(self._numobstacles):
        #     xcoor = self.map_width*self.cell_size*np.random.random_sample()
        #     ycoor = self.map_height*self.cell_size*np.random.random_sample()
        #     self._obst_coordinates.append(np.array([[xcoor], [ycoor]]))


    #TODO need to figure out a good way to do this, my previous way
    #doesn't work well for obstacles
    def render(self):
        for agent in agents:
            agent.render()
