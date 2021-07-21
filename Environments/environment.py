import numpy as np
import matplotlib.pyplot as plt
from Agents.swarm_agent import Swarm_Agent
from Agents.agent import Agent

class Environment(object):
    """
    Class Environment represents a 2d multi agent environment where
    a number of agents have to maximize their cumulative sensing reward.
    """
    def __init__(self, agents, numrobot, numobstacles, obstradius, region, dt, seed=None):
        super().__init__()

        self.agents = agents
        self._numobstacles = numobstacles
        self._obstradius = obstradius
        self._numrobot = numrobot
        self._dt = dt

        #assume region is a rectangle in R^2 defined by a lower right coordinate
        #and a sidelength in each direction
        self._region = region

        # create seed if user specifies it
        if seed is not None:
            np.random.seed(seed)

        # setting random initial agent and obstacle positions
        self.reset()

    def step(self):
        for agent in self.agents:
            agent.step(self._obstlist, self._dt)

    def reset(self):
        #reset all agents
        for agent in self.agents:
            agent.reset()

        #to save some typing
        region = self._region

        #populating obstacle position list
        self._obstlist = []
        for i in range(self._numobstacles):
            xcoor = region[1][0]*np.random.random_sample() + region[0][0]
            ycoor = region[1][1]*np.random.random_sample() + region[0][1]
            self._obstlist.append((np.array([[xcoor], [ycoor]]), self._obstradius))

        #generating robot positions
        #TODO add check that robots aren't generated within obstacle
        xlis = []
        for i in range(self._numrobot):
            xcoor = region[1][0]*np.random.random_sample() + region[0][0]
            ycoor = region[1][1]*np.random.random_sample() + region[0][1]
            xlis.append(np.array([[xcoor], [ycoor]]))
        xlis = np.squeeze(np.array(xlis), axis=2)

        print(xlis)

        #swarm agent
        if isinstance(self.agents[0], Swarm_Agent):
            self.agents[0].setPositions(xlis)

        #single agents
        else:
            for agent, x in zip(self.agents, xlis):
                agent.setPos(x)


    def render(self):
        '''
        renders the environment, including obstacles and robots
        '''

        #clear canvas
        plt.clf()

        #renders all agents
        for agent in self.agents:
            agent.render()

        #render all obstacles
        #TODO make an obstacle class
        for o in self._obstlist:
            coor = o[0]
            rad = o[1]
            plt.gca().add_patch(plt.Circle((coor[0][0], coor[1][0]), rad, color = 'r'))

        #some useful display options
        region = self._region
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim([region[0][0], region[0][0] + region[1][0]])
        plt.ylim([region[0][1], region[0][1] + region[1][1]])

        #drawing everything
        plt.draw()
        plt.pause(0.02)
