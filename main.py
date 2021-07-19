import numpy as np
import matplotlib.pyplot as plt
from Controllers.voronoi_controller import VoronoiController
from Controllers.grid_controller import GridController
from Controllers.ergodic_controller import ErgodicController
from Environments.environment import Environment
from Agents.swarm_agent import Swarm_Agent

'''Environment Parameters'''
numrobot = 6
region = np.array([[0,0], [8,8]],dtype=float) #defines rectangular region
dt = 0.02
seed = 420
numobstacles = 3
obstradius = 0.75

#controller parameters
gain = 10
res = (40,40) #resolution tells us how many regions to divide each axis into
numbasis = 50
colorlist = list(np.random.rand(numrobot))

'''Making the Controller for the Swarm Agent'''
# c = VoronoiController(numrobot, region, res, gain)
# c = GridController(numrobot, region, res, gain)
c = ErgodicController(numrobot, region, res, 50, dt, avoidobstacles=True)

'''Making the Swarm Agent'''
agents = [Swarm_Agent(numrobot, c, colorlist)]

'''Making the Environment'''
e = Environment(agents, numrobot, numobstacles, obstradius, region, dt, seed)

'''Simulating the environment forward and rendering'''
e.reset()
numsteps = 1000
for i in range(numsteps):
    e.step()
    e.render()
