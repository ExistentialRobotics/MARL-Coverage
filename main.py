import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from Controllers.voronoi_controller import VoronoiController
from Controllers.grid_controller import GridController
from Controllers.ergodic_controller import ErgodicController
from Environments.environment import Environment
from Agents.swarm_agent import Swarm_Agent
import os

<<<<<<< HEAD

NUM_ARGS = 3
ADAPTIVE_COVERAGE = "adaptive_coverage"
ERGODIC_COVERAGE = "ergodic_coverage"
ALGOS = [ADAPTIVE_COVERAGE, ERGODIC_COVERAGE]
USAGE = "TODO: INSERT CORRECT USAGE STRING"
DASH = "-----------------------------------------------------------------------"


""" Parse command line args """
if len(sys.argv) != NUM_ARGS:
    print(DASH)
    print("Incorrect number of command line arguements.")
    print(USAGE)
    print(DASH)
    sys.exit(1)

# check if algorithm is valid
algo = sys.argv[1]
if algo not in ALGOS:
    print(DASH)
    print(str(algo) + " is not an available algorithm. Check the README for a list of algoritms.")
    print(DASH)
    sys.exit(1)

# check if config path is valid
config_path = sys.argv[2]
try:
    config_file = open(config_path)
except OSError:
    print(DASH)
    print(str(config_path) + " does not exit.")
    print(DASH)
    sys.exit(1)

# load json file
try:
    env_vars = json.load(config_file)
except:
    print(DASH)
    print(str(config_path) + " is an invalid json file.")
    print(DASH)
    sys.exit(1)

print(DASH)
print("Running algorithm " + str(algo) + " with config file " + str(config_path))
print(DASH)


""" Run algorithm(s) """
if algo == ADAPTIVE_COVERAGE:
    pass
elif algo == ERGODIC_COVERAGE:
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


    #logging parameters
    makevid = True
    testname = "ergodic_simple"
    output_dir = "./tests/{}/".format(testname)

    #checking if output directory exists and making it if it doesn't
    if os.path.isdir(output_dir):
        print("directory already exists, overwriting previous test")
    else:
        os.makedirs(output_dir)

    '''Making the Controller for the Swarm Agent'''
    # c = VoronoiController(numrobot, region, res, gain)
    # c = GridController(numrobot, region, res, gain)
    c = ErgodicController(numrobot, region, res, numbasis, dt, avoidobstacles=False)

    '''Making the Swarm Agent'''
    agents = [Swarm_Agent(numrobot, c, colorlist)]

    '''Making the Environment'''
    e = Environment(agents, numrobot, numobstacles, obstradius, region, dt, seed)

    '''Simulating the environment forward and rendering'''

    #creating video writer
    if(makevid):
        writer = ani.FFMpegWriter(fps= int(1/dt))
        writer.setup(plt.gcf(), output_dir + testname + ".mp4", dpi=100)

    e.reset()
    numsteps = 1000
    for i in range(numsteps):
        e.step()
        e.render()

        #code to log video of run
        if(makevid):
            writer.grab_frame()

    #saving video
    if(makevid):
        writer.finish()
=======
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


#logging parameters
makevid = True
testname = "ergodic_simple"
output_dir = "./tests/{}/".format(testname)

#checking if output directory exists and making it if it doesn't
if os.path.isdir(output_dir):
    print("directory already exists, overwriting previous test")
else:
    os.makedirs(output_dir)


'''Making the Controller for the Swarm Agent'''
# c = VoronoiController(numrobot, region, res, gain)
# c = GridController(numrobot, region, res, gain)
c = ErgodicController(numrobot, region, res, numbasis, dt, avoidobstacles=False)

'''Making the Swarm Agent'''
agents = [Swarm_Agent(numrobot, c, colorlist)]

'''Making the Environment'''
e = Environment(agents, numrobot, numobstacles, obstradius, region, dt, seed)

'''Simulating the environment forward and rendering'''

#creating video writer
if(makevid):
    writer = ani.FFMpegWriter(fps= int(1/dt))
    writer.setup(plt.gcf(), output_dir + testname + ".mp4", dpi=100)

e.reset()
numsteps = 1000
for i in range(numsteps):
    e.step()
    e.render()

    #code to log video of run
    if(makevid):
        writer.grab_frame()

#saving video
if(makevid):
    writer.finish()




>>>>>>> 1bc724648066e1679057b06e4c69229b9a069d93
