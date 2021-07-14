import numpy as np
import matplotlib.pyplot as plt
from voronoi_controller import VoronoiController
from grid_controller import GridController
from ergodic_controller import ErgodicController

#environment parameters
numsteps = 1000
numrobot = 9
qcoor = np.array([[0,0], [8,8]],dtype=float) #defines rectangular region
res = (40,40) #resolution tells us how many regions to divide each axis into
gain = 10
dt = 0.02

#setting the random seed so that this code is deterministic
# np.random.seed(420)

#generating random points in the rectangle where the robots will start
qlis = []
for i in range(numrobot):
    xcoor = qcoor[1][0] * np.random.random_sample() + qcoor[0][0]
    ycoor = qcoor[1][1] * np.random.random_sample() + qcoor[0][1]
    qlis.append(np.array([[xcoor], [ycoor]]))

#obstacle info, assuming circular obstacles for now
numobstacles = 5
obstradius = 0.5

#generating the obstacle positions
obstlist = []
for i in range(numobstacles):
    xcoor = qcoor[1][0] * np.random.random_sample() + qcoor[0][0]
    ycoor = qcoor[1][1] * np.random.random_sample() + qcoor[0][1]
    obstlist.append(plt.Circle((xcoor, ycoor), obstradius, color = 'r'))

#making the controller
# c = VoronoiController(qlis, qcoor, res, gain)
# c = GridController(qlis, qcoor, res, gain)
c = ErgodicController(numrobot, qcoor, res, 50, dt)


#simulation loop and graphing
graphcolors = np.random.rand(numrobot)
for i in range(numsteps):
    #getting controls
    currcontrols = c.getControls(qlis)

    #integrating controls forward
    for j in range(numrobot):
        qlis[j] += currcontrols[j]*dt

    #graphing current robot positions
    plt.clf()
    currpos = qlis
    currpos = np.array(currpos).reshape(numrobot, 2)
    currpos = np.transpose(currpos)
    plt.scatter(currpos[0], currpos[1],c=graphcolors, alpha=0.5)

    #graphing obstacles
    # for obstacle in obstlist:
    #     plt.gca().add_patch(obstacle)
    plt.xlim([qcoor[0][0], qcoor[0][0] + qcoor[1][0]])
    plt.ylim([qcoor[0][1], qcoor[0][1] + qcoor[1][1]])
    plt.draw()
    plt.pause(0.02)
