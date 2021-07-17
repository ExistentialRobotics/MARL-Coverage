import numpy as np
import matplotlib.pyplot as plt
from voronoi_controller import VoronoiController
from grid_controller import GridController
from ergodic_controller import ErgodicController

#environment parameters
numsteps = 1000
numrobot = 6
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
numobstacles = 3
obstradius = 0.75

#generating the obstacle positions
olist = []
for i in range(numobstacles):
    xcoor = qcoor[1][0] * np.random.random_sample() + qcoor[0][0]
    ycoor = qcoor[1][1] * np.random.random_sample() + qcoor[0][1]
    olist.append((np.array([[xcoor], [ycoor]]), obstradius))

#making the controller
# c = VoronoiController(qlis, qcoor, res, gain)
# c = GridController(qlis, qcoor, res, gain)
c = ErgodicController(numrobot, qcoor, res, 50, dt, avoidobstacles=True)


#tracking phi (metric of ergodicity)
philis = []

#simulation loop and graphing
graphcolors = np.random.rand(numrobot)
for i in range(numsteps):
    #getting controls
    currcontrols, phi = c.getControls((qlis, olist))

    philis.append(phi)

    #integrating controls forward
    for j in range(numrobot):
        qlis[j] += currcontrols[j]*dt
        # qlis[j] = np.clip(qlis[j],
        #                   np.array([[qcoor[0][0]], [qcoor[0][1]]]),
        #                   np.array([[qcoor[0][0] + qcoor[1][0]],
        #                   [qcoor[0][1] + qcoor[1][1]]]))

    #graphing current robot positions
    plt.clf()
    currpos = qlis
    currpos = np.array(currpos).reshape(numrobot, 2)
    currpos = np.transpose(currpos)
    plt.scatter(currpos[0], currpos[1],c=graphcolors, alpha=0.5)

    #remaking obstlist
    obstlist = []
    for o in olist:
        coor = o[0]
        rad = o[1]
        obstlist.append(plt.Circle((coor[0][0], coor[1][0]), rad, color = 'r'))
    #graphing obstacles
    for obstacle in obstlist:
        plt.gca().add_patch(obstacle)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim([qcoor[0][0] - 2, qcoor[0][0] + qcoor[1][0] + 2])
    plt.ylim([qcoor[0][1] - 2, qcoor[0][1] + qcoor[1][1] + 2])
    plt.draw()
    plt.pause(0.02)

#plotting metric of ergodicity over time
plt.clf()
plt.plot(np.array(philis))
plt.show()

