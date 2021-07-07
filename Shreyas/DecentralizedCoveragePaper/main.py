from controller import Controller
from linearbasis import GaussianBasis
import numpy as np
import matplotlib.pyplot as plt

#environment parameters
numsteps = 1000
numrobot = 9
qcoor = np.array([[0,0], [8,8]],dtype=float) #defines rectangular region

#setting the random seed so that this code is deterministic
np.random.seed(420)

#generating random points in the rectangle where the robots will start
qlis = []
for i in range(numrobot):
    xcoor = qcoor[1][0] * np.random.random_sample() + qcoor[0][0]
    ycoor = qcoor[1][1] * np.random.random_sample() + qcoor[0][1]
    qlis.append(np.array([[xcoor], [ycoor]]))
# qlis = np.array(qlis)

#defining the basis functions and parameters of sensing function
mulis = [
    [6, 2],
    [2, 6]
]
sigmalis = [
    [[0.5, 0], [0, 0.5]], [[0.5, 0], [0, 0.5]]
    ]


truea = np.array([[100], [100]])
amin = np.array([[0.1], [0.1]]) 
truephi = GaussianBasis(mulis, sigmalis)
truephi.updateparam(truea)

#defining the controller to drive robots to locally optimal configuration
res = (10,10) #resolution tells us how many regions to divide each axis into
gamma = 1e2 #learning rate
c = Controller(qlis, truephi, qcoor, res, mulis, sigmalis, amin, gamma, True, 1)
# c = Controller(qlis, truephi, qcoor, res, mulis, sigmalis, amin, gamma)

#lists for tracking distance between robot parameters and true parameters
adislist = []
for i in range(numrobot):
    adislist.append([])

vargraph = []


currpos = c.step(0.02)

#main simulation loop
graphcolors = np.random.rand(numrobot)
for i in range(numsteps):
    #forward step
    currpos = c.step(0.02)

    #graphing current robot positions
    plt.clf()
    currpos = np.array(currpos).reshape(numrobot, 2)
    currpos = np.transpose(currpos)
    plt.scatter(currpos[0], currpos[1],c=graphcolors, alpha=0.5)
    plt.draw()
    plt.pause(0.02)

    varlist = []
    #adding parameter distances to list
    for j in range(numrobot):
        dist = np.linalg.norm(c._phihatlist[j].getparam() - truephi.getparam())
        adislist[j].append(dist)
        varlist.append(c._phihatlist[j].getparam())
        print(varlist[j])
    #computing and storing variance of parameters
    varlist = np.array(varlist)
    m = np.mean(varlist, axis=0)
    var = 0
    for j in range(numrobot):
        var += np.transpose(c._phihatlist[j].getparam() - m) @ (c._phihatlist[j].getparam() - m)
    var = var/numrobot
    vargraph.append(var[0][0])

plt.clf()

#graphing variance of robot parameters with respect to time
plt.plot(np.array(vargraph))
plt.show()

#graphing parameter distances with respect to time
for i in range(numrobot):
    plt.plot(np.array(adislist[i]))
    plt.show()

for i in range(numrobot):
    #checking which Lambdas are positive definite
    li = c._Lambda[i]
    print(li)
    if(np.all(np.linalg.eigvals(li) > 0)):
        print("good, Lambda is positive definite")
    else:
        print("bad, Lambda is not positive definite")

