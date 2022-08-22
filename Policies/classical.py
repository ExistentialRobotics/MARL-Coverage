import numpy as np
import random
import matplotlib.pyplot as plt

gridDim = 20

#Generate grid - 0 means unexplored and 1 means obstacle or explored
w, h = gridDim, gridDim
grid = np.zeros((w,h))
print(grid)

#generate random robot location then set for debugging
# robotPos = [random.randint(0,gridDim-1), random.randint(0,gridDim - 1)]
robotPos = np.array([0,2])
#store to check if tree is completed
startPos = robotPos
#Will store all connections for the tree
segments = []

#Main recursive function
def STC(w,x,t,gri):
    #Set current location of robot as explored
    gri[x[0]][x[1]]=1
    #rotate parent point ccw about x and store result in y
    y = rotate(w, x)

    #Loop while available neighbors
    while (availableNeighbors(x,gri)):
        print("y" + str(y))
        print("x" + str(x))

        #Check if start node has been reached
        #if (startPos[0] == x[0] and startPos[1] == x[1] and t > 0):
        #    break
        #Only used to make sure the above if statement doesn't quit on first iteration
        t = t + 1

        #if unexplored node is identified
        if gri[y[0]][y[1]] == 0 and y[0] >=0:
            print("transition")
            #Store the segment
            segments.append(Segment(x[0],x[1],y[0],y[1]))
            #Next iteration
            plt.plot([x[0], y[0]], [x[1], y[1]], color = (0, 0, 0))
            plt.show
            plt.pause(.1)
            STC(x, y, t, gri)
        else:
            ###### Fix #######
            #rotate y ccw about x and store in y
            y = rotate(y, x)

#Check if node has available neighbor nodes
def availableNeighbors(robo, gri):
    #if (gri[robo[0] - 1][robo[1]] == 1 and gri[robo[0] + 1][robo[1]] == 1 and gri[robo[0]][robo[1] - 1] == 1 and gri[robo[0]][robo[1] + 1] == 1):
    #    return False
    return True

#rotates orig about cent ccw 90 degrees
def rotate(x, y):
    # if (orig[0] < cent[0] and orig[1] == cent[1] and cent[1] <= (gridDi - 1)):
    #     return [cent[0], cent[1] + 1]
    # elif (orig[0] == cent[0] and orig[1] > cent[1] and cent[0] <= (gridDi - 1)):
    #     return [cent[0] + 1, cent[1]]
    # elif (orig[0] > cent[0] and orig[1] == cent[1] and cent[1] >= 0):
    #         return [cent[0], cent[1] - 1]
    # elif (orig[0] == cent[0] and orig[1] < cent[1] and cent[0] >= 0):
    #         return [cent[0] - 1, cent[1]]
    # else:
    #     print("Error")
    #     return [-1, -1]
    print(x)
    print(y)
    pass


#used to store segments
class Segment():
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def getP1(self):
        return [x1, y1]

    def getP2(self):
        return [x2, y2]

#Initiate loop with center of the robot position and a parent node of one above the robot
STC([0, 1], robotPos,0,grid)
