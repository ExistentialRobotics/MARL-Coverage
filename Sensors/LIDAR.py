import numpy as np
from . SensAbstract import SensAbstract
import cv2

class LidarSensor(SensAbstract):
    def __init__(self, numofLasers, maxrange):
        SensAbstract.__init__(self)
        assert numofLasers % 2 == 1, "odd number of lasers needed"
        self._numofLasers = numofLasers
        self._range = maxrange

        #assuming lasers are equally spaced
        self._thetalist = np.linspace(0, 2*np.pi, num=numofLasers, endpoint=False)

    def getMeasurement(self, x, oc):
        """Returns a lidar scan from position x w.r.t to occupancy grid oc

        Args:
           self arg1
           x : state where x[0:3] = (x,y,theta) of robot
           oc : occupancy grid to perform sense operation on

        Returns:
            2d numpy array representing scan, with size 2*range + 1 in each axis,
        egocentric observation from state
        """
        # For grid world robot is always facing forward
        x[2] = 0

        laserscan = np.zeros((2*self._range + 1, 2*self._range + 1))

        xround = int(x[0])
        yround = int(x[1])
        #loop through all lines starting at coordinate
        for theta in self._thetalist:
            currx = xround
            curry = yround

            #finding each increment
            xinc = np.cos(theta)
            yinc = np.sin(theta)

            #normalizing increment so one of them is 1
            larger = max(abs(xinc), abs(yinc))
            xinc /= larger
            yinc /= larger

            #tracking distance of beam
            distinc = np.sqrt(xinc**2 + yinc**2)
            currdist = 0

            #main section of lidar scan
            while oc.inBounds(currx, curry) and oc.isFree(currx, curry) and currdist < self._range:
                laserscan[int(currx - xround + self._range), int(curry - yround + self._range)] = 1
                currx += xinc
                curry += yinc
                currdist += distinc

            #writing the final value of the scan
            final = -1
            if oc.inBounds(currx, curry) and oc.isFree(currx, curry):
                final = 1
            laserscan[int(currx - xround + self._range), int(curry - yround + self._range)] = final

        return laserscan

    def getMeasurementProbability(self, xt, ls, oc):
        '''
        In this case we return a similarity score instead of a probability.
        '''
        score = 0

        #rounding coordinates
        x = int(xt[0])
        y = int(xt[1])
        maxrange = self._range

        #checking if position is within bounds and not occupied
        if oc.inBounds(x, y) and oc.isFree(x, y):
            #resizing occupancy grid so it can be directly scored against scan
            ocmap = cv2.resize(oc._oc, (0,0), fx=oc._celldim, fy=oc._celldim,
                            interpolation=cv2.INTER_NEAREST)
            width = ocmap.shape[0]
            length = ocmap.shape[0]


            #bounds for image
            lb = max(x - maxrange, 0)
            rb = min(x + maxrange + 1, width)
            db = max(y - maxrange, 0)
            ub = min(y + maxrange + 1, length)

            #bounds for scan
            lpad = max(0, maxrange-x)
            rpad = 2*maxrange + 1 - max(0, x + maxrange + 1 - width)
            dpad = max(0, maxrange-y)
            upad = 2*maxrange + 1 - max(0, y + maxrange + 1 - length)

            #finding difference between scan and map, so we can create a score
            diff = ocmap[lb:rb, db:ub] - ls[lpad:rpad, dpad:upad]

            #number of cells where both claim empty
            score += np.count_nonzero(diff<0)

            #number of cells where both claim occupied
            score += np.count_nonzero(diff == 2)

        return 0.2*score

