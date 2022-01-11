import numpy as np
from . Sensor import Sensor
import cv2

class LidarSensor(Sensor):
    def __init__(self, sensor_config):
        Sensor.__init__(self)
        self._num_lasers = sensor_config["num_lasers"]
        self._max_range = sensor_config["range"]

        assert self._num_lasers % 2 == 1, "odd number of lasers needed"

        #assuming lasers are equally spaced
        self._thetalist = np.linspace(0, 2*np.pi, num=self._num_lasers, endpoint=False)

    def getMeasurement(self, x, y, oc, free_map, obst_map, pad):
        """Returns a lidar scan from position x w.r.t to occupancy grid oc

        Args:
           x : state where x[0:3] = (x,y,theta) of robot
           oc : occupancy grid to perform sense operation on

        Returns:
            2d numpy array representing scan, with size 2*range + 1 in each axis,
        egocentric observation from state
        """
        gridwidth = np.shape(oc)[0]
        gridlen = np.shape(oc)[1]

        new_free = np.copy(free_map)
        new_obst = np.copy(obst_map)

        #loop through all lines starting at coordinate
        for theta in self._thetalist:
            currx = x
            curry = y

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
            while self.inbounds(currx, curry, gridlen, gridwidth) and oc[int(currx), int(curry)] >= 0 and currdist < self._max_range:
                new_free[int(currx) + pad, int(curry) + pad] = 1
                currx += xinc
                curry += yinc
                currdist += distinc

            #writing the final value of the scan
            final = -1
            if self.inbounds(currx, curry, gridlen, gridwidth) and oc[int(currx), int(curry)] >= 0:
                new_free[int(currx) + pad, int(curry) + pad] = 1
            else:
                new_obst[int(currx) + pad, int(curry) + pad] = 1

        return new_free, new_obst

    def inbounds(self, x, y, gridlen, gridwidth):
        return x >= 0 and y >= 0 and x < gridwidth and y<gridlen


