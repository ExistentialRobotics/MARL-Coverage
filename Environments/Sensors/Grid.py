import numpy as np
from . Sensor import Sensor

class SquareSensor(Sensor):
    '''
    Grid will be n x n square where n = range
    '''
    def __init__(self, range):
        super().__init__()
        self._range = range
        if (self._range % 2 == 1):
            self._range += 1
    '''
    Returns segment of occupancy grid where robot can see
    '''
    def getMeasurement(self, x, oc):
        radius = self._range / 2
        return oc[(x[0]-radius):(x[0]+radius), (x[1]-radius):(x[1]+radius)]
