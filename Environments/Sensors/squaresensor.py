import numpy as np
from . Sensor import Sensor

class SquareSensor(Sensor):
    '''
    Returns a square observation centered around current position based
    on given Chebyshev radius
    '''
    def __init__(self, sensor_config):
        super().__init__()
        self._radius = sensor_config['range']
    '''
    Returns segment of occupancy grid where robot can see
    '''
    def getMeasurement(self, x, y, oc, free_map, obst_map, pad):
        new_free = np.copy(free_map)
        new_obst = np.copy(obst_map)

        width = oc.shape[0]
        length = oc.shape[1]

        #right and left boundaries
        lb = max(x - self._radius, 0)
        rb = min(x + self._radius + 1, width)

        #up and down boundaries
        ub = min(y + self._radius + 1, length)
        db = max(y - self._radius, 0)

        #computing observation
        obs = oc[lb:rb, db:ub]

        #updating free/obst maps with observation
        new_free[(lb + pad):(rb + pad), (db + pad):(ub + pad)] = np.clip(obs, 0, 1)
        new_obst[(lb + pad):(rb + pad), (db + pad):(ub + pad)] = np.clip(-obs, 0, 1)

        return new_free, new_obst
