import numpy as np
import math

class Sensor(object):
    def __init__(self):
        super().__init__()

    def getMeasurement(self, x, y, oc, free_map, obst_map, pad):
        raise NotImplementedError()
