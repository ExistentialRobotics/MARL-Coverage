import numpy as np
import math

class SensAbstract(object):
    def __init__(self):
        super().__init__()
    def getMeasurement(self, x, oc):
        raise NotImplementedError()
    def getMeasurementProbability(self, x, z, oc):
        raise NotImplementedError()