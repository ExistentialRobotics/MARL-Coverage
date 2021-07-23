import numpy as np
from . controller import Controller

class GridRLRandomController(Controller):
    def __init__(self, numrobot):
        super().__init__(numrobot)
        self._controls = ['l', 'r', 'u', 'd']


    def getControls(self, observation):
        #TODO: decomposing observation
        ulis = []
        for i in range(self._numrobot):
            ulis.append(self._controls[np.random.randint(4)])
        return ulis



