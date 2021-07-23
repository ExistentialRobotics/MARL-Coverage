import numpy as np
from . controller import Controller

class GridRLRandomController(Controller):
    def __init__(self, numrobot, policy):
        super().__init__(numrobot, policy)

    def getControls(self, observation):
        # #TODO: decomposing observation
        return self._policy.step(observation)
