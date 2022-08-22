import numpy as np
import matplotlib.pyplot as plt

class Agent(object):
    """
    Interface Agent represents the autonomous entities that perform actions
    in an environment.
    """

    def __init__(self, controller):
        super().__init__()
        self._controller = controller

    def step(self):
        raise NotImplementedError()

    def setControls(self, u, dt):
        raise NotImplementedError()

    def sense(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def render(self):
        raise NotImplementedError()
