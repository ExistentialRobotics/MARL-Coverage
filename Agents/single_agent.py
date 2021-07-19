import numpy as np
import matplotlib.pyplot as plt
from . agent import Agent

class Single_Agent(Agent):
    """
    Abstract class Agent represents a single autonomous entity that preforms
    actions in an environment.
    """
    def __init__(self, controller, color='r'):
        super().__init__(controller)
        # agent position
        self.pos = np.array([[0], [0]], dtype='f')

        # agent color for rendering
        self.color = color

    def setControls(self, u, dt):
        self.pos += u*dt

    def setPos(self, pos):
        self.pos = pos

    def sense(self):
        pass

    def reset(self):
        pass

    def render(self, size=50):
        plt.scatter(self.pos[0], self.pos[1], s=size, c=self.color)
