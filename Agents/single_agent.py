import numpy as np
import matplotlib.pyplot as plt
from . agent import Agent

class Single_Agent(Agent):
    """
    Abstract class Agent represents a single autonomous entity that preforms
    actions in an environment.
    """
    def __init__(self, init_x, init_y, color='r'):
        super().__init__()
        # agent position
        self.pos = np.array([[init_x], [init_y]], dtype='f')
        self.start_pos = self.pos

        # agent color for rendering
        self.color = color

    def odom_command(self, u):
        self.pos[0] += u[0]
        self.pos[1] += u[1]

    def sense(self):
        raise NotImplementedError()

    def reset(self):
        self.pos = self.start_pos

    def render(self, size=50):
        plt.scatter(self.pos[0], self.pos[1], s=size, c=self.color)
