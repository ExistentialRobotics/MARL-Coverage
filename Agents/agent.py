import numpy as np
import matplotlib.pyplot as plt

class Agent(object):
    """
    Class Agent represents the autonomous entities that preform actions in an
    environment.
    """

    def __init__(self, init_x, init_y, color='r'):
        super().__init__()
        # agent position
        self.pos = np.array([[init_x], [init_y]], dtype='f')

        # agent color for rendering
        self.color = color

    def odom_command(self):
        self.pos[0] += u[0]
        self.pos[1] += u[1]

    def sense(self):
        pass

    def reset(self):
        pass

    def render_self(self, size=50):
        plt.scatter(self.pos[0], self.pos[1], s=size, c=self.color)
