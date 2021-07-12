import numpy as np
import matplotlib.pyplot as plt

class Agent(object):
    """
    Interface Agent represents the autonomous entities that preform actions
    in an environment.
    """

    def __init__(self):
        super().__init__()

    def odom_command(self):
        raise NotImplementedError()

    def sense(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def render(self):
        raise NotImplementedError()
