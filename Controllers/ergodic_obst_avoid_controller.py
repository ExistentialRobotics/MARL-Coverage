import numpy as np
from ergodic_controller import ErgodicController


class ErgodicObstAvoidController(ErgodicController):
    """
    ErgodicObstAvoidController add obstacle avoidance functionality
    to the Ergodic Controller through a repulsive vector field.
    """
    def __init__(self, numrobot, xcoor, res, numbasis, dt, coverage_dist=None,  klis=None, umax=10):
        super().__init__(numrobot, xcoor, res, numbasis, dt, coverage_dist, klis, umax)


