import numpy as np
import matplotlib.pyplot as plt
from . environment import Environment

class SuperGridRL(object):
    """
    A Multi-Agent Grid Environment with a discrete action space for RL testing.
    """
    def __init__(self, controller, numrobot, gridlen, gridwidth, sensesize=1, grid=None, seed=None):
        super().__init__()

        self._numrobot = numrobot
        self._gridlen = gridlen
        self._gridwidth = gridwidth
        self._controller = controller

        #sensing radius using chess metric(like how a king moves) -> "Chebyshev distance"
        self._sensesize = sensesize

        #generating robot positions
        self.reset()

        #blank/uniform grid by default
        if grid is None:
            self._grid = np.ones((gridwidth, gridlen))
        else:
            self._grid = grid

        #visited array
        self._visited = np.full((gridwidth, gridlen), True)

        # create seed if user specifies it
        if seed is not None:
            np.random.seed(seed)

    def step(self):
        #initialize reward for this step
        reward = 0

        #sense from all the current robot positions


        #calculate current observation


        #calculate controls from observation
        ulis = self._controller.getControls(observation)

        #update robot positions using controls
        for u,x in zip(ulis, self._xlis):
            if(u == 'l'):
                pass
            elif(u == 'r'):
                pass
            elif(u == 'u'):
                pass
            elif(u == 'd'):
                pass

        return reward

    def isOccupied(self):
        pass

    def reset(self):
        #generating random robot positions
        #TODO make sure none are spawned on obstacles/other robots
        self._xinds = np.random.randint(self._gridwidth, size=self._numrobot)
        self._yinds = np.random.randint(self._gridlen, size=self._numrobot)

    def render(self):
        pass



