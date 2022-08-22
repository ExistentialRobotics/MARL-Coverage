import numpy as np
import matplotlib.pyplot as plt
from . swarm_agent import Swarm_Agent

class AC_Swarm_Agent(Swarm_Agent):
     def __init__(self, numrobot, controller, colorlis, min_a, num_basis_fx):
          super().__init__(numrobot, controller, colorlis)
          self.a_est = np.full((numrobot, num_basis_fx), min_a)


     def step(self, dists, inds, dt):
          observation = (self._xlis, dists, inds, self.a_est)
          controls, a_est, est_mean, true_mean, a_mean = self._controller.getControls(observation)
          super().setControls(controls, dt)
          self.a_est = a_est
          return est_mean, true_mean, a_mean, a_est

     #used to set positions sampled in environment on reset
     def setPositions(self, xlis):
          super().setPositions(xlis)

     def sense(self):
          pass

     def reset(self):
          pass

     def render(self, size=25):
          super().render(size=size)
