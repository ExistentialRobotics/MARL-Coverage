import numpy as np
import matplotlib.pyplot as plt
from . agent import Agent

class Swarm_Agent(Agent):
     def __init__(self, numrobot, controller, colorlis):
          super().__init__(controller)

          #list of robot positions
          self._numrobot = numrobot
          self._colorlis = colorlis
          self._xlis = []
          for i in range(numrobot):
               self._xlis.append(np.zeros((2,1)))

     def step(self, obstlist, dt):
          observation = (self._xlis, obstlist)
          controls = self._controller.getControls(observation)
          self.setControls(controls, dt)

     def setControls(self, ulis, dt):
          for i in range(len(ulis)):
               self._xlis[i] += ulis[i] * dt

     #used to set positions sampled in environment on reset
     def setPositions(self, xlis):
          self._xlis = xlis

     def sense(self):
          pass

     def reset(self):
          pass

     def render(self, size=50):
          for x, color in zip(self._xlis, self._colorlis):
               plt.scatter(x[0], x[1], s=size, c=color)
