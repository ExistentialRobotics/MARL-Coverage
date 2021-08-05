import numpy as np
from . controller import Controller
from copy import deepcopy

class GridRLController(Controller):
    def __init__(self, numrobot, policy):
        super().__init__(numrobot, policy)

    def getControls(self, observation, testing=False):
        return self._policy.step(observation)

    def update_policy(self, episode):
        self._policy.update_policy(episode)

    def set_train(self):
        self._policy.set_train()

    def set_eval(self):
        self._policy.set_eval()

    def print_weights(self):
        self._policy.print_weights()
