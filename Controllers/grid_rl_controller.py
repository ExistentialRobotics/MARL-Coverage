import numpy as np
from . controller import Controller

class GridRLController(Controller):
    def __init__(self, numrobot, policy):
        super().__init__(numrobot, policy)
        self._best_policy = None

    def getControls(self, observation, testing=False):
        if testing:
            return self._best_policy.step(observation)
        else:
            return self._policy.step(observation)

    def update_policy(self, episode):
        # reset gradients to zero
        self._policy.optimizer.zero_grad()

        for i in range(len(episode)):
            state = episode[i][0]
            action = episode[i][1]

            # sum return
            r_return = 0
            for j in range(i, len(episode)):
                r_return += episode[i][2]

            # increment gradient
            self._policy.calc_gradient(state, action, r_return)

        # update parameters
        self._policy.optimizer.step()

    def save_policy(self):
        self._best_policy = self._policy
