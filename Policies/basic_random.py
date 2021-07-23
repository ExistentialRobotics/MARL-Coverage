import numpy as np
from . base_policy import Base_Policy

class Basic_Random(Base_Policy):

    def __init__(self, num_output, action_space):
        super().__init__(num_output, action_space)

    def step(self, state):
        ulis = []
        for i in range(self.num_output):
            ulis.append(self.action_space[np.random.randint(4)])
        return ulis

    def update(self):
        pass
