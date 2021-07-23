import numpy as np
from . base_policy import Base_Policy

class Basic_Random(Base_Policy):

    def __init__(self, num_output, action_space):
        super().__init__(num_output, action_space)

    def step(self, state):
        return self.action_space.sample(s=self.num_output)

    def update(self):
        pass
