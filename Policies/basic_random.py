import numpy as np
from . base_policy import Base_Policy

class Basic_Random(Base_Policy):

    def __init__(self, actions_space):
        super().__init__()

    def pi(self, state):
        return self.action_space.sample()

    def update(self):
        pass
