import numpy as np
from . base_space import Base_Space

class Discrete(Base_Space):
    """
    Class Discrete represnets the discrete action space of an agent/environment
    """

    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions

    def sample(self, s=None):
        return np.random.randint(self.num_actions, size=s)
