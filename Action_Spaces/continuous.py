import numpy as np
from . base_space import Base_Space

class Continuous(Base_Space):
    """
    Class Continuous reprents the continuous action space of an agent/environment
    """

    def __init__(self, low, high):
        super().__init__()
        self.low = low
        self.high = high
        self.interval = np.array([low, high])

    def sample(self, s=None):
        return np.random.uniform(self.low, self.high, size=s)
