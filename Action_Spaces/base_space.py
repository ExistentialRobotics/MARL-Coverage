import numpy as np

class Base_Space(object):
    """
    Class Base_Space is an interface for different types of action spaces.
    """

    def __init__(self):
        super().__init__()

    def sample(self):
        raise notImplementedError()
