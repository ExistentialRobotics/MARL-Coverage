import torch
import numpy as np

class Base_Policy(object):
    """
    Base class for policies.
    """

    def __init__(self):
        """
        Constructor for class Base Policy.
        """
        super().__init__()

    def pi(self, state):
        """
        Generates an action from the current state.
        """
        raise notImplementedError()

    def update_policy(self):
        """
        Updates the policy Parameters.
        """
        raise notImplementedError()

    def reset(self):
        """
        Resets the policy (at the end of an episode)
        """
        raise notImplementedError()
