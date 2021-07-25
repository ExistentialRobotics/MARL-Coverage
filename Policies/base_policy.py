import torch
import numpy as np

class Base_Policy(object):
    """
    Base class for policies.
    """

    def __init__(self, numrobot, action_space):
        """
        Constructor for class Base Policy.

        Parameters
        ----------
        num_output   - number of actions to output when stepping the policy
        action_space - list of available actions for the agent to take
        """

        super().__init__()
        self.numrobot     = numrobot
        self.action_space = action_space

    def step(self, state):
        """
        Generates an action from the current state.
        """
        raise notImplementedError()

    def update(self):
        """
        Updates the policy Parameters.
        """
        raise notImplementedError()
