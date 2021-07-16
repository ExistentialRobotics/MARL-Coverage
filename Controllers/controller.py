import numpy as np

class Controller(object):
    """
    Abstract Controller class provides framework for making a controller that
    controls a set of agents in the multi-agent sensing environment.
    """
    #TODO decide whether controller should control one agent, or group
    #one agent is more realistic, but then we have to include extra information
    #in the observation, multiple agents is much easier to code but less realistic

    def __init__(self, numrobot):
        self._numrobot = numrobot

    def getControls(self, observation):
        """
        should return a list of controls for each agent
        """
        raise NotImplementedError()
