import numpy as np
import matploblib.pyplot as plt

class Logger(object):
    """
    class to log videos, graphs, text files of runs to output directory
    """
    def __init__(self, experiment_name):
        super().__init__()
        self._output_dir = "./experiments/{}/".format(experiment_name)

