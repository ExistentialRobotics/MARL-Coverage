import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import os

class Logger(object):
    """
    class to log videos, graphs, text files of runs to output directory
    """
    def __init__(self, experiment_name, make_video, dt=None):
        super().__init__()
        self._output_dir = "./experiments/{}/".format(experiment_name)
        self._dt = dt
        self._make_vid

        #checking if output directory exists and making it if it doesn't
        if os.path.isdir(self._output_dir):
            print("directory already exists, overwriting previous test")
        else:
            os.makedirs(self._output_dir)

        if make_video:
            self._writer = ani.FFMpegWriter(fps= int(1/dt))
            self._writer.setup(plt.gcf(), self._output_dir + experiment_name + ".mp4", dpi=100)

    def update(self):
        if(self._make_vid):
            #getting current simulation frame
            self._writer.grab_frame()

    def savefig(self, figure):
        pass

    def close(self):
        if self._make_vid:
            #saving video
            self._writer.finish()





