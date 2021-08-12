import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import os, sys
import torch

class Logger(object):
    """
    class to log videos, graphs, text files of runs to output directory
    """
    def __init__(self, experiment_name, make_video, dt=None):
        super().__init__()
        self._output_dir = "./Experiments/grid_rl/{}/".format(experiment_name)
        self._dt = dt
        self._make_vid = make_video

        #checking if output directory exists and making it if it doesn't
        if os.path.isdir(self._output_dir):
            print("directory already exists, overwriting previous test")
        else:
            os.makedirs(self._output_dir)

        if make_video:
            self._writer = ani.FFMpegWriter(fps= int(1/dt))
            self._writer.setup(plt.gcf(), self._output_dir + experiment_name + ".mp4", dpi=100)

        self._timeseries = {}

        #checkpoint for model saving
        self._current_checkpoint = 1
        self._terminaloutput = None

    def update(self):
        if(self._make_vid):
            #getting current simulation frame
            self._writer.grab_frame()

    def addTimeSeries(self, label, data):
        '''
        Adds timeseries data to logger by label
        '''
        if label not in self._timeseries:
            self._timeseries[label] = []
        self._timeseries[label].append(data)

    def savefig(self, figure, name):
        '''
        figure should be a matplotlib figure that has already been populated,
        name is the what it will be stored as.
        '''
        figure.savefig(self._output_dir + name + ".png")

    def saveModelWeights(self, model):
        '''
        save a pytorch model state_dict
        '''

        #checking if models directory exists
        if os.path.isdir(self._output_dir + 'models/'):
            print("past models exist, overwriting them")
        else:
            os.makedirs(self._output_dir + 'models/')

        torch.save(model.state_dict(), self._output_dir + 'models/checkpoint'
                   + str(self._current_checkpoint) + '.pt')
        self._current_checkpoint += 1

    def saveTerminalOutput(self):
        sys.stdout = TerminalWriter("TerminalOutput.txt")

    def close(self):
        '''
        Handling all file closing and writing
        '''
        if self._make_vid:
            #saving video
            self._writer.finish()
        #check if there is any timeseries data and save to text file
        #TODO


#adopted from some stackoverflow answer
#https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
class TerminalWriter(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
