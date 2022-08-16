"""
logger.py contains the Logger class, which (unsurprisingly) logs information
about the current experiment.

Author: Peter Stratton
Email: pstratto@ucsd.edu, pstratt@umich.edu, peterstratton121@gmail.com
Author: Shreyas Arora
Email: sharora@ucsd.edu
"""
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import torch
import cv2

class Logger(object):
    """
    class Logger logs videos, graphs, text files of runs to output directory
    """
    def __init__(self, experiment_name, make_video):
        """
        Constructor for class Logger inits terminal and video writers and saves
        paths to the output directories

        Parameters:
            experiment_name - name of the experiement used to init the directory
                              where all the logs are saved
            make_video      - boolean that dictates whether or not to save a
                              video
        """
        super().__init__()
        self._output_dir = "./Experiments/grid_rl/{}/".format(experiment_name)
        self._make_vid = make_video

        #checking if output directory exists and making it if it doesn't
        if os.path.isdir(self._output_dir):
            print("directory already exists, overwriting previous test")
        else:
            os.makedirs(self._output_dir)

        if make_video:
            self._writer = cv2.VideoWriter(self._output_dir + experiment_name +\
                                           ".mp4",
                                           cv2.VideoWriter_fourcc(*'mp4v'), 30,
                                           (1075, 1075))

        self._timeseries = {}

        #checkpoint for model saving
        self._current_checkpoint = 1

        #creating terminal write for saving console output
        sys.stdout = TerminalWriter(self._output_dir + "TerminalOutput.txt")

    def addFrame(self, frame):
        """
        Adds a frame to the video

        Parameters:
            frame - image of the state of the environment
        """
        frame = np.uint8(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if(self._make_vid):
            self._writer.write(frame)

    def addTimeSeries(self, label, data):
        '''
        Adds timeseries data to logger by label

        Parameters:
            label - key in the timeseries dictionary
            data  - value in the timeseries
        '''
        if label not in self._timeseries:
            self._timeseries[label] = []
        self._timeseries[label].append(data)

    def savefig(self, figure, name):
        '''
        Saves a matplotlib figure to the experiment directory

        Parameters:
            figure - a matplotlib figure that has already been populated
            name   - name of the figure to be stored as
        '''
        figure.savefig(self._output_dir + name + ".png")

    def saveModelWeights(self, model):
        '''
        Saves a pytorch model state_dict

        Parameters:
            model - pytorch model to be saved in the experiment directory
        '''
        #checking if models directory exists
        if os.path.isdir(self._output_dir + 'models/'):
            print("past models exist, overwriting them")
        else:
            os.makedirs(self._output_dir + 'models/')

        torch.save(model.state_dict(), self._output_dir + 'models/checkpoint'
                   + str(self._current_checkpoint) + '.pt')
        self._current_checkpoint += 1

    def plot(self, list, fignum, title, xlabel, ylabel, linelabel, figname,
            show_fig):
        """
        Makes a matplotlib plot

        Parameters:
            list      - python list of data to be plotted
            fignum    - figure number
            title     - title of plot
            xlabel    - label of x data
            ylabel    - label of y data
            linelabel - label of line plotted
            figname   - name of the plot that is saved to experiment directory
            show_fig  - boolean dictating whether or not to show the figure
        """
        plt.figure(fignum)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        line, = plt.plot(list, label=linelabel)
        plt.legend(handles=[line])
        self.savefig(plt.gcf(), figname)
        if show_fig:
            plt.show()

    def close(self):
        '''
        Handling all file closing and writing
        '''
        if self._make_vid:
            #saving video
            self._writer.release()

        # sys.stdout.log.close()
        #check if there is any timeseries data and save to text file
        #TODO


#adopted from some stackoverflow answer
#https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
class TerminalWriter(object):
    """
    class TerminalWriter writes information to the terminal
    """
    def __init__(self, filename):
        """
        Constructor for class TerminalWriter

        Parameters:
            filename - name of the logfile
        """
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        """
        Writes a string to both the terminal and logfile

        Parameters:
            message - string to be written
        """
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """
        Should flush something
        """
        pass
