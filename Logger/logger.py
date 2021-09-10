import numpy as np
import matplotlib.pyplot as plt
import os, sys
import torch
import cv2

class Logger(object):
    """
    class to log videos, graphs, text files of runs to output directory
    """
    def __init__(self, experiment_name, make_video):
        super().__init__()
        self._output_dir = "./Experiments/grid_rl/{}/".format(experiment_name)
        self._make_vid = make_video

        #checking if output directory exists and making it if it doesn't
        if os.path.isdir(self._output_dir):
            print("directory already exists, overwriting previous test")
        else:
            os.makedirs(self._output_dir)

        if make_video:
            self._writer = cv2.VideoWriter(self._output_dir + experiment_name + ".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (1075, 1075))

        self._timeseries = {}

        #checkpoint for model saving
        self._current_checkpoint = 1

        #creating terminal write for saving console output
        sys.stdout = TerminalWriter(self._output_dir + "TerminalOutput.txt")

    def addFrame(self, frame):
        frame = np.uint8(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if(self._make_vid):
            self._writer.write(frame)

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

    def plot(self, list, fignum, title, xlabel, ylabel, linelabel, figname, show_fig):
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
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

