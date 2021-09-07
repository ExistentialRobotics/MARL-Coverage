import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

def gridload(grid_config):
    '''
    Loads all the images from a given directory, converts them to black and
    white, the converts them into a list of numpy arrays, where black is -1,
    and white is 1.
    '''
    grid_dir = grid_config['grid_dir']
    numgrids = grid_config['numgrids']

    gridlis = []
    i = 0
    for fname in os.listdir(grid_dir):
        if i < numgrids:
            image = np.array(Image.open(os.path.join(grid_dir, fname)))
            image = image.astype(float)
            image = np.clip((image - 1), -1, 1)
            gridlis.append(image)
        i += 1
    return gridlis

def gridgen(grid_config):
    '''
    Creates grids of the given dimensions (gridwidth x gridlen), and uses a
    bernoulli random variable to determine if there is an obstacle at a
    given square according to probability prob_obst.
    '''
    prob_obst = grid_config['prob_obst']
    gridwidth = grid_config['gridwidth']
    gridlen   = grid_config['gridlen']
    numgrids  = grid_config['numgrids']

    gridlis = []
    for i in range(numgrids):
        gridlis.append(np.random.choice(a=[1.0,-1.0], size=(gridwidth, gridlen),
                                        p=[1-prob_obst, prob_obst]))
    return gridlis
