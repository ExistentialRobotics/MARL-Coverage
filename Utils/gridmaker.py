import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

def gridload(grid_config=None):
    '''
    Loads all the images from a given directory, converts them to black and
    white, the converts them into a list of numpy arrays, where black is -1,
    and white is 1 and if no config is provided, test grids are manually created

    Parameters:
        grid_config - config file defining the parameters of grids to create

    Return:
        train_set - list of grids to use for training
        test_set  - list of grids to use for testing
    '''
    train_set = None
    test_set = None
    if grid_config is None:
        test1 = np.ones((15, 15))
        test1[11,3:8] = -1.0
        test1[7,7:13] = -1.0
        test1[2:7,3] = -1.0
        test1[2,3:8] = -1.0

        test2 = np.ones((15, 15))
        test2[10:13,3] = -1.0
        test2[12,4:8] = -1.0
        test2[10:13,8] = -1.0
        test2[2:5,6] = -1.0
        test2[2,7:11] = -1.0
        test2[2:5,11] = -1.0

        test3 = np.ones((15,15))
        test3[5:10,2] = -1.0
        test3[5:10,5] = -1.0
        test3[5:10,9] = -1.0
        test3[5,10:15] = -1.0
        test_set = [test1, test2, test3]

        train1 = np.ones((15,15))
        train1[2,3] = -1.0
        train1[3,7] = -1.0
        train1[1,10:14] = -1.0
        train1[7,5] = -1.0
        train1[8,10] = -1.0
        train1[7:13,2] = -1.0
        train1[10,6:9] = -1.0
        train1[10:14,8] = -1.0

        train2 = np.ones((15,15))
        train2[3:13,7] = -1.0
        train2[7,2:13] = -1.0

        train3 = np.ones((15,15))
        train3[2,10:13] = -1.0
        train3[2:6,12] = -1.0
        train3[4:11,6] = -1.0
        train3[7,6:8] = -1.0
        train3[10,2] = -1.0

        train4 = np.ones((15,15))
        train4[4:10,10] = -1.0
        train4[4:10,5] = -1.0
        train4[9,5:11]= -1.0

        train5 = np.ones((15,15))
        train5[3,8:12] = -1.0
        train5[6,4:8] = -1.0
        train5[9,1:4] = -1.0
        train5[9:12,11] = -1.0

        train6 = np.ones((15,15))
        train6[3:12,11] = -1.0
        train6[3:12,7] = -1.0
        train_set = [train1, train2, train3, train4, train5, train6]
    else:
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

        # split into train and test sets
        l = len(gridlis)
        if l == 1:
            train_set = gridlis
            test_set = gridlis
        else:
            train_set = gridlis[:l]
            test_set = gridlis[l:]

    return train_set, test_set


def gridgen(grid_config):
    '''
    Creates grids of the given dimensions (gridwidth x gridlen), and uses a
    bernoulli random variable to determine if there is an obstacle at a
    given square according to probability prob_obst.

    Parameters:
        grid_config - config file defining the parameters of grids to create

    Return:
        train_set - list of grids to use for training
        test_set  - list of grids to use for testing
    '''
    prob_obst = grid_config['prob_obst']
    gridwidth = grid_config['gridwidth']
    gridlen = grid_config['gridlen']
    numgrids = grid_config['numgrids']

    gridlis = []
    for i in range(numgrids):
        gridlis.append(np.random.choice(a=[1.0, -1.0], size=(gridwidth, gridlen),
                                        p=[1-prob_obst, prob_obst]))

    # split into train and test sets
    l = len(gridlis)
    if l == 1:
        train_set = gridlis
        test_set = gridlis
    else:
        train_set = gridlis[:l]
        test_set = gridlis[l:]

    return train_set, test_set


if __name__ == "__main__":
    test1 = np.zeros((15, 15))
    test1[11, 3:8] = -1.0
    test1[7, 7:13] = -1.0
    test1[2:7, 3] = -1.0
    test1[2, 3:8] = -1.0

    test2 = np.zeros((15, 15))
    test2[10:13, 3] = -1.0
    test2[12, 4:8] = -1.0
    test2[10:13, 8] = -1.0
    test2[2:5, 6] = -1.0
    test2[2, 7:11] = -1.0
    test2[2:5, 11] = -1.0

    test3 = np.zeros((15, 15))
    test3[5:10, 2] = -1.0
    test3[5:10, 5] = -1.0
    test3[5:10, 9] = -1.0
    test3[5, 10:15] = -1.0
    test_set = [test1, test2, test3]

    train1 = np.zeros((15, 15))
    train1[2, 3] = -1.0
    train1[3, 7] = -1.0
    train1[1, 10:14] = -1.0
    train1[7, 5] = -1.0
    train1[8, 10] = -1.0
    train1[7:13, 2] = -1.0
    train1[10, 6:9] = -1.0
    train1[10:14, 8] = -1.0

    train2 = np.zeros((15, 15))
    train2[3:13, 7] = -1.0
    train2[7, 2:13] = -1.0

    train3 = np.zeros((15, 15))
    train3[2, 10:13] = -1.0
    train3[2:6, 12] = -1.0
    train3[4:11, 6] = -1.0
    train3[7, 6:8] = -1.0
    train3[10, 2] = -1.0

    train4 = np.zeros((15, 15))
    train4[4:10, 10] = -1.0
    train4[4:10, 5] = -1.0
    train4[9, 5:11] = -1.0

    train5 = np.zeros((15, 15))
    train5[3, 8:12] = -1.0
    train5[6, 4:8] = -1.0
    train5[9, 1:4] = -1.0
    train5[9:12, 11] = -1.0

    train6 = np.zeros((15, 15))
    train6[3:12, 11] = -1.0
    train6[3:12, 7] = -1.0
    train_set = [train1, train2, train3, train4, train5, train6]
