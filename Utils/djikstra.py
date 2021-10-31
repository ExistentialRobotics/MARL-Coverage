import numpy as np


def djikstra(x0, y0, grid):
    """
    Args:
       x0 : the initial x position on the grid
       y0 : the initial y position
       grid : an array representing the environment, 0 is explored,
             1 is unexplored, and -1 is obstacle

    Returns:
       the cost to every node in the grid as an array of the same shape as the grid.

    Notes:
    we will assume that the goals points are unexplored cells
    """
    cost = -1 *np.ones(grid.shape) #starting all nodes with negative cost
    pass

