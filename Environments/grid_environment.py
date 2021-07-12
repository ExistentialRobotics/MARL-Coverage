import numpy as np
import matplotlib.pyplot as plt
from environment import Environment

class Grid_Environment(Environment):

    def __init__(self, numobstacles, dt, map_width, map_height, cell_size, seed=None):
        super().__init__(numobstacles, dt, seed=seed)

        # set map specific instance vars
        self.map_width = map_width
        self.map_height = map_height
        self.cell_size = cell_size

        # create numpy map
        self.map = np.zeros((map_height, map_width))
        self.grows, self.gcols = np.mgrid[0:map_height, 0:map_width]
        self.grows = self.grows.ravel()
        self.gcols = self.gcols.ravel()

    def coord_to_gcell(self, coord):
        return int((coord[1] / self.cell_size)), int((coord[0] /
                    self.cell_size))

    def gcell_to_coord(self, cell):
        return int(cell[1] * self.cell_size), int(cell[0] * self.cell_size)



if __name__ == "__main__":
    env = Grid_Environment(1, 2, 3, 4, 5, 6)
