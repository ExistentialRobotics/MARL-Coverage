import numpy as np
import matplotlib.pyplot as plt

class CoverageEnvironment(object):
    """
    Class CoverageEnvironment represents a 2d multi agent environment where
    a number of robots have to maximize their cumulative sensing reward.
    """
    def __init__(self, map_width, map_height, cell_size, numrobot, numobstacles, dt, seed=None):
        #TODO incorporate sensing function in environment
        #TODO include random seed???
        super().__init__()

        self._numrobot = numrobot
        self._numobstacles = numobstacles
        self._dt = dt

        # create seed if user specifies it
        if seed is not None:
            np.random.seed(seed)

        # set map specific instance vars
        self.map_width = map_width
        self.map_height = map_height
        self.cell_size = cell_size

        # create numpy map
        self.map = np.zeros((map_height, map_width))
        self.grows, self.gcols = np.mgrid[0:map_height, 0:map_width]
        self.grows = self.grows.ravel()
        self.gcols = self.gcols.ravel()

        # setting random initial robot and obstacle positions
        self.reset()

    def coord_to_gcell(self, coord):
        return int((coord[1] / self.cell_size)), int((coord[0] /
                    self.cell_size))

    def gcell_to_coord(self, cell):
        return int(cell[1] * self.cell_size), int(cell[0] * self.cell_size)

    def step(self, U):
        #euler integrating the controls forward
        for i in range(len(U)):
            #TODO handle obstacle 'collisions'
            self._robot_coordinates[i] += U[i] * self._dt

        #returning current robot and obstacle positions
        #TODO add reward (negative coverage cost)
        return self._robot_coordinates, self._obst_coordinates

    def reset(self):
        #populating robot position list
        self._robot_coordinates = []
        for i in range(self._numrobot):
            #TODO make this for coordinates other than unit square
            xcoor = np.random.random_sample()
            ycoor = np.random.random_sample()
            self._robot_coordinates.append(np.array([[xcoor], [ycoor]]))

        #populating obstacle position list
        self._obst_coordinates = []
        for i in range(self._numobstacles):
            #TODO make this for coordinates other than unit square
            xcoor = np.random.random_sample()
            ycoor = np.random.random_sample()
            self._obst_coordinates.append(np.array([[xcoor], [ycoor]]))

    #TODO need to figure out a good way to do this, my previous way
    #doesn't work well for obstacles
    def render(self):
        pass
