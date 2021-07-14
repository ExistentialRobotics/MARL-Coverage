import numpy as np
import matplotlib.pyplot as plt
from environment import Environment

class Grid_Environment(Environment):

    def __init__(self, agents, numobstacles, dt, map_width, map_height, cell_size, seed=None):
        super().__init__(agents, numobstacles, dt, seed=seed)

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

    def set_tree(self):
        positions = [self.coord_to_gcell(agent.pos) for agent in self.agents]
        self.tree = KDTree(positions)

    def set_agent_voronoi(self):
        dists, inds = self.tree.query(np.c_[self.grows, self.gcols], k=1)
        inds = inds.reshape(self.map_height, self.map_width)

        # reset each agent's voronoi grid cells
        self.reset()

    def calc_agent_voronoi(self):
        for agent in self.agents:
            agent.calc_est_centroid()

    def plot_voronoi(self):
        positions = [self.coord_to_gcell(agent.pos) for agent in self.agents]
        vor = Voronoi(positions, qhull_options='Qbb Qc Qx')
        fig = voronoi_plot_2d(vor)
        plt.show()
