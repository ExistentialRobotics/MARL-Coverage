import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree, Voronoi, voronoi_plot_2d
from . environment import Environment

class Grid_Environment(Environment):

    def __init__(self, agents, numrobot, numobstacles, obstradius, region, dt, map_width, map_height, cell_size, seed=None):
        super().__init__(agents, numrobot, numobstacles, obstradius, region, dt, seed=seed)

        # set map specific instance vars
        self.map_width = map_width
        self.map_height = map_height
        self.cell_size = cell_size

        # create numpy map
        self.map = np.zeros((map_height, map_width))
        self.grows, self.gcols = np.mgrid[0:map_height, 0:map_width]
        self.grows = self.grows.ravel()
        self.gcols = self.gcols.ravel()

    def step(self):
        # use KDTree to determine which grid cells are in each agent's voronoi region
        self.set_tree()
        dists, inds = self.tree.query(np.c_[self.grows, self.gcols], k=1)
        inds = inds.reshape(self.map_height, self.map_width)

        for agent in self.agents:
            est_mean, true_mean, a_mean, a_est = agent.step(dists, inds, self._dt)
        return est_mean, true_mean, a_mean, a_est

    def coord_to_gcell(self, coords):
        coords[:, [1, 0]] = coords[:, [0, 1]]
        return (coords / self.cell_size).astype('int')

    def gcell_to_coord(self, cell):
        cell[:, [1, 0]] = cell[:, [0, 1]]
        return cell * self.cell_size

    def set_tree(self):
        positions = self.coord_to_gcell(self.agents[0]._xlis)
        self.tree = KDTree(positions)
