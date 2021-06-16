import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, KDTree
from utils import *

class Map:

    def __init__(self, map_width, map_height, cell_size):
        # set instance vars
        self.map_width = map_width
        self.map_height = map_height
        self.cell_size = cell_size
        self.tree = None

        # create numpy map
        self.m = np.zeros((map_height, map_width))
        self.grows, self.gcols = np.mgrid[0:map_height, 0:map_width]
        self.grows = self.grows.ravel()
        self.gcols = self.gcols.ravel()

    def coord_to_gcell(self, coord):
        return int((coord[1] / self.cell_size)), int((coord[0] / self.cell_size))

    def set_tree(self, agents):
        positions = [self.coord_to_gcell(agent.pos) for agent in agents]
        self.tree = KDTree(positions)

    def voronoi_calculations(self, agents):
        """ Calculates the centroid, moment, and mass of each agent's voronoi
            partition
        """
        test = np.zeros((self.map_height, self.map_width))
        dists, inds = self.tree.query(np.c_[self.grows, self.gcols], k=1)
        print(inds.shape)

        for i in range(test.shape[0]):
            for j in range(test.shape[1]):
                test[i][j] = agents[inds[i * self.map_width + j]].color

        return test


    def render_agents(self, agents):
        for a in agents:
            a.render()

    def render_voronoi_regions(self, agents):
        for a in agents:
            a.render_voronoi_region()

    def set_agent_voronoi(self, agents):
        positions = [self.coord_to_gcell(agent.pos) for agent in agents]
        vor = Voronoi(positions, qhull_options='Qbb Qc Qx')

        vertices = vor.vertices
        p_to_reg = vor.point_region
        regions = vor.regions
        for i in range(len(p_to_reg)):
            for j in range(len(regions[i])):
                if regions[i][j] != -1:
                    agents[i].vor_vert.append(vertices[regions[i][j]])

        fig = voronoi_plot_2d(vor)
