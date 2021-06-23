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

    def gcell_to_coord(self, cell):
        return int(cell[1] * self.cell_size), int(cell[0] * self.cell_size)

    def set_tree(self, agents):
        positions = [self.coord_to_gcell(agent.pos) for agent in agents]
        self.tree = KDTree(positions)

    def reset_agents(self, agents):
        for agent in agents:
            agent.reset()

    def set_agent_voronoi(self, agents):
        """ Determines the grid cells corresponding to each agent's Voronoi
            partition. An agents Voronoi cell positions are in map (row, col)
            form.
        """
        dists, inds = self.tree.query(np.c_[self.grows, self.gcols], k=1)
        inds = inds.reshape(self.map_height, self.map_width)

        # reset each agent's voronoi grid cells
        self.reset_agents(agents)

        # assign grid cells to each agent's voronoi partition
        for i in range(inds.shape[0]):
            for j in range(inds.shape[1]):
                x, y = self.gcell_to_coord((i, j))
                agents[inds[i, j]].v_part_list.append((x, y))

        # convert each agent's voronoi partition to a numpy array
        for agent in agents:
            agent.update_v_part()

    def calc_agent_voronoi(self, agents):
        """ Calculates the centroid, moment, and mass of each agent's voronoi
            partition
        """
        for agent in agents:
            agent.calc_est_centroid()

    def render_agents(self, agents):
        for a in agents:
            a.render()

    def render_voronoi_regions(self, agents):
        for a in agents:
            a.render_voronoi_region()

    def plot_voronoi(self, agents):
        positions = [self.coord_to_gcell(agent.pos) for agent in agents]
        vor = Voronoi(positions, qhull_options='Qbb Qc Qx')
        fig = voronoi_plot_2d(vor)
        plt.show()
