"""
map.py contains class Map. Map contains the numpy matrix that represents a
discretized grid cell environment, along with methods to perform operations
related to each agent's voronoi partitions.

Authors: Peter Stratton, Hannah Hui
Emails: pstratto@ucsd.edu, hahui@ucsd.edu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, KDTree
from utils import *

class Map:
    """
    Class Map contains numpy matrix represting the grid cell environment, along
    with methods to perform operations related to each agent's voronoi
    partitions.
    """

    def __init__(self, map_width, map_height, cell_size):
        """
        Constructor for class Map. Initializes the map's instance variables.

        Parameters
        ----------
        map_width  : int reprenting the number of grid cells wide the map is
        map_height : int reprenting the number of grid cells tall the map is
        cell_size  : int reprenting size of each grid cell
        """
        super().__init__()
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

    def a_error(self, agents):
        a_mean = 0
        for agent in agents:
            # print(np.linalg.norm((agent.a_opt - agent.a_est)))
            a_mean += np.linalg.norm((agent.a_opt - agent.a_est))
        return (a_mean / len(agents))

    def coord_to_gcell(self, coord):
        """
        coord_to_gcell converts a x, y coordinate to a row, col grid cell.

        Parameters
        ----------
        coord : tuple reprenting the coordinate to convert to a grid cell

        Returns
        -------
        tuple reprenting the coordinate as a grid cell
        """
        return int((coord[1] / self.cell_size)), int((coord[0] /
                    self.cell_size))

    def gcell_to_coord(self, cell):
        """
        gcell_to_coord converts a row, col grid cell to a x, y coordinate.

        Parameters
        ----------
        cell : tuple representing the grid cell to convert to a coordinate

        Returns
        -------
        tuple reprenting the grid cell as a coordinate
        """
        return int(cell[1] * self.cell_size), int(cell[0] * self.cell_size)

    def set_tree(self, agents):
        """
        set_tree uses the agents' positions to initialize a KDTree.

        Parameters
        ----------
        agents : list of agents whose positions are used for the KDTree
        """
        positions = [self.coord_to_gcell(agent.pos) for agent in agents]
        self.tree = KDTree(positions)

    def reset_agents(self, agents):
        """
        reset_agents resets each agent using the agents reset method.

        Parameters
        ----------
        agents : list of agents to reset
        """
        for agent in agents:
            agent.reset()

    def set_agent_voronoi(self, agents):
        """
        set_agent_voronoi determines the grid cells corresponding to each
        agent's Voronoi region. An agents Voronoi cell positions are in grid
        cell (row, col) form.

        Parameters
        ----------
        agents : list of agents to determine the voronoi regions of
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
        """
        calc_agent_voronoi calculates the centroid, moment, and mass of each
        agent's voronoi region.

        Parameters
        ----------
        agents : list of agents whose centroid, moment, and mass are calculated
        """
        for agent in agents:
            agent.calc_est_centroid()

    def render_agents(self, agents):
        """
        render_agents renders each agent.

        Parameters
        ----------
        agents : list of agents to render
        """
        for a in agents:
            a.render()

    def plot_voronoi(self, agents):
        """
        plot_voronoi plots each agents voronoi region.

        Parameters
        ----------
        agents : list of agents who's voronoi regions are plotted
        """
        positions = [self.coord_to_gcell(agent.pos) for agent in agents]
        vor = Voronoi(positions, qhull_options='Qbb Qc Qx')
        fig = voronoi_plot_2d(vor)
        plt.show()