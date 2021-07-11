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
from scipy.spatial import Delaunay
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

    def pdef_check(self, agent):
        """
        pdef_check verifies that the basis functions are positive definite at
        each position in the environment.

        Parameter
        ---------
        agent : agent used to calculate the basis functions at each position,
                any agent works as this parameter

        Return
        ------
        boolean reprenting if the basis functions were positive definite or not
        """
        for i in range(self.m.shape[0]):
            for j in range(self.m.shape[1]):
                b = agent.calc_basis(self.m[i, j])
                if np.all((b @ b.T <= 0)):
                    return False
        return True

    def set_consensus(self, agents, length_w=False):
        """
        set_consensus sets each agent's consensus parameter by summing the
        differences between the agent's estimated paramters and its voronoi
        neighbor's estimated parameters.

        Parameters
        ----------
        agents   : list of agents to calculate the consensus parameters of
        length_w : boolean representing whether or not to weighting consensus
                   terms according length of shared voronoi edge
        """
        # get Delaunay triangulation to determine agent neighbors
        points = np.array([np.array([agent.pos[0,0], agent.pos[1,0]]) for agent in agents])
        tri = Delaunay(points).simplices

        c_terms = [0 for agent in agents]
        for t in tri:
            # get dists between agents
            d_01 = 1
            d_02 = 1
            d_12 = 1
            if length_w:
                d_01 = np.linalg.norm((agents[t[0]].pos - agents[t[1]].pos))
                d_02 = np.linalg.norm((agents[t[0]].pos - agents[t[2]].pos))
                d_12 = np.linalg.norm((agents[t[1]].pos - agents[t[2]].pos))

            # inc consensus terms for agents in the triangle
            c_terms[t[0]] += d_01 * (agents[t[0]].a_est - agents[t[1]].a_est) +\
                             d_02 * (agents[t[0]].a_est - agents[t[2]].a_est)
            c_terms[t[1]] += d_01 * (agents[t[1]].a_est - agents[t[0]].a_est) +\
                             d_12 * (agents[t[1]].a_est - agents[t[2]].a_est)
            c_terms[t[2]] += d_12 * (agents[t[2]].a_est - agents[t[1]].a_est) +\
                             d_02 * (agents[t[2]].a_est - agents[t[0]].a_est)

        # set each agent's consensus term
        for i in range(len(c_terms)):
            agents[i].c_term = c_terms[i]

    def a_error(self, agents):
        """
        a_error calculates the parameter error between each agent's estimated
        parameters and the optimal parameters.

        Parameter
        ---------
        agents : list of agents to calculate the parameter errors of

        Return
        ------
        float reprenting the mean parameter error of all the agents
        """
        a_mean = 0
        for agent in agents:
            a_mean += np.linalg.norm((agent.a_opt - agent.a_est))
        # print((a_mean / len(agents)))
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
