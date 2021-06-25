"""
agent.py contains class Agent. Agent reprents the autonomous entities that are
learning the distribution of sensing information in the environment. Agents can
compute their own voronoi region, as well as values needed to perform the
parameter update in the central algorithm.

Authors: Peter Stratton, Hannah Hui
Emails: pstratto@ucsd.edu, hahui@ucsd.edu
"""

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


class Agent:
    """
    Class Agent represents the autonomous entities that are learning the
    distribution of sensing information in the environment. Agents can compute
    their own voronoi region, as well as values needed to perform the parameter
    update in the central algorithm.
    """

    def __init__(self, init_x, init_y, basis_f, means, min_a, a_opt, pos_dim,
                 color='r'):
        """
        Constructor for class Agent. It initializes its instance variables.

        Parameters
        ----------
        init_x  : int reprenting the agent's starting x coordinate
        init_y  : int reprenting the agent's starting y coordinate
        basis_f : list of gaussian basis functions used for the sensing function
        means   : list of means for the gaussian basis functions
        min_a   : float reprenting the min value allowed in each agent's
                  parameter vector
        a_opt   : numpy array containing the optimal sensing parameters
        pos_dim : int reprenting the number of positive dimensions
        color   : character designating the color of the agent when rendering
        """
        super().__init__()
        # agent position
        self.pos = np.array([[init_x], [init_y]], dtype='f')

        # agent color for rendering
        self.color = color

        # min parameter, optimal parameter vector, robot's parameter vector est
        self.MIN_A = min_a
        self.a_opt = a_opt.reshape(a_opt.shape[0], 1)
        self.a_est = np.full((len(basis_f), 1), self.MIN_A)
        self.POS_DIM = pos_dim

        # basis functions and mean of the basis functions
        self.basis_f = basis_f
        self.means = means

        # grid cells corresponding to the agent's voronoi partition
        self.v_part_list = []
        self.v_part = np.zeros((1, self.POS_DIM))

        # Eq 7: est mass, moment, and centroid of agent's voronoi partition
        self.e_mass = 0
        self.e_moment = np.zeros((self.POS_DIM, 1))
        self.e_centroid = np.zeros((self.POS_DIM, 1))

        # true mass, moment, and centroid of agent's voronoi partition
        self.t_mass = 0
        self.t_moment = np.zeros((self.POS_DIM, 1))
        self.t_centroid = np.zeros((self.POS_DIM, 1))

        # from equation 11
        self.la = np.zeros((a_opt.shape[0], a_opt.shape[0])) # capital lambda
        self.lb = np.zeros((a_opt.shape[0], 1)) # lowercase lambda

        # consensus term for agent
        self.c_term = 0

    def odom_command(self, gain_matrix):
        """
        odom_command executes the odometry command for the agent based on its
        distance from its estimated centroid.

        Parameters
        ----------
        gain_matrix : matrix used to scale the odom command
        """
        u = gain_matrix @ (self.e_centroid - self.pos) # eq 11
        self.pos[0] += u[0]
        self.pos[1] += u[1]

    def calc_est_centroid(self):
        """
        calc_est_centroid calculates the agent's estimated centroid based on its
        estimated parameter vector.
        """
        for coord in self.v_part:
            phi_approx = self.sense_approx(coord)[0]
            self.e_mass += phi_approx
            self.e_moment += np.array(coord).reshape(len(coord), -1) * \
                             phi_approx
        self.e_centroid = self.e_moment / self.e_mass

    def calc_true_centroid(self):
        """
        calc_true_centroid calculates the agent's true centroid based on the
        true parameter vector.
        """
        for coord in self.v_part:
                phi_approx = self.sense_approx(coord, opt=True)[0]
                self.t_mass += phi_approx
                self.t_moment += np.array(coord).reshape(len(coord), -1) * \
                                 phi_approx
        self.t_centroid = self.t_moment / self.t_mass

    def sense_true(self):
        """
        sense_true returns the true sensing value at the robot's current
        position.

        Returns
        -------
        float representing the value of the true sensing function
        """
        return self.calc_basis(np.array([self.pos[0,0], self.pos[1,0]])).T @ \
               self.a_opt

    def sense_approx(self, q, opt=False):
        """
        sense_approx calculates the estimated sensor value at a position q.

        Parameters
        ----------
        q   : 2x1 numpy array reprenting the position to calculate the sensing
              value at
        opt : boolean dictating whether to use the optimal parameter vector or
              not

        Return
        ------
        float reprenting the value of the approximate sensing function
        """
        if opt:
            return self.calc_basis(q).T @ self.a_opt
        return self.calc_basis(q).T @ self.a_est

    def calc_basis(self, pos):
        """
        calc_basis evaluates the basis functions at a given coordinate.

        Parameter
        ---------
        pos : 2x1 numpy array to evaluate the basis functions at

        Return
        ------
        numpy array representing the value of the coordinate evaluated at each
        basis function
        """
        basis = []
        for f in self.basis_f:
            basis.append(f.pdf(pos))
        return np.array(basis).reshape(len(basis), 1)

    def calc_F(self, gain_matrix):
        """
        calc_F calculates the F matrix used in equation 13 to update each
        agent's parameter vector.

        Parameter
        ---------
        gain_matrix : 2x2 numpy array used to scale odom commands

        Return
        ------
        Square numpy matrix who's dim depends on the number of basis functions,
        each value in it should be the same
        """
        num = np.zeros((len(self.basis_f), len(self.pos))) # 9x2 matrix

        # calculate left and right voronoi integrations in numerator of 12
        for coord in self.v_part:
            # calculate basis
            basis = self.calc_basis(coord)
            coord = np.array(coord).reshape(len(coord), -1)

            # increment numerator matrix
            num += basis @ (coord - self.pos).T

        # calc F according to equation 12
        return (num @ gain_matrix @ num.T) / self.e_mass

    def calc_I(self, a_pre):
        """
        calc_I calculates the projection matrix used to project a_pre to legal
        values. The calculation of the projection matrix is specified in
        equation 15 and it is used in equation 15.

        Parameter
        ---------
        a_pre : numpy array who's values dictate the elements in the projection
                matrix

        Return
        ------
        Diagonal matrix who's elements are dictated by equation 15
        """
        I = np.zeros((a_pre.shape[0], a_pre.shape[0]))

        # calc I according to eq 15
        for i in range(a_pre.shape[0]):
            j = 1
            if self.a_est[i] > self.MIN_A:
                j = 0
            if self.a_est[i] == self.MIN_A and a_pre[i] >= 0:
                j = 0
            I[i, i] = j

        return I

    def reset(self):
        """
        reset resets the agent's voronoi related instance variables.
        """
        # reset grid cells in voronoi region
        self.v_part_list = []
        self.v_part = np.zeros((1, self.POS_DIM))

        # reset estimated centroid calculations
        self.e_mass = 0
        self.e_moment = np.zeros((self.POS_DIM, 1))
        self.e_centroid = np.zeros((self.POS_DIM, 1))

        # reset true centroid calculations
        self.t_mass = 0
        self.t_moment = np.zeros((self.POS_DIM, 1))
        self.t_centroid = np.zeros((self.POS_DIM, 1))

        # reset consensus term
        self.c_term = 0

    def update_v_part(self):
        """
        update_v_part converts the list containing the agent's voronoi region
        grid cells to a numpy array.
        """
        self.v_part = np.array(self.v_part_list)

    def render_self(self):
        """
        render_self displays the agent's position using matplotlib.
        """
        plt.scatter(self.pos[0], self.pos[1], s=50, c=self.color)

    def render_centroid(self):
        """
        render_centroid displays the agent's centroid position using matplotlib.
        """
        plt.scatter(self.e_centroid[0], self.e_centroid[1], s=25, c=self.color)
