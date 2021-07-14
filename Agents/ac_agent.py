import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from . single_agent import Single_Agent

class AC_Agent(Single_Agent):
    """
    Class Ac_Agent is an agent used for the Adaptive Coverage Algorithm.
    """
    def __init__(self, init_x, init_y, basis_f, means, min_a, a_opt, pos_dim,
                 color='r'):
        super().__init__(init_x, init_y, color='r')

        # min parameter, optimal parameter vector, robot's parameter vector est
        self.min_a = min_a
        self.a_opt = a_opt.reshape(a_opt.shape[0], 1)
        self.a_est = np.full((len(basis_f), 1), self.min_a)
        self.pos_dim = pos_dim

        # basis functions and mean of the basis functions
        self.basis_f = basis_f
        self.means = means

        # grid cells corresponding to the agent's voronoi partition
        self.v_part_list = []
        self.v_part = np.zeros((1, self.pos_dim))

        # Eq 7: est mass, moment, and centroid of agent's voronoi partition
        self.e_mass = 0
        self.e_moment = np.zeros((self.pos_dim, 1))
        self.e_centroid = np.zeros((self.pos_dim, 1))

        # true mass, moment, and centroid of agent's voronoi partition
        self.t_mass = 0
        self.t_moment = np.zeros((self.pos_dim, 1))
        self.t_centroid = np.zeros((self.pos_dim, 1))

        # from equation 11
        self.la = np.zeros((a_opt.shape[0], a_opt.shape[0])) # capital lambda
        self.lb = np.zeros((a_opt.shape[0], 1)) # lowercase lambda

        # consensus term for agent
        self.c_term = 0

    def odom_command(self, gain_matrix):
        u = gain_matrix @ (self.e_centroid - self.pos) # eq 11
        super().odom_command(u)

    def calc_est_centroid(self):
        for coord in self.v_part:
            phi_approx = self.sense_approx(coord)[0]
            self.e_mass += phi_approx
            self.e_moment += np.array(coord).reshape(len(coord), -1) * \
                             phi_approx
        self.e_centroid = self.e_moment / self.e_mass

    def calc_true_centroid(self):
        for coord in self.v_part:
                phi_approx = self.sense_approx(coord, opt=True)[0]
                self.t_mass += phi_approx
                self.t_moment += np.array(coord).reshape(len(coord), -1) * \
                                 phi_approx
        self.t_centroid = self.t_moment / self.t_mass

    def sense(self):
        return self.calc_basis(np.array([self.pos[0,0], self.pos[1,0]])).T @ \
               self.a_opt

    def sense_approx(self, q, opt=False):
        if opt:
            return self.calc_basis(q).T @ self.a_opt
        return self.calc_basis(q).T @ self.a_est

    def calc_basis(self, pos):
        basis = []
        for f in self.basis_f:
            basis.append(f.pdf(pos))
        return np.array(basis).reshape(len(basis), 1)

    def calc_F(self, gain_matrix):
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
        I = np.zeros((a_pre.shape[0], a_pre.shape[0]))

        # calc I according to eq 15
        for i in range(a_pre.shape[0]):
            j = 1
            if self.a_est[i] > self.min_a:
                j = 0
            if self.a_est[i] == self.min_a and a_pre[i] >= 0:
                j = 0
            I[i, i] = j

        return I

    def reset(self):
        # reset grid cells in voronoi region
        self.v_part_list = []
        self.v_part = np.zeros((1, self.pos_dim))

        # reset estimated centroid calculations
        self.e_mass = 0
        self.e_moment = np.zeros((self.pos_dim, 1))
        self.e_centroid = np.zeros((self.pos_dim, 1))

        # reset true centroid calculations
        self.t_mass = 0
        self.t_moment = np.zeros((self.pos_dim, 1))
        self.t_centroid = np.zeros((self.pos_dim, 1))

        # reset consensus term
        self.c_term = 0

    def update_v_part(self):
        self.v_part = np.array(self.v_part_list)

    def render_centroid(self):
        plt.scatter(self.e_centroid[0], self.e_centroid[1], s=25, c=self.color)
