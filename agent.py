import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


class Agent:

    def __init__(self, init_x, init_y, basis_f, means, min_a, opt_a, pos_dim, color='r'):
        # agent position
        self.pos = np.array([[init_x], [init_y]], dtype='f')

        # agent color for rendering
        self.color = color

        # min parameter, optimal parameter vector, robot's current est of parameter vector
        self.MIN_A = min_a
        self.a_opt = opt_a.reshape(opt_a.shape[0], 1)
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

        self.la = np.zeros((opt_a.shape[0], opt_a.shape[0])) # capital lambda from equation 11
        self.lb = np.zeros((opt_a.shape[0], 1)) # lowercase lambda from equation 11

    def odom_command(self, gain_matrix):
        u = gain_matrix @ (self.e_centroid - self.pos) # eq 11
        self.pos[0] += u[0]
        self.pos[1] += u[1]

    def calc_est_centroid(self):
        for coord in self.v_part:
            phi_approx = self.sense_approx(coord)[0]
            self.e_mass += phi_approx
            self.e_moment += np.array(coord).reshape(len(coord), -1) * phi_approx # changes cell from row to column vector -- is this correct?
        self.e_centroid = self.e_moment / self.e_mass

    def calc_true_centroid(self):
        for coord in self.v_part:
                phi_approx = self.sense_approx(coord, opt=True)[0]
                self.t_mass += phi_approx
                self.t_moment += np.array(coord).reshape(len(coord), -1) * phi_approx # changes cell from row to column vector -- is this correct?
        self.t_centroid = self.t_moment / self.t_mass

    def sense_true(self):
        return self.calc_basis(np.array([self.pos[0,0], self.pos[1,0]])).T @ self.a_opt

    def sense_approx(self, q, opt=False):
        """ Looks up estimated sensor value at a position q """
        r = -1
        if opt:
            r = self.calc_basis(q).T @ self.a_opt
        else:
            r = self.calc_basis(q).T @ self.a_est
        return r

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
            if self.a_est[i] > self.MIN_A:
                j = 0
            if self.a_est[i] == self.MIN_A and a_pre[i] >= 0:
                j = 0
            I[i, i] = j

        return I

    def reset(self):
        self.v_part_list = []
        self.v_part = np.zeros((1, self.POS_DIM))

        self.e_mass = 0
        self.e_moment = np.zeros((self.POS_DIM, 1))
        self.e_centroid = np.zeros((self.POS_DIM, 1))

    def update_v_part(self):
        self.v_part = np.array(self.v_part_list)

    def render_self(self):
        plt.scatter(self.pos[0], self.pos[1], s=50, c=self.color)

    def render_centroid(self):
        plt.scatter(self.e_centroid[0], self.e_centroid[1], s=25, c=self.color)
