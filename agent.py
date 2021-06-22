import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt


class Agent:

    def __init__(self, init_x, init_y, num_basis_functions, means, sigma, min_a, opt_a, pos_dim, color='r'):
        self.pos = np.array([[init_x], [init_y]], dtype='f')
        self.MIN_A = min_a
        self.opt_a = opt_a.reshape(opt_a.shape[0], 1)
        self.num_basis_fx = num_basis_functions
        self.means = means
        self.sigma = sigma
        self.a_est = np.full((self.num_basis_fx, 1), self.MIN_A)
        self.POS_DIM = pos_dim
        self.color = color

        # grid cells corresponding to the agent's voronoi partition
        self.v_part_list = []
        self.v_part = np.zeros((1, self.POS_DIM))

        # Eq 7: est mass, moment, and centroid of agent's voronoi partition
        self.v_mass = 0
        self.v_moment = np.zeros((self.POS_DIM, 1))
        self.v_centroid = np.zeros((self.POS_DIM, 1))

        # true mass, moment, and centroid of agent's voronoi partition
        self.t_mass = 0
        self.t_moment = np.zeros((self.POS_DIM, 1))
        self.t_centroid = np.zeros((self.POS_DIM, 1))

        self.la = np.zeros((opt_a.shape[0], opt_a.shape[0])) # capital lambda from equation 11
        self.lb = np.zeros((opt_a.shape[0], 1)) # lowercase lambda from equation 11

    def odom_command(self, gain_matrix):
        u = gain_matrix @ (self.v_centroid - np.array(self.pos).reshape(len(self.pos), 1)) # eq 11
        self.pos[0] += u[0]
        self.pos[1] += u[1]

    def calc_centroid(self, true_c=False):
        for coord in self.v_part:
            # increment mass and moment
            if true_c:
                phi_approx = self.sense_approx(coord, opt=True)[0]
            else:
                phi_approx = self.sense_approx(coord)[0]

            if true_c:
                self.v_mass += phi_approx
                self.v_moment += np.array(coord).reshape(len(coord), -1) * phi_approx # changes cell from row to column vector -- is this correct?
            else:
                self.t_mass += phi_approx
                self.t_moment += np.array(coord).reshape(len(coord), -1) * phi_approx # changes cell from row to column vector -- is this correct?

        # calc centroid now that mass and moment have been obtained
        if true_c:
            self.v_centroid = self.v_moment / self.v_mass
        else:
            self.t_centroid = self.t_moment / self.t_mass

    def sense_true(self):
        pos = (self.pos[0] + self.pos[1]) / 2
        return self.calc_basis(pos).T @ self.opt_a

    def sense_approx(self, q, opt=False):
        """ Looks up estimated sensor value at a position q """
        q = (q[0] + q[1]) / 2
        r = -1
        if opt:
            r = self.calc_basis(q).T @ self.opt_a
        else:
            r = self.calc_basis(q).T @ self.a_est
        return r

    def calc_basis(self, pos):
        basis = []
        for mu in self.means:
            basis.append(1/(self.sigma * np.sqrt(2 * np.pi)) * np.exp( - (pos - mu)**2 / (2 * self.sigma**2)))
        return np.array(basis)

    def calc_F(self, gain_matrix):
        left = np.zeros((self.num_basis_fx, len(self.pos))) # 9x2 matrix
        right = np.zeros((len(self.pos), self.num_basis_fx)) # 2x9 matrix

        # calculate left and right voronoi integrations in numerator of 12
        pos = np.array(self.pos).reshape(len(self.pos), -1)
        for coord in self.v_part:
            # calculate basis
            c = (coord[0] + coord[1]) / 2
            coord = np.array(coord).reshape(len(coord), -1)
            basis = self.calc_basis(c)

            # increment left and right matrix calculations
            left += (basis.reshape(basis.shape[0], -1) @ (coord - pos).T)
            right += ((coord - pos) @ basis.reshape(basis.shape[0], -1).T)

        # calc F according to equation 12
        return (left @ gain_matrix @ right) / self.v_mass

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

    def sense(self):
        pass

    def render_all(self):
        self.render_self()
        self.render_voronoi_region()
        self.render_centroid()

    def render_self(self):
        plt.scatter(self.pos[0], self.pos[1], s=50, c=self.color)

    def render_centroid(self):
        plt.scatter(self.v_centroid[0], self.v_centroid[1], s=25, c=self.color)

    def render_voronoi_region(self):
        rows = [v[0] for v in self.vor_vert]
        cols = [v[1] for v in self.vor_vert]
        plt.scatter(rows, cols, s=25, c=self.color)

    def reset(self):
        self.v_part_list = []
        self.v_part = np.zeros((1, self.POS_DIM))

        self.v_mass = 0
        self.v_moment = np.zeros((self.POS_DIM, 1))
        self.v_centroid = np.zeros((self.POS_DIM, 1))

    def update_v_part(self):
        self.v_part = np.array(self.v_part_list)
