import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt


class Agent:

    def __init__(self, init_x, init_y, num_basis_functions, means, sigma, min_a, pos_dim, color='r'):
        self.pos = (init_x, init_y)
        self.MIN_A = min_a
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

    def move_agent(self, tv, av):
        pass

    def calc_centroid(self):
        for cell in self.v_part:
            # increment mass and moment
            phi_approx = self.sense_approx(cell)
            self.v_mass += phi_approx
            self.v_moment += np.array(cell).reshape(len(cell), -1) * phi_approx # changes cell from row to column vector -- is this correct?

        # calc centroid now that mass and moment have been obtained
        self.v_centroid = self.v_moment / self.v_mass

    def sense_true(self):
        return self.calc_basis(self.pos).T @ self.a

    def sense_approx(self, q):
        """ Looks up estimated sensor value at a position q """
        # print(self.calc_basis(q).T)
        return self.calc_basis(q).T @ self.a_est

    def calc_basis(self, pos):
        basis = []
        for mu in self.means:
            basis.append(1/(self.sigma * np.sqrt(2 * np.pi)) * np.exp( - (pos - mu)**2 / (2 * self.sigma**2)))
        return np.array(basis)

    def calc_F(self, gain_matrix):
        left = np.zeros((self.num_basis_functions, len(self.pos))) # 9x2 matrix
        right = np.zeros((len(self.pos), self.num_basis_functions))) # 2x9 matrix

        pos = np.array(self.pos).reshape(len(self.pos), -1)
        for cell in self.v_part:
            left += (self.calc_basis(cell) @ (cell - pos).T)
            right += ((cell - pos) @ self.calc_basis(cell).T)

        # calc F according to equation 12
        return (left @ gain_matrix @ right) / self.v_mass

    def sense(self):
        pass

    def render_all(self):
        self.render_self()
        self.render_voronoi_region()
        self.render_centroid()

    def render_self(self):
        plt.scatter(self.pos[1], self.pos[0], s=50, c=self.color)

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
