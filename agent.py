import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt


class Agent:

    def __init__(self, init_x, init_y, num_basis_functions, means, sigma, min_a, color='r'):
        self.pos = (init_x, init_y)
        self.min_a = min_a
        self.num_basis_f = num_basis_functions
        self.means = means
        self.sigma = sigma
        self.a_est = np.full((self.num_basis_f, 1), self.min_a)
        self.color = color

        # grid cells corresponding to the agent's voronoi partition
        self.v_part = []

        # Eq 7: est mass, momenet, and centroid of agent's voronoi partition
        self.v_mass = None
        self.v_moment = None
        self.v_centroid = None

    def move_agent(self, tv, av):
        pass

    def calc_voronoi(self):
        pass

    def sense_true(self):
        return self.calc_basis(self.pos).T @ self.a

    def sense_approx(self, q):
        # print(self.calc_basis(q).T)
        return self.calc_basis(q).T @ self.a_est

    def calc_basis(self, pos):
        basis = []
        for mu in self.means:
            basis.append(1/(self.sigma * np.sqrt(2 * np.pi)) * np.exp( - (pos - mu)**2 / (2 * self.sigma**2)))
        return np.array(basis)

    def sense(self):
        pass

    def render(self):
        plt.scatter(self.pos[1], self.pos[0], s=50, c=self.color)

    def render_voronoi_region(self):
        rows = [v[0] for v in self.vor_vert]
        cols = [v[1] for v in self.vor_vert]
        plt.scatter(rows, cols, s=25, c=self.color)
