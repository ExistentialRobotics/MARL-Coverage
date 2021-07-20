import numpy as np
import json
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

from controller import Controller
from Environments.ac_grid_environment import AC_Grid_Environment
from Agents.ac_agent import AC_Agent

# path to config file used to load hyperparameters
CONFIG = 'Consensus_Experiments/Experiment_1/config.json'

class AC_Controller(Controller):

    def __init__(self, numrobot):
        super().__init__(numrobot)

    def getControls(self, observation):
        pass

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
