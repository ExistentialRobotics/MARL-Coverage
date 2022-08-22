import numpy as np
import json
from scipy.stats import multivariate_normal
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from . controller import Controller

class AC_Controller(Controller):

    def __init__(self, numrobot, num_basis_fx, basis_sigma, map_width, map_height, grid_cell_size, min_a, gain_const, gamma_const, lr_gain, consensus_gain, pos_dim, consensus, lw, render_a, pdef, d_f, dt):
        super().__init__(numrobot)

        # set controller instance variables
        self.num_basis_fx = num_basis_fx
        self.basis_sigma = basis_sigma
        self.map_width = map_width
        self.map_height = map_height
        self.grid_cell_size = grid_cell_size
        self.min_a = min_a
        self.gain_const = gain_const
        self.gamma_const = gamma_const
        self.lr_gain = lr_gain
        self.c_gain = consensus_gain
        self.pos_dim = pos_dim
        self.consensus = consensus
        self.lw = lw
        self.render_a = render_a
        self.pdef = pdef
        self.d_f = d_f
        self.dt = dt

        # set a
        self.a_opt = np.array([100, 100])
        if self.num_basis_fx == 9:
            self.a_opt = np.array([100, self.min_a, self.min_a, self.min_a, self.min_a, self.min_a, self.min_a, self.min_a,
                              100])

        # set gaussian basis functions and their means
        self.cov_m = self.basis_sigma * np.eye(2)
        self.means = np.array([(25, 25), (75, 75)])
        self.basis_f = [multivariate_normal(mean=np.array([25, 25]), cov=self.cov_m),
                   multivariate_normal(mean=np.array([75, 75]), cov=self.cov_m)]
        if self.num_basis_fx == 9:
            # calc basis centers assuming they're in the center of each quadrant
            self.basis_f = []
            self.means = []
            for i in range(int(np.sqrt(self.num_basis_fx))):
                y = (self.map_height / (np.sqrt(self.num_basis_fx) * 2)) + i * (self.map_height /
                     np.sqrt(self.num_basis_fx))
                for j in range(int(np.sqrt(self.num_basis_fx))):
                    x = (self.map_width / (np.sqrt(self.num_basis_fx) * 2)) + j * (self.map_width /
                         np.sqrt(self.num_basis_fx))
                    self.means.append((x, y))

                    # add multivariate normals as each basis function
                    self.basis_f.append(multivariate_normal(mean=np.array([x, y]),
                                   cov=self.cov_m))

        # init gain and gamma matricies
        self.gain_matrix = self.gain_const * np.eye(2)
        self.gamma_matrix = self.gamma_const * np.eye(self.a_opt.shape[0])

        # mass and moments for agents
        self.e_phi_approx = np.zeros((numrobot, 1))
        self.e_mass = np.zeros((numrobot, 1))
        self.e_moment = np.zeros((numrobot, 2))
        self.e_centroid = np.zeros((numrobot, 1))
        self.t_phi_approx = np.zeros((numrobot, 1))
        self.t_mass = np.zeros((numrobot, 1))
        self.t_moment = np.zeros((numrobot, 2))
        self.t_centroid = np.zeros((numrobot, 1))

        # from equation 11
        self.la = np.zeros((self._numrobot, self.a_opt.shape[0], self.a_opt.shape[0])) # capital lambda
        self.lb = np.zeros((self._numrobot, self.a_opt.shape[0], 1)) # lowercase lambda


    def getControls(self, observation):
        agent_pos, dists, inds, a_est = observation

        num = np.zeros((self._numrobot, len(self.basis_f), agent_pos[0].shape[0]))
        for i in range(inds.shape[0]):
            for j in range(inds.shape[1]):
                coord = np.array([j * self.grid_cell_size, i * self.grid_cell_size])

                # calc est mass and moment
                e_phi_approx = self.sense_approx(coord, a_est[inds[i, j]])[0]
                self.e_mass[inds[i, j]] += e_phi_approx
                self.e_moment[inds[i, j]] += coord * e_phi_approx

                # calc true mass and moment
                t_phi_approx = self.sense_approx(coord, a_est[inds[i, j]], opt=True)[0]
                self.t_mass[inds[i, j]] += t_phi_approx
                self.t_moment[inds[i, j]] += coord * t_phi_approx

                # calculate basis and inc numerator matrix for F calculation
                basis = self.calc_basis(coord)
                num[inds[i, j]] += basis @ (coord - agent_pos[inds[i, j]]).reshape(1, len(self.basis_f))

        # calc est and true centroid
        self.e_centroid = self.e_moment / self.e_mass
        self.t_centroid = self.t_moment / self.t_mass

        # get agent consensus terms if necessary
        c_terms = None
        if self.consensus:
            c_terms = self.consensus_terms(agent_pos, a_est, length_w=self.lw)

        # calc updated params and get the controls for each agent
        u = np.zeros((self._numrobot, 2))
        est_mean = 0
        true_mean = 0
        a_mean = 0
        for i in range(self._numrobot):
            F = (num[i] @ self.gain_matrix @ num[i].T) / self.e_mass[i]
            # use parameter update based on if consensus is used or not
            a_temp = a_est[i].reshape(a_est[i].shape[0], -1)
            if self.consensus:
                c_temp = c_terms[i].reshape(c_terms[i].shape[0], 1)
                a_pre = (F @ a_temp) - self.lr_gain * (self.la[i] @ a_temp -
                         self.lb[i]) - self.c_gain * c_temp # eq 20
            else:
                a_pre = (F @ a_temp) - self.lr_gain * (self.la[i] @ a_temp -
                         self.lb[i]) # eq 13

            I_proj = self.calc_I(a_pre, a_est[i]) # eq 15
            a_dot = a_pre - I_proj @ a_pre
            a_est[i] = np.squeeze((a_temp + self.dt * self.gamma_matrix @ a_dot), axis=1) # eq 14
            a_est[i] = np.where(a_est[i] < self.min_a, self.min_a,
                                   a_est[i])

            # update lambdas
            basis = self.calc_basis(agent_pos[i])
            self.la[i] += self.d_f * (basis @ basis.T) # eq 11
            self.lb[i] += self.d_f * (basis * (basis.T @ self.a_opt.reshape(self.a_opt.shape[0], -1))) # eq 11

            # get estimated and true position errors
            est_mean += np.linalg.norm((self.e_centroid[i] - agent_pos[i]))
            true_mean += np.linalg.norm((self.t_centroid[i] - agent_pos[i]))
            a_mean += np.linalg.norm(a_est[i] - self.a_opt)

            # generate control input
            u[i] = self.gain_matrix @ (self.e_centroid[i] - agent_pos[i]) # eq 10
        return u, a_est, (est_mean / self._numrobot), (true_mean / self._numrobot), (a_mean / self._numrobot)

    def consensus_terms(self, agent_pos, a_est, length_w=False):
        tri = Delaunay(agent_pos).simplices
        c_terms = np.zeros((agent_pos.shape[0], 2))
        for t in tri:
            # get dists between self.agents
            d_01 = 1
            d_02 = 1
            d_12 = 1
            if length_w:
                d_01 = np.linalg.norm((agent_pos[t[0]] - agent_pos[t[1]]))
                d_02 = np.linalg.norm((agent_pos[t[0]] - agent_pos[t[2]]))
                d_12 = np.linalg.norm((agent_pos[t[1]] - agent_pos[t[2]]))

            # inc consensus terms for self.agents in the triangle
            c_terms[t[0]] += d_01 * (a_est[t[0]] - a_est[t[1]]) +\
                             d_02 * (a_est[t[0]] - a_est[t[2]])
            c_terms[t[1]] += d_01 * (a_est[t[1]] - a_est[t[0]]) +\
                             d_12 * (a_est[t[1]] - a_est[t[2]])
            c_terms[t[2]] += d_12 * (a_est[t[2]] - a_est[t[1]]) +\
                             d_02 * (a_est[t[2]] - a_est[t[0]])
        return c_terms

    def calc_I(self, a_pre, a_est):
        I = np.zeros((a_pre.shape[0], a_pre.shape[0]))

        # calc I according to eq 15
        for i in range(a_pre.shape[0]):
            j = 1
            if a_est[i] > self.min_a:
                j = 0
            if a_est[i] == self.min_a and a_pre[i] >= 0:
                j = 0
            I[i, i] = j
        return I

    def sense_approx(self, q, a_est, opt=False):
        if opt:
            return self.calc_basis(q).T @ self.a_opt
        return self.calc_basis(q).T @ a_est.reshape(a_est.shape[0], -1)

    def calc_basis(self, pos):
        basis = []
        for f in self.basis_f:
            basis.append(f.pdf(pos))
        return np.array(basis).reshape(len(basis), 1)
