import numpy as np
import matplotlib.pyplot as plt

class AC_Grid_Environment(Grid_Environment):

    def __init__(self, agents, numobstacles, dt, map_width, map_height, cell_size, gain_matrix, seed=None):
        super().__init__(agents, numobstacles, dt, map_width, map_height, cell_size, seed=None)
        self.gain_matrix

    def pdef_check(self):
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
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                b = self.agents[0].calc_basis(self.map[i, j])
                if np.all((b @ b.T <= 0)):
                    return False
        return True

    def set_consensus(self, length_w=False):
        """
        set_consensus sets each agent's consensus parameter by summing the
        differences between the agent's estimated paramters and its voronoi
        neighbor's estimated parameters.

        Parameters
        ----------
        self.agents   : list of self.agents to calculate the consensus parameters of
        length_w : boolean representing whether or not to weighting consensus
                   terms according length of shared voronoi edge
        """
        # get Delaunay triangulation to determine agent neighbors
        points = np.array([np.array([agent.pos[0,0], agent.pos[1,0]]) for agent in self.agents])
        tri = Delaunay(points).simplices

        c_terms = [0 for agent in self.agents]
        for t in tri:
            # get dists between self.agents
            d_01 = 1
            d_02 = 1
            d_12 = 1
            if length_w:
                d_01 = np.linalg.norm((self.agents[t[0]].pos - self.agents[t[1]].pos))
                d_02 = np.linalg.norm((self.agents[t[0]].pos - self.agents[t[2]].pos))
                d_12 = np.linalg.norm((self.agents[t[1]].pos - self.agents[t[2]].pos))

            # inc consensus terms for self.agents in the triangle
            c_terms[t[0]] += d_01 * (self.agents[t[0]].a_est - self.agents[t[1]].a_est) +\
                             d_02 * (self.agents[t[0]].a_est - self.agents[t[2]].a_est)
            c_terms[t[1]] += d_01 * (self.agents[t[1]].a_est - self.agents[t[0]].a_est) +\
                             d_12 * (self.agents[t[1]].a_est - self.agents[t[2]].a_est)
            c_terms[t[2]] += d_12 * (self.agents[t[2]].a_est - self.agents[t[1]].a_est) +\
                             d_02 * (self.agents[t[2]].a_est - self.agents[t[0]].a_est)

        # set each agent's consensus term
        for i in range(len(c_terms)):
            self.agents[i].c_term = c_terms[i]

    def a_error(self):
        """
        a_error calculates the parameter error between each agent's estimated
        parameters and the optimal parameters.

        Parameter
        ---------
        self.agents : list of self.agents to calculate the parameter errors of

        Return
        ------
        float reprenting the mean parameter error of all the self.agents
        """
        a_mean = 0
        for agent in self.agents:
            a_mean += np.linalg.norm((agent.a_opt - agent.a_est))
        # print((a_mean / len(self.agents)))
        return (a_mean / len(self.agents))

    def step(self):
        # update a_est
        est_mean = 0
        true_mean = 0
        for agent in self.agents:
            # calc true centroid
            agent.calc_true_centroid()

            # calc F
            F = -agent.calc_F(self.gain_matrix)

            # use parameter update based on if consensus is used or not
            if self.consensus:
                a_pre = (F @ agent.a_est) - lr_gain * (agent.la @ agent.a_est -
                         agent.lb) - c_gain * agent.c_term # eq 20
            else:
                a_pre = (F @ agent.a_est) - lr_gain * (agent.la @ agent.a_est -
                         agent.lb) # eq 13

            I_proj = agent.calc_I(a_pre) # eq 15
            a_dot = a_pre - I_proj @ a_pre
            agent.a_est = agent.a_est + self.dt * self.gamma_matrix @ a_dot # eq 14
            agent.a_est = np.where(agent.a_est < agent.MIN_A, agent.MIN_A,
                                   agent.a_est)


            # update lambdas
            basis = agent.calc_basis(np.array([agent.pos[0,0], agent.pos[1,0]]))
            agent.la += self.data_weighting * (basis @ basis.T) # eq 11
            agent.lb += self.data_weighting * (basis * agent.sense_true()) # eq 11

            # inc estimated and true position errors
            est_mean += np.linalg.norm((agent.e_centroid - agent.pos))
            true_mean += np.linalg.norm((agent.t_centroid - agent.pos))

            # apply control input
            agent.odom_command(self.gain_matrix) # eq 10

        return (est_mean / len(self.agents)), (true_mean / len(self.agents)), 
