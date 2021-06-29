"""
adaptive_coverage.py contains the adaptive coverage algorithm given in the
paper: Decentralized, Adaptive Coverage Control for Networked Robots,
environment variable parsing, and instanciation of the map and agents.

Authors: Peter Stratton, Hannah Hui
Emails: pstratto@ucsd.edu, hahui@ucsd.edu
"""

import numpy as np
import json
from scipy.stats import multivariate_normal
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from map import *
from agent import *
from utils import *
import matplotlib.colors as mcolors

CONFIG = 'Experiment_1/config.json' # which config file to load hyperparameters

def adaptive_coverage(map, agents, a_opt, lr_gain, c_gain, gain_const, gamma_const,
                      data_weighting=None, DT=1, render_agents=False,
                      iters=100, consensus=False, lw=False, posdef_check=False):
    """
    adaptive_coverage implements the central algorithm produced by the paper:
    Decentralized, Adaptive Coverage Control for Networked Robots. It controls a
    group of agents to learn the distribution of sensing data in the environment
    using only information local to each agent.

    Parameters
    ----------
    map            : python object that represents the environment the agent
                     takes action in
    agents         : list containing the autonomous learners who's parameter
                     updates constitute "learning" the environment
    a_opt          : numpy array containing optimal parameter vector that the
                     agents are trying to learn
    lr_gain        : float reprenting the learning rate for agent parameter
                     updates
    gain_const     : float that scales the gain matrix, gain matrix is applied
                     to control inputs
    gamma_const    : float that scales the gamma matrix, gamma matrix is applied
                     when projecting agent parameters
    data_weighting : function to weight the data
    DT             : float representing the time interval between each algorithm
                     step
    render_agents  : boolean to display the agents each iteration or not
    iters          : int dictating the number of iterations to run the algorithm
                     for
    consensus      : boolean used to determine if the algo runs with the
                     parameter consensus update step or not
    posdef_check   : boolean used to determine if corollary 2 is checked on each
                     iteration or not

    Returns
    -------
    est_dists  : list containing mean distance from the agents to their
                 estimated centroids on each iteration
    true_dists : list contain mean distance from agents to their true centroids
                 on each iteration
    a_errors   : list containing the mean error from each agent's estimated
                 parameters and the optimal parameters on each iteration
    """
    plt.figure(1)

    # effectively have no data weighting if no function is provided
    if data_weighting is None:
        data_weighting = 1

    # init gain and gamma matricies
    gain_matrix = gain_const * np.eye(2)
    gamma_matrix = gamma_const * np.eye(a_opt.shape[0])

    # iterate until each agent's estimated paramters are close to the optimal
    est_errors = []
    true_errors = []
    a_errors = []
    for _ in range(iters):
        if (_ + 1) % 5 == 0:
            print("-----------------ITER: " + str(_ + 1) + "------------------")

        # potentially render agents
        if render_agents:
            plt.clf()
            plt.title("Agent Positions (Circles) and Gaussian Centers (Xs)")
            plt.xlabel('X')
            plt.ylabel('Y')
            # render areas that agents should be moving towards
            plt.scatter(agents[0].means[0][0], agents[0].means[0][1], s=50,
                        c='r', marker='x')
            plt.scatter(agents[0].means[len(agents[0].means) - 1][0],
                        agents[0].means[len(agents[0].means) - 1][1], s=50,
                        c='r', marker='x')
            for agent in agents:
                agent.render_self()
                agent.render_centroid()
            plt.draw()
            plt.pause(0.02)

        # reset KDTree used to compute Voronoi regions
        map.set_tree(agents)

        # calc centroid, mass, and moment for each agent
        map.set_agent_voronoi(agents)
        map.calc_agent_voronoi(agents)

        # calculate consensus terms if using consensus
        if consensus:
            map.set_consensus(agents, length_w=lw)

        # update a_est
        est_mean = 0
        true_mean = 0
        for agent in agents:
            # calc true centroid
            agent.calc_true_centroid()

            # calc F
            F = -agent.calc_F(gain_matrix)

            # use parameter update based on if consensus is used or not
            if consensus:
                a_pre = (F @ agent.a_est) - lr_gain * (agent.la @ agent.a_est -
                         agent.lb) - c_gain * agent.c_term # eq 20
            else:
                a_pre = (F @ agent.a_est) - lr_gain * (agent.la @ agent.a_est -
                         agent.lb) # eq 13

            I_proj = agent.calc_I(a_pre) # eq 15
            a_dot = a_pre - I_proj @ a_pre
            agent.a_est = agent.a_est + DT * gamma_matrix @ a_dot # eq 14
            agent.a_est = np.where(agent.a_est < agent.MIN_A, agent.MIN_A,
                                   agent.a_est)


            # update lambdas
            basis = agent.calc_basis(np.array([agent.pos[0,0], agent.pos[1,0]]))
            agent.la += data_weighting * (basis @ basis.T) # eq 11
            agent.lb += data_weighting * (basis * agent.sense_true()) # eq 11

            # inc estimated and true position errors
            est_mean += np.linalg.norm((agent.e_centroid - agent.pos))
            true_mean += np.linalg.norm((agent.t_centroid - agent.pos))

            # apply control input
            agent.odom_command(gain_matrix) # eq 10

        # average estimated position errors, true position errors, and a error
        est_errors.append((est_mean / len(agents)))
        true_errors.append((true_mean / len(agents)))
        a_errors.append(map.a_error(agents))

        # verify if corollary 2 holds on each iteration
        if posdef_check:
            p = map.pdef_check(agents[0])
            print("Positive definite basis functions: " + str(p))

    return est_errors, true_errors, a_errors


if __name__ == "__main__":
    # set seed to get reproducable outputs
    np.random.seed(2)
    np.set_printoptions(suppress=True)

    # agent color options
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    # get environment variables from json file
    with open(CONFIG) as f:
        print("---------------------------------------------------------------")
        print("Running experiement using: " + str(CONFIG))
        print("---------------------------------------------------------------")
        env_vars = json.load(f)

    # print env vars for debugging purposes
    print_env_vars(env_vars)

    # set environment variables
    NUM_AGENTS = env_vars["num_agents"]
    NUM_BASIS_FX = env_vars["num_basis_functions"]
    BASIS_SIGMA = env_vars["basis_sigma"]
    MAP_WIDTH = env_vars["map_width"]
    MAP_HEIGHT = env_vars["map_height"]
    GRID_CELL_SIZE = env_vars["grid_cell_size"]
    MIN_A = env_vars["min_a"]
    GAIN_CONST = env_vars["gain_const"]
    GAMMA_CONST = env_vars["gamma_const"]
    LR_GAIN = env_vars["lr_gain"]
    DATA_WEIGHTING = env_vars["data_weighting"]
    CONSENSUS_GAIN = env_vars["positive_consensus_gain"]
    POS_DIM = env_vars["pos_dim"]
    DT = env_vars["dt"]
    ITERS = env_vars["iters"]

    # determine whether or not to use consensus
    CONSENSUS = False
    if env_vars["consensus"] == 1:
        CONSENSUS = True

    # determine whether or not to weight consensus parameters by length
    LW = False
    if env_vars["length_w"] == 1:
        LW = True

    # determine whether or not to render agents
    RENDER_A = False
    if env_vars["render_a"] == 1:
        RENDER_A = True

    # determine whether or not to check gaussian basis positive definiteness
    PDEF = False
    if env_vars["posdef_check"] == 1:
        PDEF = True

    # determine whether or not to use a data weighting function
    if DATA_WEIGHTING == -1:
        d_f = None
    else:
        d_f = DATA_WEIGHTING

    # set a (change this to be in config later)
    opt_a = np.array([100, 100])
    if NUM_BASIS_FX == 9:
        opt_a = np.array([100, MIN_A, MIN_A, MIN_A, MIN_A, MIN_A, MIN_A, MIN_A,
                          100])

    # set gaussian basis functions and their means
    cov_m = BASIS_SIGMA * np.eye(2)
    means = np.array([(25, 25), (75, 75)])
    basis_f = [multivariate_normal(mean=np.array([25, 25]), cov=cov_m),
               multivariate_normal(mean=np.array([75, 75]), cov=cov_m)]
    if NUM_BASIS_FX == 9:
        # calc basis centers assuming they're in the center of each quadrant
        basis_f = []
        means = []
        for i in range(int(np.sqrt(NUM_BASIS_FX))):
            y = (MAP_HEIGHT / (np.sqrt(NUM_BASIS_FX) * 2)) + i * (MAP_HEIGHT /
                 np.sqrt(NUM_BASIS_FX))
            for j in range(int(np.sqrt(NUM_BASIS_FX))):
                x = (MAP_WIDTH / (np.sqrt(NUM_BASIS_FX) * 2)) + j * (MAP_WIDTH /
                     np.sqrt(NUM_BASIS_FX))
                means.append((x, y))

                # add multivariate normals as each basis function
                basis_f.append(multivariate_normal(mean=np.array([x, y]),
                               cov=cov_m))

    # instantiate map
    map = Map(MAP_WIDTH, MAP_HEIGHT, GRID_CELL_SIZE)

    # create agents with random initial positions
    agents = []
    for i in range(NUM_AGENTS):
        agents.append(Agent(np.random.randint(MAP_WIDTH),
                            np.random.randint(MAP_HEIGHT), basis_f, means,
                            MIN_A, opt_a, POS_DIM, color=colors[2]))

    # print agent coordinates for debugging purposes
    print_agent_coords(agents)

    # run adaptive coverage algorithm
    est_errors, true_errors, a_errors = adaptive_coverage(map, agents, opt_a,
                                                LR_GAIN, CONSENSUS_GAIN,
                                                GAIN_CONST, GAMMA_CONST,
                                                data_weighting=d_f, DT=DT,
                                                render_agents=RENDER_A,
                                                iters=ITERS,
                                                consensus=CONSENSUS, lw=LW,
                                                posdef_check=PDEF)

    # plot agent positions and means corresponding to highest a_opt values
    plt.figure(2)
    plt.title("Agent Positions (Circles) and Gaussian Centers (Xs)")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.scatter(agents[0].means[0][0], agents[0].means[0][1], s=50, c='r',
                marker='x')
    plt.scatter(agents[0].means[len(agents[0].means) - 1][0],
                agents[0].means[len(agents[0].means) - 1][1], s=50, c='r',
                marker='x')
    for agent in agents:
        agent.render_self()
        agent.render_centroid()

    # plot final voronoi diagram
    map.plot_voronoi(agents)

    # print final agent paramters
    print_agent_params(agents)

    # plot the dist from centroids per iteration
    plt.figure(4)
    plt.title("Dist from True and Estimated Centroids")
    plt.xlabel('Iterations')
    plt.ylabel('Dist')
    line_e, = plt.plot(est_errors, label="Est Centroid")
    line_t, = plt.plot(true_errors, label="True Centroid")
    plt.legend(handles=[line_e, line_t])
    plt.show()

    # plot the mean parameter error per interation
    plt.figure(3)
    plt.title("Mean Agent Parameter Error")
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    line_a, = plt.plot(a_errors, label="||a_tilde - a||")
    plt.legend(handles=[line_a])
    plt.show()
