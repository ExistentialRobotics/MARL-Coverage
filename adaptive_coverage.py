import numpy as np
import json
from scipy.spatial import Voronoi, voronoi_plot_2d, KDTree
import matplotlib.pyplot as plt
from map import *
from agent import *
from utils import *
import matplotlib.colors as mcolors

ERROR = 0.01


def conv_check(agents, a_opt, error=0.01):
    for a in agents:
        if np.linalg.norm(a.a_est - a_opt) > 0.01:
            return False
    return True


def adaptive_coverage(map, agents, opt_a, lr_gain, gain_const, gamma_const, data_weighting, render_agents=False, iters=2):
    # init gain and gamma matricies
    gain_matrix = gain_const * np.eye(2)
    gamma_matrix = gamma_const * np.eye(9)

    la = np.zeros((opt_a.shape[0], opt_a.shape[0])) # capital lambda from equation 11
    lb = np.zeros((opt_a.shape[0], 1)) # lowercase lambda from equation 11

    # iterate until each agent's estimated paramters are close to the optimal
    # while conv_check(agents, opt_a, error=ERROR) == False:
    for _ in range(iters):
        print("---------------------ITER: " + str(_ + 1) + "----------------------")

        # reset KDTree used to compute Voronoi regions
        map.set_tree(agents)

        # calc centroid, mass, and moment for each agent
        map.set_agent_voronoi(agents)
        map.calc_agent_voronoi(agents)

        print_agent_centroids(agents)

        # update a_est
        for agent in agents:
            F = -agent.calc_F(gain_matrix)

            print("F = " + str(F))

            a_pre = (F @ agent.a_est) - lr_gain * (la @ agent.a_est - lb) # eq 13
            print(str(F @ agent.a_est))

            I_proj = agent.calc_I(a_pre) # eq 15
            agent.a_est = gamma_matrix @ (a_pre - I_proj @ a_pre) # eq 14

        # update lambdas
        pos = (agent.pos[0] + agent.pos[1]) / 2
        basis = agent.calc_basis(pos)
        basis = basis.reshape(basis.shape[0], 1)
        la += data_weighting * (basis @ basis.T) # eq 11
        lb += data_weighting * (basis * agent.sense_true()) # eq 11

    # potentially render agents
    if render_agents:
        for agent in agents:
            agent.render_self()
            agent.render_centroid()

    map.plot_voronoi(agents)


if __name__ == "__main__":
    # set seed to get reproducable outputs
    np.random.seed(2)
    np.set_printoptions(suppress=True)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    # colors = [0.0, 0.25, 0.5, 0.75, 1]

    # get environment variables from json file
    with open('config.json') as f:
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
    POS_CONSENSUS_GAIN = np.array(env_vars["positive_consensus_gain"])
    POS_DIM = env_vars["pos_dim"]

    # set a (change this to be in config later)
    opt_a = np.array([100, MIN_A, MIN_A, MIN_A, MIN_A, MIN_A, MIN_A, MIN_A, 100])

    # instantiate map
    map = Map(MAP_WIDTH, MAP_HEIGHT, GRID_CELL_SIZE)

    # calc basis centers, assuming you want them in the center of each quadrant
    means = []
    for i in range(int(np.sqrt(NUM_BASIS_FX))):
        y = (MAP_HEIGHT / (np.sqrt(NUM_BASIS_FX) * 2)) + i * (MAP_HEIGHT / np.sqrt(NUM_BASIS_FX))
        for j in range(int(np.sqrt(NUM_BASIS_FX))):
            x = (MAP_WIDTH / (np.sqrt(NUM_BASIS_FX) * 2)) + j * (MAP_WIDTH / np.sqrt(NUM_BASIS_FX))
            means.append(((x + y) / 2))

    # create agents with random initial positions
    agents = []
    for i in range(NUM_AGENTS):
        agents.append(Agent(np.random.randint(MAP_WIDTH), np.random.randint(MAP_HEIGHT), NUM_BASIS_FX, means, BASIS_SIGMA, MIN_A, opt_a, POS_DIM, color=colors[i]))

    # print agent coordinates for debugging purposes
    print_agent_coords(agents)

    # run adaptive coverage algorithm
    adaptive_coverage(map, agents, opt_a, LR_GAIN, GAIN_CONST, GAMMA_CONST, DATA_WEIGHTING, render_agents=True)

    plt.show()
