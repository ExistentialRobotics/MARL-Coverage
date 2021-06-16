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


def adaptive_coverage(map, agents, opt_a):
    la = np.zeros((opt_a.shape)) # capital lambda from equation 11
    lb = np.zeros((opt_a.shape)) # lowercase lambda from equation 11

    # iterate until each agent's estimated paramters are close to the optimal
    # while conv_check(agents, opt_a, error=ERROR) == False:
        # reset KDTree used to compute Voronoi regions
    map.set_tree(agents)

    # calc centroid, mass, and moment for each agent
    map.set_agent_voronoi(agents)
    test = map.voronoi_calculations(agents)

    plt.figure(2)
    plt.imshow(test)
    plt.show()






if __name__ == "__main__":
    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    colors = [0.0, 0.25, 0.5, 0.75, 1]


    # get environment variables from json file
    with open('config.json') as f:
        env_vars = json.load(f)

    # print env vars for debugging purposes
    print_env_vars(env_vars)

    # set environment variables
    num_agents = env_vars["num_agents"]
    num_basis_functions = env_vars["num_basis_functions"]
    basis_sigma = env_vars["basis_sigma"]
    map_width = env_vars["map_width"]
    map_height = env_vars["map_height"]
    grid_cell_size = env_vars["grid_cell_size"]
    min_a = env_vars["min_a"]
    gain_matrix = env_vars["gain_matrix"]
    gamma_matrix = env_vars["gamma_matrix"]
    lr_gain = env_vars["lr_gain"]
    data_weighting = env_vars["data_weighting"]
    positive_consensus_gain = env_vars["positive_consensus_gain"]

    # set a (change this to be in config later)
    opt_a = np.array([100, min_a, min_a, min_a, min_a, min_a, min_a, min_a, 100])

    # instanciate map
    map = Map(map_width, map_height, grid_cell_size)

    # calc basis centers, assuming you want them in the center of each quadrant
    means = []
    for i in range(int(np.sqrt(num_basis_functions))):
        y = (map_height / (np.sqrt(num_basis_functions) * 2)) + i * (map_height / np.sqrt(num_basis_functions))
        for j in range(int(np.sqrt(num_basis_functions))):
            x = (map_width / (np.sqrt(num_basis_functions) * 2)) + j * (map_width / np.sqrt(num_basis_functions))
            means.append(((x + y) / 2))

    # create agents with random initial positions
    agents = []
    for i in range(num_agents):
        agents.append(Agent(np.random.randint(map_width), np.random.randint(map_height), num_basis_functions, means, basis_sigma, min_a, color=colors[i]))

    # print agent coordinates for debugging purposes
    print_agent_coords(agents)

    # run adaptive coverage algorithm
    adaptive_coverage(map, agents, opt_a)

    plt.show()
