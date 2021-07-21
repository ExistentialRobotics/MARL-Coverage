import sys
import numpy as np
import json
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from Controllers.ac_controller import AC_Controller
from Agents.ac_swarm_agent import AC_Swarm_Agent
from Environments.grid_environment import Grid_Environment

DASH = "-----------------------------------------------------------------------"

# prevent decimal printing
np.set_printoptions(suppress=True)


""" Read config file """
# check if config path is valid
config_path = sys.argv[1]
try:
    config_file = open(config_path)
except OSError:
    print(DASH)
    print(str(config_path) + " does not exit.")
    print(DASH)
    sys.exit(1)

# load json file
try:
    hyperparams = json.load(config_file)
except:
    print(DASH)
    print(str(config_path) + " is an invalid json file.")
    print(DASH)
    sys.exit(1)

print(DASH)
print("Running experiement using: " + str(config_path))
print(DASH)


""" Get hyperparameters """
# set environment variables as controller instance variables
num_agents = hyperparams["num_agents"]
num_basis_fx = hyperparams["num_basis_functions"]
basis_sigma = hyperparams["basis_sigma"]
map_width = hyperparams["map_width"]
map_height = hyperparams["map_height"]
grid_cell_size = hyperparams["grid_cell_size"]
min_a = hyperparams["min_a"]
gain_const = hyperparams["gain_const"]
gamma_const = hyperparams["gamma_const"]
lr_gain = hyperparams["lr_gain"]
consensus_gain = hyperparams["positive_consensus_gain"]
pos_dim = hyperparams["pos_dim"]
dt = hyperparams["dt"]
iters = hyperparams["iters"]
num_obstacles = hyperparams["num_obstacles"]
obstacle_radius = hyperparams["obstacle_radius"]
np.random.seed(hyperparams["seed"])

# determine whether or not to use consensus
consensus = False
if hyperparams["consensus"] == 1:
    consensus = True

# determine whether or not to weight consensus parameters by length
lw = False
if hyperparams["length_w"] == 1:
    lw = True

# determine whether or not to render agents
render_a = False
if hyperparams["render_a"] == 1:
    render_a = True

# determine whether or not to check gaussian basis positive definiteness
pdef = False
if hyperparams["posdef_check"] == 1:
    pdef = True

# determine whether or not to use a data weighting function
if hyperparams["data_weighting"] == -1:
    d_f = 1
else:
    d_f = hyperparams["data_weighting"]

colorlist = list(np.random.rand(num_agents))
region = np.array([[0,0], [map_width,map_height]],dtype=float)


""" Instanciate objects """
# create controller
c = AC_Controller(num_agents, num_basis_fx, basis_sigma, map_width, map_height, grid_cell_size, min_a, gain_const, gamma_const, lr_gain, consensus_gain, pos_dim, consensus, lw, render_a, pdef, d_f, dt)

# create agents
agents = [AC_Swarm_Agent(num_agents, c, colorlist, min_a, num_basis_fx)]

# create environment
env = Grid_Environment(agents, num_agents, num_obstacles, obstacle_radius, region, dt, map_width, map_height, grid_cell_size)

""" Main loop """
for i in range(iters):
    env.step()
    env.render()
