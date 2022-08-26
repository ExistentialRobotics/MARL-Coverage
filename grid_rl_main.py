"""
grid_rl_main.py is the main file for the RL coverage codebase. It takes command
line arguements to run an experiment using one of the available policy types,
runs the experiment, and outputs the results to the command line. The user also
has the option to run a saved model.

Author: Peter Stratton
Email: pstratto@ucsd.edu, pstratt@umich.edu, peterstratton121@gmail.com
Author: Shreyas Arora
Email: sharora@ucsd.edu
"""

import numpy as np
import getopt
import sys
import json

from Environments.super_grid_rl import SuperGridRL
from Environments.sim_env import SuperGrid_Sim
from Environments.dec_grid_rl import DecGridRL

from Action_Spaces.discrete import Discrete

from Policies.stc import STC
from Policies.bsa import BSA
from Policies.ba_star import BA_Star
from Policies.mastc import MASTC
from Policies.dijkstra_frontier import DijkstraFrontier
from Policies.basic_random import Basic_Random
from Policies.dqn import DQN
from Policies.drqn import DRQN
from Policies.vdn import VDN
from Policies.ha_star import HA_Star

from Logger.logger import Logger

from Utils.utils import train_RLalg, test_RLalg
from Utils.gridmaker import gridgen, gridload

from Policies.Networks.grid_rl_conv import Grid_RL_Conv
from Policies.Networks.grid_rl_recur import Grid_RL_Recur
from Policies.Networks.gnn import GNN
from Policies.Networks.vin import VIN
from Policies.Networks.rvin import RVIN
from Policies.Networks.alpha_net import Alpha_Net
from Policies.Networks.alpha_net_wstep import Alpha_Net_WStep


NON_LEARNING = ["random", "bsa", "ba_star",
                "dijkstra_frontier", "mastc", "stc"]
DASH = "-----------------------------------------------------------------------"

# prevent decimal printing
np.set_printoptions(suppress=True)

'''Read config file'''
# Options
options = "m:"

# Long options
long_options = ["model ="]

# Remove 1st argument from the list of command line arguments
argumentList = sys.argv[1:]

saved_model = False
model_path = None
output_dir = None
try:
    # Parsing argument
    arguments, values = getopt.getopt(argumentList, options, long_options)

    # checking each argument
    for currentArgument, currentValue in arguments:

        if currentArgument in ("-m", "--model"):
            print(("Running saved model directly from % s") % (currentValue))
            model_path = currentValue
            saved_model = True
        else:
            print("Not a valid arg.")

except getopt.error as err:
    # output error, and return with an error code
    print(str(err))

if saved_model:
    # check if model path is valid
    try:
        model_file = open(model_path)
    except OSError:
        print(DASH)
        print(str(model_path) + " does not exist.")
        print(DASH)
        sys.exit(1)

    # determine index of second \ character
    sc = 0
    for i in range(len(model_path) - 1, -1, -1):
        if model_path[i] == "/":
            sc += 1
        if sc == 2:
            break

    # create config file path string
    output_dir = model_path[:i + 1]
    config_path = output_dir + "config.json"
else:
    if len(sys.argv) != 2:
        print(DASH)
        print("No config file specified.")
        print(DASH)
        sys.exit(1)
    config_path = sys.argv[1]
    for i in range(len(config_path) - 1, -1, -1):
        if config_path[i] == "/":
            break
    output_dir = config_path[:i + 1]

# check if config path is valid
try:
    config_file = open(config_path)
except OSError:
    print(DASH)
    print(str(config_path) + " does not exist.")
    print(DASH)
    sys.exit(1)

# load json file
try:
    exp_parameters = json.load(config_file)
except:
    print(DASH)
    print(str(config_path) + " is an invalid json file.")
    print(DASH)
    sys.exit(1)

'''Environment Parameters'''
env_name = exp_parameters["env_name"]
env_config = exp_parameters['env_config']
numrobot = env_config['numrobot']

'''Experiment Parameters'''
exp_config = exp_parameters['exp_config']
exp_name = exp_config["exp_name"]
train_episodes = exp_config["train_episodes"]
test_episodes = exp_config["test_episodes"]
show_fig = exp_config["render_plots"]
render_test = exp_config['render_test']
render_train = exp_config['render_train']
makevid = exp_config["makevid"]
ignore_done = exp_config['ignore_done']

'''Model Parameters'''
if "model_config" in exp_parameters:
    model_config = exp_parameters['model_config']
    model_name = model_config['model_name']

'''Policy Parameters'''
policy_config = exp_parameters['policy_config']
policy_name = policy_config['policy_name']

print(DASH)
print("Running experiment using: " + str(config_path))
print(DASH)

'''Init logger'''
logger = Logger(output_dir, exp_name, makevid)

'''Making the list of grids'''
grid_config = exp_parameters['grid_config']
if grid_config['gridload']:
    if grid_config['grid_dir'] == 0:
        train_set, test_set = gridload(grid_config=None)
    else:
        train_set, test_set = gridload(grid_config=grid_config)
else:
    train_set, test_set = gridgen(grid_config)

print("Number of training environments: " + str(len(train_set)))
print("Number of testing environments: " + str(len(test_set)))

'''Making the environment'''
if env_name == 'SuperGridRL':
    env = SuperGridRL(train_set, env_config, test_set=test_set)
elif env_name == 'DecGridRL':
    env = DecGridRL(train_set, env_config, use_graph=policy_config["use_graph"],
                    test_set=test_set)

num_actions = env._num_actions
obs_dim = env._obs_dim

'''Init action space'''
action_space = Discrete(num_actions)

'''Init policy'''
learning_policy = True
if policy_name in NON_LEARNING:
    learning_policy = False

    if policy_name == "random":
        policy = Basic_Random(action_space)
    if policy_name == "stc":
        policy = STC(policy_config["internal_grid_rad"])
    elif policy_name == "bsa":
        policy = BSA(policy_config["internal_grid_rad"])
    elif policy_name == "ba_star":
        policy = BA_Star(policy_config["internal_grid_rad"],
                         env_config["egoradius"])
    elif policy_name == "mastc":
        policy = MASTC()
    else:
        policy = DijkstraFrontier()
else:
    '''Init model net'''
    if model_name == "vin":
        net = VIN(num_actions, obs_dim, model_config)
    elif model_name == "rvin":
        net = RVIN(num_actions, obs_dim, model_config)
    elif model_name == "cnn":
        net = Grid_RL_Conv(num_actions, obs_dim, model_config)
    elif model_name == "crnn":
        net = Grid_RL_Recur(num_actions, obs_dim, model_config)
    elif model_name == "gnn":
        net = GNN(num_actions, obs_dim, model_config)
    elif model_name == "alpha_net":
        net = Alpha_Net(num_actions, obs_dim, model_config)
    elif model_name == "alpha_net_wstep":
        net = Alpha_Net_WStep(num_actions, obs_dim, model_config)
    else:
        print(DASH)
        print(str(model_name) + " is an invalid model.")
        print(DASH)
        sys.exit(1)
    print(net)

    '''Init policy'''
    sim = SuperGrid_Sim(env._obs_dim, env_config)
    if policy_name == "vdn":
        policy = VDN(net, num_actions, obs_dim, numrobot, policy_config,
                     model_path=model_path)
    elif policy_name == "drqn":
        policy = DRQN(sim, net, num_actions, obs_dim, policy_config,
                      model_config, model_path=model_path)
    elif policy_name == "dqn":
        policy = DQN(net, num_actions, obs_dim, policy_config,
                     model_path=model_path)
    elif policy_name == "alphazero":
        policy = AlphaZero(env, sim, net, num_actions, obs_dim, policy_config,
                           model_path=model_path)
    elif policy_name == "ha_star":
        sim = SuperGrid_Sim(env._obs_dim, env_config)
        policy = HA_Star(env, sim, net, num_actions, obs_dim, logger,
                         policy_config, model_path=model_path)
    else:
        print(DASH)
        print(str(policy_name) + " is an invalid policy.")
        print(DASH)
        sys.exit(1)

# train a policy if not testing a saved model
if not saved_model:
    '''Train policy'''
    if not learning_policy:
        print("----------Running {} for ".format(policy_name)
              + str(train_episodes) + " episodes-----------")
        policy.printNumParams()

        train_rewardlis, losslist, test_percent_covered = train_RLalg(env,
                                                                      policy, logger, train_episodes=train_episodes,
                                                                      test_episodes=test_episodes,
                                                                      render=render_train, ignore_done=ignore_done)
    else:
        print("---------------------Running Random Policy---------------------")


'''Test policy'''
print("-----------------------------Testing Policy----------------------------")
test_rewardlis, average_percent_covered = test_RLalg(env, policy, logger,
                                                     episodes=50,
                                                     render_test=render_test,
                                                     makevid=makevid)
test_percent_covered.append(average_percent_covered)


# get max and average coverage across all tests
max_coverage = max(test_percent_covered)
avg_coverage = sum(test_percent_covered) / len(test_percent_covered)

'''Display testing results'''
print(DASH)
print("Trained policy covered " + str(max_coverage)
      + " percent of the environment on its best test!")
print("Trained policy covered " + str(avg_coverage)
      + " percent of the environment on average across all tests!")
print(DASH)

if not saved_model:
    # plot training rewards
    logger.plot(train_rewardlis, 2, "Training Reward per Episode", 'Episodes',
                'Reward', "Training Reward", "Training Reward",
                show_fig=show_fig)

    # plot training loss
    logger.plot(losslist, 3, "Training Loss per Episode", 'Episodes', 'Loss',
                "Training Loss", "Training Loss", show_fig=show_fig)

    # plot testing rewards
    logger.plot(test_rewardlis, 4, "Testing Reward per Episode", 'Episodes',
                'Reward', "Testing Reward", "Testing Reward", show_fig=show_fig)

    # plot average percent covered when testing
    logger.plot(test_percent_covered, 5, "Average Percent Covered",
                'Episode (x10)', 'Percent Covered', "Percent Covered",
                "Percent Covered", show_fig=show_fig)

#closing logger
logger.close()
