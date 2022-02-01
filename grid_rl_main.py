import numpy as np
import getopt, sys
import json

from Environments.super_grid_rl import SuperGridRL
from Environments.sim_env import SuperGrid_Sim
from Environments.dec_grid_rl import DecGridRL
from Action_Spaces.discrete import Discrete
from Policies.basic_random import Basic_Random
from Policies.dqn import DQN
from Policies.drqn import DRQN
from Policies.vdn import VDN
from Policies.alphazero import AlphaZero
from Policies.ha_star import HA_Star
from Logger.logger import Logger
from Utils.utils import train_RLalg, test_RLalg
from Policies.Networks.grid_rl_conv import Grid_RL_Conv
from Policies.Networks.grid_rl_recur import Grid_RL_Recur
from Policies.Networks.gnn import GNN
from Policies.Networks.vin import VIN
from Policies.Networks.rvin import RVIN
from Policies.Networks.alpha_net import Alpha_Net
from Utils.gridmaker import gridgen, gridload


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
try:
    # Parsing argument
    arguments, values = getopt.getopt(argumentList, options, long_options)

    # checking each argument
    for currentArgument, currentValue in arguments:

        if currentArgument in ("-m", "--model"):
            print (("Running saved model directly from % s") % (currentValue))
            model_path = currentValue
            saved_model = True
        else:
            print("Not a valid arg.")

except getopt.error as err:
    # output error, and return with an error code
    print (str(err))

if saved_model:
    # run testing with a saved model
    random_policy = False

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
    config_path = model_path[:i + 1] + "config.json"
else:
    if len(sys.argv) != 2:
        print(DASH)
        print("No config file specified.")
        print(DASH)
        sys.exit(1)
    config_path = sys.argv[1]

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
env_name       = exp_parameters["env_name"]
env_config     = exp_parameters['env_config']
numrobot       = env_config['numrobot']

'''Experiment Parameters'''
exp_name       = exp_parameters["exp_name"]
exp_config     = exp_parameters['exp_config']
train_episodes = exp_config["train_episodes"]
test_episodes  = exp_config["test_episodes"]
show_fig       = exp_config["render_plots"]
render_test    = exp_config['render_test']
render_train   = exp_config['render_train']
makevid        = exp_config["makevid"]
ignore_done    = exp_config['ignore_done']

'''Model Parameters'''
model_config   = exp_parameters['model_config']
model_name     = model_config['model_name']

'''Policy Parameters'''
policy_config  = exp_parameters['policy_config']
policy_name    = policy_config['policy_name']


print(DASH)
print("Running experiment using: " + str(config_path))
print(DASH)

'''Init logger'''
logger = Logger(exp_name, makevid)

'''Making the list of grids'''
if exp_parameters['gridload']:
    gridlis = gridload(exp_parameters['grid_config'])
else:
    gridlis = gridgen(exp_parameters['grid_config'])

'''Making the environment'''
if env_name == 'SuperGridRL':
    env = SuperGridRL(gridlis, env_config)
elif env_name == 'DecGridRL':
    env = DecGridRL(gridlis, env_config, use_graph=policy_config["use_graph"])

num_actions = env._num_actions
obs_dim = env._obs_dim

'''Init action space'''
action_space = Discrete(num_actions)

'''Init policy'''
random_policy = False
if policy_name == "random":
    policy = Basic_Random(action_space)
    random_policy = True
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
    else:
        print(DASH)
        print(str(model_name) + " is an invalid model.")
        print(DASH)
        sys.exit(1)
    print(net)

    '''Init policy'''
    if policy_name == "vdn":
        policy = VDN(net, num_actions, obs_dim, numrobot, policy_config,
                     model_path=model_path)
    elif policy_name == "drqn":
        policy = DRQN(net, num_actions, obs_dim, policy_config, model_config,
                      model_path=model_path)
    elif policy_name == "dqn":
        policy = DQN(net, num_actions, obs_dim, policy_config,
                     model_path=model_path)
    elif policy_name == "alphazero":
        sim = SuperGrid_Sim(env._grid, env._obs_dim, env_config)
        policy = AlphaZero(env, sim, net, num_actions, obs_dim, policy_config,
                     model_path=model_path)
    elif policy_name == "ha_star":
        sim = SuperGrid_Sim(env._grid, env._obs_dim, env_config)
        policy = HA_Star(env, sim, net, num_actions, obs_dim, logger, policy_config,
                     model_path=model_path)
    else:
        print(DASH)
        print(str(policy_name) + " is an invalid policy.")
        print(DASH)
        sys.exit(1)

# train a policy if not testing a saved model
# if not saved_model and policy_name != "alphazero" and policy_name != "ha_star":
if not saved_model:
    '''Train policy'''
    if not random_policy:
        print("----------Running {} for ".format(policy_name) + str(train_episodes) + " episodes-----------")
        policy.printNumParams()

        train_rewardlis, losslist, test_percent_covered = train_RLalg(env, policy, logger, train_episodes=train_episodes, test_episodes=test_episodes,
                                                                        render=render_train, ignore_done=ignore_done)
    else:
        print("-----------------------Running Random Policy-----------------------")
# else:
#     print("----------Training " + str(policy_name) + "----------")
#     policy.train()

'''Test policy'''
print("-----------------------------Testing Policy----------------------------")
# if policy_name != "alphazero" and policy_name != "ha_star":
test_rewardlis, average_percent_covered = test_RLalg(env, policy, logger, episodes=test_episodes, render_test=render_test,
                                                     makevid=makevid)
test_percent_covered.append(average_percent_covered)
# else:
#     episode = policy.rollout()
#     policy.simulate(episode)

# get max and average coverage across all tests
max_coverage = max(test_percent_covered)
avg_coverage = sum(test_percent_covered) / len(test_percent_covered)

'''Display testing results'''
print(DASH)
print("Trained policy covered " + str(max_coverage) + " percent of the environment on its best test!")
print("Trained policy covered " + str(avg_coverage) + " percent of the environment on average across all tests!")
print(DASH)

if not saved_model:
    # plot training rewards
    logger.plot(train_rewardlis, 2, "Training Reward per Episode", 'Episodes',
                'Reward', "Training Reward", "Training Reward",
                show_fig=show_fig)

    # plot training loss
    logger.plot(losslist, 3, "Training Loss per Episode", 'Episodes', 'Loss', "Training Loss",
                "Training Loss", show_fig=show_fig)

    # plot testing rewards
    logger.plot(test_rewardlis, 4, "Testing Reward per Episode", 'Episodes', 'Reward', "Testing Reward"
                , "Testing Reward", show_fig=show_fig)

    # plot average percent covered when testing
    logger.plot(test_percent_covered, 5, "Average Percent Covered", 'Episode (x10)', 'Percent Covered', "Percent Covered"
                , "Percent Covered", show_fig=show_fig)

#closing logger
logger.close()
