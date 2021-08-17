import numpy as np
import getopt, sys
import json

from Environments.super_grid_rl import SuperGridRL
from Action_Spaces.discrete import Discrete
from Policies.basic_random import Basic_Random
from Policies.vpg import VPG
from Policies.dqn import DQN
from Policies.ac import AC
from Policies.replaybuffer import ReplayBuffer
from Logger.logger import Logger
from Utils.utils import train_RLalg, test_RLalg
from Policies.Networks.grid_rl_conv import Grid_RL_Conv
import torch.nn as nn

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
exp_name       = exp_parameters["experiment_name"]
numrobot       = exp_parameters["numrobot"]
gridwidth      = exp_parameters["gridwidth"]
gridlen        = exp_parameters["gridlen"]
seed           = exp_parameters["seed"]
lr             = exp_parameters["lr"]
train_episodes = exp_parameters["train_episodes"]
test_episodes  = exp_parameters["test_episodes"]
maxsteps       = exp_parameters['maxsteps']
collision_p    = exp_parameters["collision_p"]
gamma          = exp_parameters["gamma"]

weight_decay = 0
if exp_parameters["weight_decay"] > 0:
    weight_decay = exp_parameters["weight_decay"]

makevid = False
if exp_parameters["makevid"] == 1:
    makevid = True

render_test = False
if exp_parameters["render_test"] == 1:
    render_test = True

render_train = False
if exp_parameters["render_train"] == 1:
    render_train = True

show_fig = exp_parameters["render_plots"]

conv_channels = []
for channel in exp_parameters["conv_channels"]:
    conv_channels.append(channel["_"])

conv_filters = []
for filter in exp_parameters["conv_filters"]:
    conv_filters.append((filter["_"], filter["_"]))

hidden_sizes = []
for size in exp_parameters["hidden_sizes"]:
    hidden_sizes.append(size["_"])

if exp_parameters["conv_activation"] == "relu":
    conv_activation = nn.ReLU

if exp_parameters["hidden_activation"] == "relu":
    hidden_activation = nn.ReLU


print(DASH)
print("Running experiment using: " + str(config_path))
print(DASH)

'''Init logger'''
logger = Logger(exp_name, makevid, 0.05)

'''Making the environment'''
env = SuperGridRL(numrobot, gridlen, gridwidth, maxsteps ,collision_penalty=collision_p)
num_actions = env._num_actions
obs_dim = env._obs_dim

'''Init action space'''
action_space = Discrete(num_actions)

ignore_done = False
'''Init policy'''
random_policy = False
if exp_parameters["policy_type"] == "random":
    policy = Basic_Random(action_space)
    random_policy = True
else:
    #creating neural net, same constructor params for both vpg and dqn
    net = Grid_RL_Conv(num_actions, obs_dim, conv_channels, conv_filters,
                         conv_activation, hidden_sizes, hidden_activation)

    if exp_parameters["policy_type"] == "vpg":
        # init vpg policy
        policy = VPG(net, numrobot, action_space, lr,
                     weight_decay=weight_decay, gamma=gamma,
                     model_path=model_path)
    elif exp_parameters["policy_type"] == "dqn":
        #determines batch size for q-network
        batch_size = None
        if exp_parameters["batch_size"] > 0:
            batch_size = exp_parameters["batch_size"]

        #dqn specific parameters
        tau            = exp_parameters["tau"]
        buffer_maxsize = exp_parameters["buffer_size"]
        ignore_done    = exp_parameters['ignore_done']

        #creating buffer
        buff = ReplayBuffer(obs_dim, None, buffer_maxsize)

        # init policy
        policy = DQN(net, buff, num_actions, lr, batch_size=batch_size,
                     model_path=model_path, weight_decay=weight_decay,
                     gamma=gamma, tau=tau)
    elif exp_parameters["policy_type"] == "ac":
        #ac specific params
        gae = False
        if exp_parameters["GAE"] > 0:
            gae = True

        # init critic using same structure as actor
        critic = Grid_RL_Conv(1, obs_dim, conv_channels, conv_filters,
                             conv_activation, hidden_sizes, hidden_activation)

        # init policy
        policy = AC(net, critic, numrobot, action_space, lr,
                     weight_decay=weight_decay, model_path=model_path, gae=gae)

# train a policy if not testing a saved model
if not saved_model:
    '''Train policy'''
    train_rewardlis = []
    losslist = []
    if not random_policy:
        print("----------Running {} for ".format(exp_parameters["policy_type"]) + str(train_episodes) + " episodes-----------")
        policy.printNumParams()
        train_rewardlis, losslist, test_percent_covered = train_RLalg(env, policy, logger, episodes=train_episodes,
                                                                      render=render_train, ignore_done=ignore_done)
    else:
        print("-----------------------Running Random Policy-----------------------")

'''Test policy'''
print("-----------------------------Testing Policy----------------------------")
#testing the policy and collecting data
test_rewardlis, average_percent_covered = test_RLalg(env, policy, logger, episodes=test_episodes, render_test=render_test,
                                                     makevid=makevid)
test_percent_covered.append(average_percent_covered)

'''Display results'''
print(DASH)
print("Trained policy covered " + str(average_percent_covered) + " percent of the environment on average!")
print(DASH)

if not saved_model:
    # plot training rewards
    logger.plot(train_rewardlis, 2, "Training Reward per Episode", 'Episodes', 'Reward', "Training Reward",
                "Training Reward", show_fig=show_fig)

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

# from Policies.Networks.grid_rl_conv import Grid_RL_Conv
# import torch
# import torch.nn as nn
# print(num_actions)
# print(obs_dim)
# print(conv_channels)
# print(conv_filters)
# print(conv_activation)
# print(hidden_sizes)
# print(hidden_activation)
# net = Grid_RL_Conv(64, (4, 25, 25), [64, 32], [(9, 9), (7, 7)], nn.ReLU, [500, 100], nn.ReLU)
# net.load_state_dict(torch.load('Experiments/grid_rl_dqn_arash_3/checkpoint3.pt'))
# a = net.layers._modules['11'].weight.data.numpy()
