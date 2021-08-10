import numpy as np
import matplotlib.pyplot as plt
import getopt, sys
import json

from Environments.super_grid_rl import SuperGridRL
from Controllers.grid_rl_random_controller import GridRLRandomController
from Controllers.grid_rl_controller import GridRLController
from Action_Spaces.discrete import Discrete
from Policies.basic_random import Basic_Random
from Policies.policy_gradient import PolicyGradient
from Policies.dqn import DQN
from Policies.replaybuffer import ReplayBuffer
from Logger.logger import Logger
from Utils.utils import train_RLalg, test_RLalg
import torch.nn as nn

DASH = "-----------------------------------------------------------------------"

# prevent decimal printing
np.set_printoptions(suppress=True)

'''Read config file'''
# Options
options = "m:"

# Long options
long_options = ["model ="]

# Remove 1st argument from the
# list of command line arguments
argumentList = sys.argv[1:]

saved_model = False
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
        print(str(model_path) + " does not exit.")
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
    print(str(config_path) + " does not exit.")
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
num_actions    = exp_parameters["num_actions"]
lr             = exp_parameters["lr"]
train_episodes = exp_parameters["train_episodes"]
test_episodes  = exp_parameters["test_episodes"]
train_iters    = exp_parameters["train_iters"]
test_iters     = exp_parameters["test_iters"]
collision_p    = exp_parameters["collision_p"]
buffer_maxsize = (train_episodes * train_iters) // exp_parameters["buf_divisor"]

action_net_input = False
if exp_parameters["action_net_input"] > 0:
    action_net_input = True

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

if exp_parameters["output_activation"] == "sigmoid":
    output_activation = nn.Sigmoid
elif exp_parameters["output_activation"] == "softmax":
    output_activation = nn.Softmax

print(DASH)
print("Running experiment using: " + str(config_path))
print(DASH)

'''Init logger'''
logger = Logger(exp_name, makevid, 0.02)

'''Making the environment'''
env = SuperGridRL(numrobot, gridlen, gridwidth, collision_penalty=collision_p, ani=action_net_input)

'''Init action space'''
action_space = Discrete(num_actions)

'''Init policy'''
obs_dim = np.squeeze(env.get_state(), axis=0).shape
random_policy = False
if exp_parameters["policy_type"] == "random":
    policy = Basic_Random(numrobot, action_space)
    random_policy = True
elif exp_parameters["policy_type"] == "pg":
    policy = PolicyGradient(numrobot, action_space, lr, obs_dim, conv_channels,
                            conv_filters, conv_activation, hidden_sizes,
                            hidden_activation, output_activation,
                            weight_decay=weight_decay)
elif exp_parameters["policy_type"] == "dqn":
    #TODO add other DQN parameters into config
    #could add weight_decay, gamma, tau as parameters if we want to change them
    batch_size = None
    if exp_parameters["batch_size"] > 0:
        batch_size = exp_parameters["batch_size"]
    policy = DQN(numrobot, action_space, lr, obs_dim, conv_channels,
                 conv_filters, conv_activation, hidden_sizes, hidden_activation,
                 output_activation, batch_size=batch_size, buffer_size=buffer_maxsize, ani=action_net_input)

'''Making the Controller for the Swarm Agent'''
controller = GridRLController(numrobot, policy)

# train a policy if not testing a saved model
if not saved_model:
    '''Train policy'''
    train_rewardlis = []
    losslist = []
    if not random_policy:
        print("----------Running {} for ".format(exp_parameters["policy_type"]) + str(train_episodes) + " episodes-----------")
        controller._policy.printNumParams()
        train_rewardlis, losslist = train_RLalg(env, controller, logger, episodes=train_episodes, iters=train_iters, render=render_train, ani=action_net_input)

        # plot training rewards
        plt.figure(2)
        plt.title("Training Reward per Episode")
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        line_r, = plt.plot(train_rewardlis, label="Training Reward")
        plt.legend(handles=[line_r])
        logger.savefig(plt.gcf(), 'TrainingReward')
        plt.show()
    else:
        print("-----------------------Running Random Policy-----------------------")

'''Test policy'''
print("-----------------------------Testing Policy----------------------------")

# set policy network to eval mode
if not random_policy:
    controller.set_eval()

#testing the policy and collecting data
test_rewardlis, average_percent_covered = test_RLalg(env, controller, logger, episodes=test_episodes, iters=test_iters, render_test=render_test,
                                                     make_vid=makevid, ani=action_net_input)

'''Display results'''
print(DASH)
print("Trained policy covered " + str(average_percent_covered) + " percent of the environment on average!")
print(DASH)

# plot testing rewards
plt.figure(3)
plt.title("Testing Reward per Episode")
plt.xlabel('Episodes')
plt.ylabel('Reward')
line_r, = plt.plot(test_rewardlis, label="Testing Reward")
plt.legend(handles=[line_r])
logger.savefig(plt.gcf(), 'TestingReward')
plt.show()

# plot training loss
plt.figure(4)
plt.title("Training Loss per Episode")
plt.xlabel('Episodes')
plt.ylabel('Loss')
line_r, = plt.plot(losslist, label="Training Loss")
plt.legend(handles=[line_r])
logger.savefig(plt.gcf(), 'TrainingLoss')
plt.show()

#closing logger
logger.close()
