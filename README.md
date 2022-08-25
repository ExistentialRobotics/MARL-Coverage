# Welcome to the UCSD Existential Robotics Laboratory Coverage Repository!

This repository contains an open source (gym-esque) grid world environment to
enable researchers (or anyone interested in coverage) to develop and benchmark
robotic coverage algorithms, with software infrastructure in place to train
Deep Reinforcement Learning agents. In addition, we have implementations and
benchmarks of several non-learning based staple robotic coverage algorithms:
BSA, BA*, and Spiral STC.

Coverage Path Planning, or CPP, involves controlling a robot or group of robots 
to visit every open space in an environment. CPP can be applied in situations 
where robots need to cover large sweeps of area, such as search and rescue, 
environmental cleanup, and surveillance. It is known that the solution to the 
CPP problem corresponds to generating a Traveling Salesman Problem (TSP), a 
NP-Hard problem which entails finding a path through a graph such that every 
vertex is visited exactly once and no edge is used twice, through the 
environment's grid cells. path through the environment, which is a NP-hard 
problem. In addition, to be viable in environments where GPS is not available, 
the robots need to be able to use their on board sensors to inform their next 
decision.

By using this repository one will be able to:
  - Quickly implement, run, and evaluate new RL/non RL based coverage algorithms
  - Learn about existing coverage algorithms using our implementations
  - Alter environment dynamics as needed
  - Load maps to use for training and testing

# Installation

## Windows
First, install python by following this link and the instructions specified
there: https://www.python.org/downloads/. 

Next, install pip by first getting a copy of get-pip.py via the command line
and then running get-pip.py:
```
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py
```

We want to use a virtual environment to isolate the required packages for 
the repo, so use this command to install venv, a package used to create and
manage virtual environments:
```
pip install virtualenv
```

After that, we need to clone the repo and navigate to the right directory. To
do this, run these commands:
```
git clone https://github.com/ExistentialRobotics/MARL-Coverage.git
cd MARL_Coverage
```

Now, we want to create a virtual environment. Run this command to create the
virtual environment:
```
virtualenv coverage_env
```

Then, we need to activate the environment. Do this by running:
```
coverage_env\scripts\activate
```

Next, run the script easy_install_windows.bat to install all the required 
packages:
```
easy_install_windows
```

Last, deactivate the virtual environment:
```
deactivate
```

## Linux
First, install python executing this in a terminal:
```
sudo apt-get install python3
```

Next, install pip by running:
```
sudo apt install python3-pip
```

We want to use a virtual environment to isolate the required packages for 
the repo, so use this command to install venv, a package used to create and
manage virtual environments:
```
pip install virtualenv
```

After that, we need to clone the repo and navigate to the right directory. To
do this, run these commands:
```
git clone https://github.com/ExistentialRobotics/MARL-Coverage.git
cd MARL_Coverage
```

Now, we want to create a virtual environment. Run this command to create the
virtual environment:
```
virtualenv coverage_env
```

Then, we need to activate the environment. Do this by running:
```
source coverage_env/bin/activate
```

Next, run the script easy_install_linux.sh to install all the required 
packages:
```
bash ./easy_install_linux.sh
```

Last, deactivate the virtual environment:
```
deactivate
```

# How to Run the Code

The process for running the code is to set up a folder which holds a 
config.json file that determines the environment, policy, sensor, and types of 
grid maps used for the experiment. Additionally, the config.json file sets 
assorted other hyperparameters related to the experiment, such as the number
of training episodes (if an RL agent is trained), the neural network model 
(if an RL agent is trained), number of testing episodes, whether or not to 
record a video of the tests running, etc. In the top level directory of the 
cloned repo there is a folder called Example_Experiments, which contains 
examples of the config files used for different policy types in this repo.

First, activate the environment if it hasn't already been activated:
```
source coverage_env/bin/activate
```

Once a config file has been created, simply run:
```
python grid_rl_main.py [insert path to config.json file here]
```
For example, run this command to run an experiment using a STC policy:
```
python grid_rl_main.py ./Example_Experiments/Non_Learning/STC/Example/config.json
```
After the experiment has been completed, it will save the model weights for each
checkpoint (if running RL), graphs of the average reward per episode for 
training and testing, recordings of the policy being ran during testing, and 
any terminal output that occured. 

To create your own experiments, create a new folder that holds the config file of
your experiment and run the above command, except using the path to your newly
created config file. Feel free to check out and run the other example experiments
for the different policies in this repo!
