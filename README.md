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

# How to Run the Code

- Clone the Repository
- Create a virtual environment and install all dependencies in requirements.txt
- Install ffmpeg for video logging
- edit main.py to use the controller you want with the parameters you want
- run main.py


# Results

- All results are currently located in the tests directory, grouped by controller
- TODO: improve logging and generation of results
