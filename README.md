# Welcome to the UCSD Existential Robotics Laboratory Coverage Repository!

This repository contains an open source (gym-esque) grid world environment to
enable researchers (or anyone interested in coverage) to develop and benchmark
robotic coverage algorithms, with software infrastructure in place to train
Deep Reinforcement Learning agents. In addition, we have implementations and
benchmarks of several non-learning based staple robotic coverage algorithms:
BSA, BA*, and Spiral STC.

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
