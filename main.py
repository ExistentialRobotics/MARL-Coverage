import sys
import matplotlib.pyplot as plt
import numpy as np
import math
import json

def print_env_vars(env_vars):
    print("----------------------Printing Env Variables-----------------------")
    for e in env_vars:
        print(e + " " + str(env_vars[e]))
    print("-------------------------------------------------------------------")


if __name__ == "__main__":
    # get environment variables from json file
    with open('config.json') as f:
        env_vars = json.load(f)

    print_env_vars(env_vars)
