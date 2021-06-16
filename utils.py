import numpy as np
import matplotlib.pyplot as plt

def print_env_vars(env_vars):
    print("----------------------Printing Env Variables-----------------------")
    for e in env_vars:
        print(e + " " + str(env_vars[e]))
    print("-------------------------------------------------------------------")


def print_agent_coords(agents):
    print("-----------------------Printing Agent Coords-----------------------")
    for a in agents:
        print("x: " + str(a.pos[0]) + " y: " + str(a.pos[1]))
    print("-------------------------------------------------------------------")


def plot_agent_voronoi(agents):
    pass
