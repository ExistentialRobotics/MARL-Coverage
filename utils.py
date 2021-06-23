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

def print_agent_centroids(agents):
    print("-----------------------Printing Agent Centroids-----------------------")
    for a in agents:
        print("x: " + str(a.e_centroid[0]) + " y: " + str(a.e_centroid[1]))
    print("-------------------------------------------------------------------")

def print_agent_voronoi(agent):
    print("-----------------------Printing Agent Voronoi Partitions-----------------------")
    print("Agent: x = " + str(agent.pos[0]) + " y = " + str(agent.pos[1]))
    for v in agent.v_part_list:
        print("x: " + str(v[0]) + " y: " + str(v[1]))
    print("-------------------------------------------------------------------")
