"""
utils.py contains debugging methods meann't to print certain values to the
console window

Authors: Peter Stratton, Hannah Hui
Emails: pstratto@ucsd.edu, hahui@ucsd.edu
"""

import numpy as np
import matplotlib.pyplot as plt


def print_env_vars(env_vars):
    """
    print_env_vars prints the program's environment variables to the console
    window.

    Parameter
    ---------
    env_vars : dictionary of environment variables to print
    """
    print("----------------------Printing Env Variables-----------------------")
    for e in env_vars:
        print(e + " " + str(env_vars[e]))
    print("-------------------------------------------------------------------")


def print_agent_coords(agents):
    """
    print_agent_coords prints each agent's position to the console window.

    Parameter
    ---------
    agents : list of agents who's positions need printing
    """
    print("-----------------------Printing Agent Coords-----------------------")
    for a in agents:
        print("x: " + str(a.pos[0]) + " y: " + str(a.pos[1]))
    print("-------------------------------------------------------------------")


def print_agent_centroids(agents):
    """
    print_agent_centroids prints each agent's estimated centroid position to the
    console window.

    Parameter
    ---------
    agents : list of agents who's estimated centroid positions need printing
    """
    print("---------------------Printing Agent Centroids----------------------")
    for a in agents:
        print("x: " + str(a.e_centroid[0]) + " y: " + str(a.e_centroid[1]))
    print("-------------------------------------------------------------------")


def print_agent_voronoi(agent):
    """
    print_agent_voronoi prints an agent's grid cells corresponding to its
    voronoi region to the console window.

    Parameter
    ---------
    agent : agent who's voronoi partition is printed
    """
    print("-----------------Printing Agent Voronoi Partitions-----------------")
    print("Agent: x = " + str(agent.pos[0]) + " y = " + str(agent.pos[1]))
    for v in agent.v_part_list:
        print("x: " + str(v[0]) + " y: " + str(v[1]))
    print("-------------------------------------------------------------------")


def print_agent_params(agents):
    print("----------------Printing Agent Estimated Parameters----------------")
    for agent in agents:
        print("Agent: x = " + str(agent.pos[0]) + " y = " + str(agent.pos[1]))
        print("a: " + str(agent.a_est))
        print("---------------------------------------------------------------")
