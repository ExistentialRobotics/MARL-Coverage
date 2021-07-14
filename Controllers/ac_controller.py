import numpy as np
from MARL-Coverage.Environments.grid_environment

# path to config file used to load hyperparameters
CONFIG = 'Consensus_Experiments/Experiment_4/config.json'

class AC_Controller(Controller):

    def __init__(self, env_vars):
        # agent color options
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        # set environment variables as controller instance variables
        self.num_agents = env_vars["num_agents"]
        self.num_basis_fx = env_vars["num_basis_functions"]
        self.basis_sigma = env_vars["basis_sigma"]
        self.map_width = env_vars["map_width"]
        self.map_height = env_vars["map_height"]
        self.grid_cell_size = env_vars["grid_cell_size"]
        self.min_a = env_vars["min_a"]
        self.gain_const = env_vars["gain_const"]
        self.gamma_const = env_vars["gamma_const"]
        self.lr_gain = env_vars["lr_gain"]
        self.consensus_gain = env_vars["positive_consensus_gain"]
        self.pos_dim = env_vars["pos_dim"]
        self.dt = env_vars["dt"]
        self.iters = env_vars["iters"]

        # determine whether or not to use consensus
        self.consensus = False
        if env_vars["consensus"] == 1:
            self.consensus = True

        # determine whether or not to weight consensus parameters by length
        self.lw = False
        if env_vars["length_w"] == 1:
            self.lw = True

        # determine whether or not to render agents
        self.render_a = False
        if env_vars["render_a"] == 1:
            self.render_a = True

        # determine whether or not to check gaussian basis positive definiteness
        self.pdef = False
        if env_vars["posdef_check"] == 1:
            self.pdef = True

        # determine whether or not to use a data weighting function
        if env_vars["data_weighting"] == -1:
            self.d_f = 1
        else:
            self.d_f = env_vars["data_weighting"]

        # set a (change this to be in config later)
        self.opt_a = np.array([100, 100])
        if self.num_basis_fx == 9:
            self.opt_a = np.array([100, self.min_a, self.min_a, self.min_a, self.min_a, self.min_a, self.min_a, self.min_a,
                              100])

        # set gaussian basis functions and their means
        self.cov_m = self.basis_sigma * np.eye(2)
        self.means = np.array([(25, 25), (75, 75)])
        self.basis_f = [multivariate_normal(mean=np.array([25, 25]), cov=self.cov_m),
                   multivariate_normal(mean=np.array([75, 75]), cov=self.cov_m)]
        if self.num_basis_fx == 9:
            # calc basis centers assuming they're in the center of each quadrant
            self.basis_f = []
            self.means = []
            for i in range(int(np.sqrt(self.num_basis_fx))):
                y = (self.map_height / (np.sqrt(self.num_basis_fx) * 2)) + i * (self.map_height /
                     np.sqrt(self.num_basis_fx))
                for j in range(int(np.sqrt(self.num_basis_fx))):
                    x = (self.map_width / (np.sqrt(self.num_basis_fx) * 2)) + j * (self.map_width /
                         np.sqrt(self.num_basis_fx))
                    self.means.append((x, y))

                    # add multivariate normals as each basis function
                    self.basis_f.append(multivariate_normal(mean=np.array([x, y]),
                                   cov=self.cov_m))


        # init gain and gamma matricies
        self.gain_matrix = self.gain_const * np.eye(2)
        self.gamma_matrix = self.gamma_const * np.eye(a_opt.shape[0])

        # instantiate map
        self.map = self.map(self.map_width, self.map_height, self.grid_cell_size)

        # create agents with random initial positions
        self.agents = []
        for i in range(self.num_agents):
            agents.append(Agent(np.random.randint(self.map_width),
                                np.random.randint(self.map_height), self.basis_f, self.means,
                                self.min_a, self.opt_a, self.pos_dim, color=self.COLORS[2]))

        self.map = AC_Grid_Environment(self.agents, 0, self.dt, self.map_width, self.map_height, self.grid_cell_size, self.gain_matrix)

    def run(self):
        # print agent coordinates for debugging purposes
        self.print_agent_coords()

        est_errors = []
        true_errors = []
        a_errors = []
        for _ in range(iters):
            if (_ + 1) % 5 == 0:
                print("-----------------ITER: " + str(_ + 1) + "------------------")

            # potentially render agents
            if self.render_a:
                self.render_agents()

            # reset KDTree used to compute Voronoi regions
            self.map.set_tree()

            # calc centroid, mass, and moment for each agent
            self.map.set_agent_voronoi()
            self.map.calc_agent_voronoi()

            # calculate consensus terms if using consensus
            if self.consensus:
                self.map.set_consensus(length_w=lw)

            # step world
            self.map.step()

            # average estimated position errors, true position errors, and a error
            est_errors.append()
            true_errors.append()
            a_errors.append(self.map.a_error(self.agents))

            # verify if corollary 2 holds on each iteration
            if self.pdef:
                p = self.map.pdef_check()
                print("Positive definite basis functions: " + str(p))

        # plot final voronoi diagram
        self.map.plot_voronoi(self.agents)

        # print final agent paramters
        self.print_agent_params()

        # reset KDTree used to compute Voronoi regions
        self.map.set_tree(self.agents)

        # calc centroid, mass, and moment for each agent
        self.map.set_agent_voronoi(self.agents)
        self.map.calc_agent_voronoi(self.agents)

    return est_errors, true_errors, a_errors

    def render_agents(self):
        plt.clf()
        plt.title("Agent Positions (Circles) and Gaussian Centers (Xs)")
        plt.xlabel('X')
        plt.ylabel('Y')
        # render areas that self.agents should be moving towards
        plt.scatter(self.agents[0].means[0][0], self.agents[0].means[0][1], s=50,
                    c='r', marker='x')
        plt.scatter(self.agents[0].means[len(self.agents[0].means) - 1][0],
                    self.agents[0].means[len(self.agents[0].means) - 1][1], s=50,
                    c='r', marker='x')
        for agent in self.agents:
            agent.render_self()
            agent.render_centroid()
        plt.draw()
        plt.pause(0.02)

    def print_agent_coords(self):
        """
        print_agent_coords prints each agent's position to the console window.

        Parameter
        ---------
        agents : list of agents who's positions need printing
        """
        print("-----------------------Printing Agent Coords-----------------------")
        for a in self.agents:
            print("x: " + str(a.pos[0]) + " y: " + str(a.pos[1]))
        print("-------------------------------------------------------------------")

    def print_agent_params(agents):
        """
        print_agent_params prints each agent's estimated parameters.

        Parameter
        ---------
        agents : list of agents to print the parameters of
        """
        print("----------------Printing Agent Estimated Parameters----------------")
        for agent in self.agents:
            print("Agent: x = " + str(agent.pos[0]) + " y = " + str(agent.pos[1]))
            print("a: " + str(agent.a_est))
            print("---------------------------------------------------------------")

if __name__ == "__main__":
    # prevent decimal printing
    np.set_printoptions(suppress=True)

    # get environment variables from json file
    with open(CONFIG) as f:
        print("---------------------------------------------------------------")
        print("Running experiement using: " + str(CONFIG))
        print("---------------------------------------------------------------")
        env_vars = json.load(f)

    # print env vars for debugging purposes
    # print_env_vars(env_vars)

    # instanciate controller
    controller = AC_Controller(env_vars)

    # run algorithm
    est_errors, true_errors, a_errors = controller.run()

    # plot the dist from centroids per iteration
    plt.figure(4)
    plt.title("Dist from True and Estimated Centroids")
    plt.xlabel('Iterations')
    plt.ylabel('Dist')
    line_e, = plt.plot(est_errors, label="Est Centroid")
    line_t, = plt.plot(true_errors, label="True Centroid")
    plt.legend(handles=[line_e, line_t])
    plt.show()

    # plot the mean parameter error per interation
    plt.figure(3)
    plt.title("Mean Agent Parameter Error")
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    line_a, = plt.plot(a_errors, label="||a_tilde - a||")
    plt.legend(handles=[line_a])
    plt.show()
