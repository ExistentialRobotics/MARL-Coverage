import numpy as np

# path to config file used to load hyperparameters
CONFIG = 'Consensus_Experiments/Experiment_4/config.json'

class AC_Controller(Controller):

    def __init__(self, env_vars):
        # agent color options
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        # set environment variables as controller instance variables
        self.NUM_AGENTS = env_vars["num_agents"]
        self.NUM_BASIS_FX = env_vars["num_basis_functions"]
        self.BASIS_SIGMA = env_vars["basis_sigma"]
        self.MAP_WIDTH = env_vars["map_width"]
        self.MAP_HEIGHT = env_vars["map_height"]
        self.GRID_CELL_SIZE = env_vars["grid_cell_size"]
        self.MIN_A = env_vars["min_a"]
        self.GAIN_CONST = env_vars["gain_const"]
        self.GAMMA_CONST = env_vars["gamma_const"]
        self.LR_GAIN = env_vars["lr_gain"]
        selfl.CONSENSUS_GAIN = env_vars["positive_consensus_gain"]
        self.POS_DIM = env_vars["pos_dim"]
        self.DT = env_vars["dt"]
        self.ITERS = env_vars["iters"]

        # determine whether or not to use consensus
        self.CONSENSUS = False
        if env_vars["consensus"] == 1:
            self.CONSENSUS = True

        # determine whether or not to weight consensus parameters by length
        self.LW = False
        if env_vars["length_w"] == 1:
            self.LW = True

        # determine whether or not to render agents
        self.RENDER_A = False
        if env_vars["render_a"] == 1:
            self.RENDER_A = True

        # determine whether or not to check gaussian basis positive definiteness
        self.PDEF = False
        if env_vars["posdef_check"] == 1:
            self.PDEF = True

        # determine whether or not to use a data weighting function
        if env_vars["data_weighting"] == -1:
            self.D_F = 1
        else:
            self.D_F = env_vars["data_weighting"]

        # set a (change this to be in config later)
        self.OPT_A = np.array([100, 100])
        if self.NUM_BASIS_FX == 9:
            self.OPT_A = np.array([100, self.MIN_A, self.MIN_A, self.MIN_A, self.MIN_A, self.MIN_A, self.MIN_A, self.MIN_A,
                              100])

        # set gaussian basis functions and their means
        self.COV_M = self.BASIS_SIGMA * np.eye(2)
        self.MEANS = np.array([(25, 25), (75, 75)])
        self.BASIS_F = [multivariate_normal(mean=np.array([25, 25]), cov=self.COV_M),
                   multivariate_normal(mean=np.array([75, 75]), cov=self.COV_M)]
        if self.NUM_BASIS_FX == 9:
            # calc basis centers assuming they're in the center of each quadrant
            self.BASIS_F = []
            self.MEANS = []
            for i in range(int(np.sqrt(self.NUM_BASIS_FX))):
                y = (self.MAP_HEIGHT / (np.sqrt(self.NUM_BASIS_FX) * 2)) + i * (self.MAP_HEIGHT /
                     np.sqrt(self.NUM_BASIS_FX))
                for j in range(int(np.sqrt(self.NUM_BASIS_FX))):
                    x = (self.MAP_WIDTH / (np.sqrt(self.NUM_BASIS_FX) * 2)) + j * (self.MAP_WIDTH /
                         np.sqrt(self.NUM_BASIS_FX))
                    self.MEANS.append((x, y))

                    # add multivariate normals as each basis function
                    self.BASIS_F.append(multivariate_normal(mean=np.array([x, y]),
                                   cov=self.COV_M))


        # init gain and gamma matricies
        gain_matrix = self.GAIN_CONST * np.eye(2)
        gamma_matrix = self.GAMMA_CONST * np.eye(a_opt.shape[0])

    def run(self):
        # instantiate map
        map = Map(self.MAP_WIDTH, self.MAP_HEIGHT, self.GRID_CELL_SIZE)

        # create agents with random initial positions
        agents = []
        for i in range(self.NUM_AGENTS):
            agents.append(Agent(np.random.randint(self.MAP_WIDTH),
                                np.random.randint(self.MAP_HEIGHT), self.BASIS_F, self.MEANS,
                                self.MIN_A, self.OPT_A, self.POS_DIM, color=self.COLORS[2]))

        # print agent coordinates for debugging purposes
        print_agent_coords(agents)

    def render_agents(self, agents):
        plt.clf()
        plt.title("Agent Positions (Circles) and Gaussian Centers (Xs)")
        plt.xlabel('X')
        plt.ylabel('Y')
        # render areas that agents should be moving towards
        plt.scatter(agents[0].means[0][0], agents[0].means[0][1], s=50,
                    c='r', marker='x')
        plt.scatter(agents[0].means[len(agents[0].means) - 1][0],
                    agents[0].means[len(agents[0].means) - 1][1], s=50,
                    c='r', marker='x')
        for agent in agents:
            agent.render_self()
            agent.render_centroid()
        plt.draw()
        plt.pause(0.02)

    def print_agent_coords(self, agents):
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
    controller.run()

    # run adaptive coverage algorithm
    est_errors, true_errors, a_errors = adaptive_coverage(map, agents, opt_a,
                                                LR_GAIN, CONSENSUS_GAIN,
                                                GAIN_CONST, GAMMA_CONST,
                                                data_weighting=d_f, DT=DT,
                                                render_agents=RENDER_A,
                                                iters=ITERS,
                                                consensus=CONSENSUS, lw=LW,
                                                posdef_check=PDEF)

    # plot agent positions and means corresponding to highest a_opt values
    plt.figure(2)
    plt.title("Agent Positions (Circles) and Gaussian Centers (Xs)")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.scatter(agents[0].means[0][0], agents[0].means[0][1], s=50, c='r',
                marker='x')
    plt.scatter(agents[0].means[len(agents[0].means) - 1][0],
                agents[0].means[len(agents[0].means) - 1][1], s=50, c='r',
                marker='x')
    for agent in agents:
        agent.render_self()
        agent.render_centroid()

    # plot final voronoi diagram
    map.plot_voronoi(agents)

    # print final agent paramters
    print_agent_params(agents)

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
