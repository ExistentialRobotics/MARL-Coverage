import numpy as np
from Utils.gridmaker import gridload
from queue import PriorityQueue
from Environments.dec_grid_rl import DecGridRL
from Logger.logger import Logger
import copy


class BA_Star(object):
    '''
    Online controller that takes incremental observations of the environment and
    can achieve optimal and full coverage in certain conditions (see Shreyas'
    notes).
    '''

    def __init__(self, internal_grid_rad, egoradius):
        super().__init__()

        # describes what subcell we are currently in
        self._internal_grid_rad = internal_grid_rad

        self._egoradius = egoradius

        self.num_actions = 4

        # reset policy (creates visited array and curr_x, curr_y)
        self.reset()

    def get_obs_vis(self, state):
        # getting only the obstacle layer
        obs = copy.deepcopy(np.squeeze(state)[2])
        vis = copy.deepcopy(
            self._visited[self._curr_x - 1:self._curr_x + 2, self._curr_y - 1:self._curr_y + 2])

        # print("ori vis: " + str(vis))
        # print("ori obs: " + str(obs))

        if self._curr_a == 0:
            r = 1
        elif self._curr_a == 90:
            r = 0
        elif self._curr_a == 180:
            r = 3
        else:
            r = 2

        for i in range(r):
            vis = np.rot90(vis)

        if self._curr_a == 0:
            r = 1
        elif self._curr_a == 90:
            r = 0
        elif self._curr_a == 180:
            r = 3
        else:
            r = 2

        for i in range(r):
            obs = np.rot90(obs)

        # print("rot obs: " + str(obs))
        # print("rot vis: " + str(vis))

        return obs, vis

    def b(self, p1, p2, vis, obs):
        # print("p1: " + str(p1) + " obs: "
        #       + str(obs[p1]) + " vis: " + str(vis[p1]))
        # print("p2: " + str(p2) + " obs: "
        #       + str(obs[p2]) + " vis: " + str(vis[p2]))
        if vis[p1] == 0 and obs[p1] == 0:
            if vis[p2] == 1 or obs[p2] == 1:
                return 1
        return 0

    def is_bpoint(self, i, j, vis, obs):
        points = {
            1: (i + 1, j),
            2: (i + 1, j + 1),
            3: (i, j + 1),
            4: (i - 1, j + 1),
            5: (i - 1, j),
            6: (i - 1, j - 1),
            7: (i, j - 1),
            8: (i + 1, j - 1)
        }

        u = self.b(points[1], points[2], vis, obs) + \
            self.b(points[1], points[8], vis, obs) + \
            self.b(points[3], points[2], vis, obs) + \
            self.b(points[3], points[4], vis, obs) + \
            self.b(points[5], points[4], vis, obs) + \
            self.b(points[5], points[6], vis, obs) + \
            self.b(points[7], points[6], vis, obs) + \
            self.b(points[7], points[8], vis, obs)

        if u >= 1:
            return True
        return False

    def backtracking_point(self, vis, obs, start):
        backtracking_list = []
        for i in range(1, vis.shape[0]):
            for j in range(1, vis.shape[1]):
                # print("is bpoint: " + str((i, j)))
                if vis[i, j] == 1 and self.is_bpoint(i, j, vis, obs):
                    backtracking_list.append((i, j))

        goal = None
        m = (vis.shape[0] * vis.shape[1]) + 1
        # print(backtracking_list)
        for p in backtracking_list:
            dist = np.linalg.norm(np.array(p) - np.array(start))
            if dist < m:
                goal = p
                m = dist

        return goal

    def pi(self, state):
        # print("------")

        obs, vis = self.get_obs_vis(state)

        # print("current angle: " + str(self._curr_a))
        # print("frontiering: " + str(self._frontiering))
        # print("turning: " + str(self._turning))

        # update the robot's maps
        self.update_maps(state)

        # check if we need to turn
        init_boustro = False
        if (obs[1, 2] == 1 or vis[1, 2] == 1) and self._turning is False and self._frontiering is False:
            # print("attempting to init turn, prev turn: " + str(self._prev_turn))

            # check if robot is able to turn, if not init frontier
            if self._prev_turn == "right":
                # check left if we last turned right
                if obs[0, 1] == 1 or vis[0, 1] == 1:
                    self.init_frontier()
            else:
                # check right if we last turned left
                if obs[2, 1] == 1 or vis[2, 1] == 1:
                    self.init_frontier()

            # if able to turn, init turn
            if self._frontiering is False:
                self.init_turn()

        # print("frontiering: " + str(self._frontiering))
        # print("turning: " + str(self._turning))

        # turn
        if self._turning:
            # keep turning unless obstacle in way
            if self._prev_turn == "right":
                if obs[2, 1] == 1 or vis[2, 1] == 1:
                    self.init_frontier()
                else:
                    self.turn_right()
            else:
                if obs[0, 1] == 1 or vis[0, 1] == 1:
                    self.init_frontier()
                else:
                    self.turn_left()

            # reset turning boolean if we are done turning or frontier started
            if (self._init_a + 180) % 360 == self._curr_a or self._frontiering:
                self._turning = False

        # print("frontiering: " + str(self._frontiering))
        # print("turning: " + str(self._turning))

        # execute frontier based exploration
        if self._frontiering:
            if self._goal_point == (self._curr_x, self._curr_y):
                self._frontiering = False
                self._path_map = None
                self._goal_point = None
                self._prev_a = self.to_open()
                # print("Reached goal point! Action to open space: "
                #       + str(self._prev_a))
                if self._prev_a == -1:
                    self.init_frontier()
                    self._prev_a = self.frontier_based()
                else:
                    init_boustro = True
                    # self.init_boustro(state)
            else:
                self._prev_a = self.frontier_based()
                # print("frontier output: " + str(self._prev_a))

        u = self._prev_a

        # update robot's controls
        self.update_controls(u)

        # initialize boustrophedon motion if necessary
        if init_boustro:
            self.init_boustro(state)

        return u

    def to_open(self):
        if self._obstacles[self._curr_x + 1, self._curr_y] == 0 and self._free[self._curr_x + 1, self._curr_y] == 0 and self._visited[self._curr_x + 1, self._curr_y] == 0:
            u = 0
        elif self._obstacles[self._curr_x - 1, self._curr_y] == 0 and self._free[self._curr_x - 1, self._curr_y] == 0 and self._visited[self._curr_x - 1, self._curr_y] == 0:
            u = 2
        elif self._obstacles[self._curr_x, self._curr_y + 1] == 0 and self._free[self._curr_x, self._curr_y + 1] == 0 and self._visited[self._curr_x, self._curr_y + 1] == 0:
            u = 1
        elif self._obstacles[self._curr_x, self._curr_y - 1] == 0 and self._free[self._curr_x, self._curr_y - 1] == 0 and self._visited[self._curr_x, self._curr_y - 1] == 0:
            u = 3
        else:
            u = -1
            print("No open spaces!")

        if u != -1:
            if u == 0:
                self._curr_a = 0
            elif u == 1:
                self._curr_a = 90
            elif u == 2:
                self._curr_a = 180
            else:
                self._curr_a = 270

        return u

    def init_boustro(self, state):
        # if self._visited[self._curr_x + 1, self._curr_y] == 0 and self._obstacles[self._curr_x + 1, self._curr_y] == 0:
        #     self._curr_a = 0
        # elif self._visited[self._curr_x, self._curr_y + 1] == 0 and self._obstacles[self._curr_x, self._curr_y + 1] == 0:
        #     self._curr_a = 90
        # elif self._visited[self._curr_x - 1, self._curr_y] == 0 and self._obstacles[self._curr_x - 1, self._curr_y] == 0:
        #     self._curr_a = 180
        # else:
        #     self._curr_a = 270

        obs, vis = self.get_obs_vis(state)

        # print("init boustro obs: " + str(obs))
        # print("init boustro vis: " + str(vis))

        if vis[0, 1] == 1 or obs[0, 1] == 1:
            self._prev_turn = "left"
        else:
            self._prev_turn = "right"

    def init_turn(self):
        if self._prev_turn == "right":
            self._prev_turn = "left"
        else:
            self._prev_turn = "right"
        # print("initializing turn! setting prev turn to: " + str(self._prev_turn))
        self._turning = True
        self._init_a = self._curr_a

    def init_frontier(self):
        # init frontier based and flip 0s and 1s
        free_copy = copy.deepcopy(self._visited)
        free_copy = np.bitwise_not(
            free_copy.astype('?')).astype(np.uint8)
        free_copy = free_copy.astype('int')

        # get explored positions
        exp_inds = np.argwhere(free_copy > 0)

        # mark not free positions as obstacles
        free_copy[exp_inds[:, 0], exp_inds[:, 1]] = -1

        # get obstacle positions
        obs_inds = np.argwhere(self._obstacles > 0)

        # mark obstacle positions
        free_copy[obs_inds[:, 0], obs_inds[:, 1]] = -1

        # get closest corner point
        self._goal_point = self.backtracking_point(
            self._visited, self._obstacles, (self._curr_x, self._curr_y))

        # generate path to corner point
        self._path_map = self.dijkstra_path_map(
            free_copy, self._curr_x, self._curr_y, self._goal_point)

        # set status to frontiering
        self._frontiering = True

        # print("visited: " + str(self._visited))
        # print(self._free)
        # print("obstacles: " + str(self._obstacles))
        # print("goal point: " + str(self._goal_point))
        # print("path map: " + str(self._path_map))
        # print("starting frontier based!")

    def turn_left(self):
        # print("turning left!")
        self._curr_a = (self._curr_a + 90) % 360
        self._prev_a = (self._prev_a + 1) % 4
        self._prev_turn = "left"

    def turn_right(self):
        # print("turning right!")
        self._curr_a -= 90
        if self._curr_a < 0:
            self._curr_a = 270
        self._prev_a -= 1
        if self._prev_a < 0:
            self._prev_a = 3
        self._prev_turn = "right"

    def update_controls(self, u):
        # updating robot x, y based on controls
        if u == 0:
            self._curr_x += 1
        elif u == 1:
            self._curr_y += 1
        elif u == 2:
            self._curr_x -= 1
        elif u == 3:
            self._curr_y -= 1

        # print("pos: " + str((self._curr_x, self._curr_y)))

    def update_maps(self, state):
        pos_map, free, obs, dist = state[0]

        # print("free: " + str(free))
        # print("obs: " + str(obs))

        # visiting the current cell
        self._visited[self._curr_x][self._curr_y] = 1

        # update free and obstacle maps
        for i in range(self._curr_x - self._egoradius, self._curr_x + self._egoradius + 1):
            for j in range(self._curr_y - self._egoradius, self._curr_y + self._egoradius + 1):
                m = i - self._curr_x + self._egoradius
                n = j - self._curr_y + self._egoradius

                self._free[i, j] = free[m, n]
                self._obstacles[i, j] = obs[m, n]

    def frontier_based(self):
        pos = (self._curr_x, self._curr_y)
        # print("path map inside frontier: " + str(self._path_map))

        # determine action
        u = -1
        for i in range(self.num_actions):
            if i == 0:
                p = (pos[0] + 1, pos[1])
            elif i == 1:
                p = (pos[0], pos[1] + 1)
            elif i == 2:
                p = (pos[0] - 1, pos[1])
            elif i == 3:
                p = (pos[0], pos[1] - 1)
            # print("p: " + str(p))
            if self._path_map[p] == 1:
                self._path_map[pos] = 0
                u = i
                break

        if u == -1:
            u = 0

        if u == 0:
            self._curr_a = 0
        elif u == 1:
            self._curr_a = 90
        elif u == 2:
            self._curr_a = 180
        else:
            self._curr_a = 270

        return u

    def in_bounds(self, x, y, grid):
        return x >= 0 and y >= 0 and x < grid.shape[0] and y < grid.shape[1]

    def get_valid_neighbors(self, x, y, grid, visited):
        """
        Args:
           x : the x position on the grid
           y : the y position
           grid : an array representing the environment, 0 is explored,
                 1 is unexplored, and -1 is obstacle
           visited: an array representing whether we have visited a cell
                    or not, 1 is visited, 0 is not visited

        Returns:
           A list of valid neighbors and their coordinates
        """
        neighbors = []

        if self.in_bounds(x+1, y, grid) and visited[x+1][y] == 0 and grid[x+1][y] != -1:
            neighbors.append((x+1, y))

        if self.in_bounds(x-1, y, grid) and visited[x-1][y] == 0 and grid[x-1][y] != -1:
            neighbors.append((x-1, y))

        if self.in_bounds(x, y+1, grid) and visited[x][y+1] == 0 and grid[x][y+1] != -1:
            neighbors.append((x, y+1))

        if self.in_bounds(x, y-1, grid) and visited[x][y-1] == 0 and grid[x][y-1] != -1:
            neighbors.append((x, y-1))

        return neighbors

    def dijkstra_path_map(self, grid, start_x, start_y, end_point):
        """
        Args:
           grid : an array representing the environment, 1 is explored,
                 0 is unexplored, and -1 is obstacle
           start_x : starting x position
           start_y : starting y position

        Returns:
           an array showing the shortest path to an unexplored cell

        """
        open_set = PriorityQueue()
        visited = np.zeros(grid.shape)
        cost = -1*np.ones(grid.shape)

        # adding starting point to the open set
        open_set.put((0, (start_x, start_y)))

        # main dijkstra loop
        while not open_set.empty():
            cell = open_set.get()

            # check if cell has already been visited
            if visited[cell[1][0]][cell[1][1]] == 1:
                continue

            # mark as visited, finalize cost
            visited[cell[1][0]][cell[1][1]] = 1
            cost[cell[1][0]][cell[1][1]] = cell[0]

            # checking if cell is unexplored
            if cell[1] == end_point:
                break

            # looping over all neighbors and updating their costs
            neighbors = self.get_valid_neighbors(
                cell[1][0], cell[1][1], grid, visited)

            for neighbor in neighbors:
                open_set.put((cell[0] + 1, neighbor))

        # using cost array to make optimal path
        path_array = np.zeros(grid.shape)

        # handling case where we can't reach any unexplored points
        if end_point is None:
            return path_array

        curr = end_point
        curr_cost = cost[curr[0], curr[1]]
        path_array[curr[0], curr[1]] = 1

        # reset visited to not interfere with neighbor check
        visited = 1 - visited

        while curr[0] != start_x or curr[1] != start_y:
            neighbors = self.get_valid_neighbors(
                curr[0], curr[1], grid, visited)

            # finding the neighbor with the minimum cost
            for neighbor in neighbors:
                if cost[neighbor[0], neighbor[1]] == curr_cost - 1:
                    curr_cost -= 1
                    curr = neighbor
                    break

            # adding the current cell to the path
            path_array[curr[0], curr[1]] = 1

        assert path_array[start_x, start_y] == 1

        return path_array

    def reset(self):
        '''
        resets the policy to run again on a different environment
        '''
        internal_grid_rad = self._internal_grid_rad

        # internal coordinate system for keeping track of where we have been
        self._visited = np.zeros((2*internal_grid_rad, 2*internal_grid_rad))

        # internal coordinate system for tracking observed cells and obstacles
        self._obstacles = np.zeros((2*internal_grid_rad, 2*internal_grid_rad))
        self._free = np.zeros((2*internal_grid_rad, 2*internal_grid_rad))

        # setting the starting x,y
        index = int((internal_grid_rad + internal_grid_rad % 2)/2 - 1)

        self._curr_x = 2*index
        self._curr_y = 2*index

        self._curr_a = 0
        self._prev_a = 0
        self._init_a = 0

        self._turning = False
        self._frontiering = False
        self._prev_turn = "right"

        self._path_map = None
        self._goal_point = None

        # visiting the current cell
        self._visited[self._curr_x][self._curr_y] = 1


if __name__ == "__main__":
    # testing spanning tree coverage on dec_grid_rl environment
    env_config = {
        "numrobot": 1,
        "maxsteps": 10000,
        "collision_penalty": 5,
        "egoradius": 1,
        "done_thresh": 1,
        "done_incr": 0,
        "terminal_reward": 30,
        "mini_map_rad": 0,
        "comm_radius": 0,
        "allow_comm": 0,
        "map_sharing": 0,
        "single_square_tool": 1,
        "dist_reward": 0,
        "dijkstra_input": 1,
        "sensor_type": "square_sensor",
        "sensor_config": {
            "range": 1
            }
    }

    grid_config = {
        "grid_dir": "./Grids/bg2_100x100",
        "gridwidth": 100,
        "gridlen": 100,
        "numgrids": 30,
        "prob_obst": 0
    }

    '''Making the list of grids'''
    gridlis = gridload(grid_config)
    # train_set, test_set = gridload(grid_config)

    env = DecGridRL(gridlis, env_config)

    # logger stuff
    makevid = True
    exp_name = "stcEmptyGrid1"
    logger = Logger(exp_name, makevid)

    # testing bsa
    bsa_controller = BA_Star(105, env_config["egoradius"])

    state = env.reset()  # getting only the obstacle layer
    done = False
    render = True

    # simulating
    while not done:
        # determine action
        action = bsa_controller.pi(state)

        # step environment and save episode results
        state, reward = env.step(action)

        # determine if episode is completed
        done = env.done()

        # render if necessary
        if render:
            frame = env.render()
            if(makevid):
                logger.addFrame(frame)
