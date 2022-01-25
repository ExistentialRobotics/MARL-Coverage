import numpy as np
import sys

class Node(object):
    def __init__(self, num_action):
        self.num_action = num_action
        self.Q = np.zeros(num_action) #q-values for each action
        self.P = np.zeros(num_action) #prior probabilities for each action
        self.N = np.zeros(num_action) #visit count for each action
        self.ucb_table = np.zeros(num_action)
        self.num_visits = 0 #visit count for the node in total
        self.children = {} #dict of this nodes children
        # self.depth = depth #depth of the node in the search tree


        # value = -(number of free cells remaining * constant + steps taken)

    def sampleUCB(self):
        max = -sys.maxint - 1
        for i in range(self.num_action):
            ucb = self.Q[i] + c * self.P[i] * math.sqrt(self.num_visits / (self.N[i] + 1))
            self.ucb_table = ucb
            if ucb > max:
                best_action = action
                max = ucb
        return action, ucb

    def updateQ(self, u, v):
        #TODO updates Q of action u with value v
        pass

class MCTS(object):
    def __init__(self, env, policy, value, mcts_sims):
        super().__init__()
        self.nodes = {} #hashmap taking states to nodes
        self.num_action = env._num_actions #number of actions from each state
        self.env = env #env class
        self.mcts_sims = mcts_sims #number of mcts searches
        self.policy = policy #policy function
        self.value = value #value function

    def pi(self, x0):
        '''
        Returns the action probabilities from the current state x0 after
        doing num_sims iterations of MCTS
        '''
        #doing all MCTS searches
        for i in range(self.mcts_sims):
            self.search(x0)

        #getting first node visit counts to find probs
        node = self.nodes[str(x0)]

        #TODO should we do this or softmax, I think they did softmax in the paper
        probs = (1.0/np.sum(node.N))*node.N
        return probs

    def search(self, x0):
        '''
        Recursively performs an iteration of MCTS by performing UCB sampling
        until a leaf node is reached, where the value function is used to
        evaluate the state and update the action-values of all nodes on the path
        '''
        str_x0 = str(x0) #used as key in hashmaps

        #check whether we have been to this node before
        if str_x0 in self.nodes:
            #internal node
            node = self.nodes[str_x0]

            #getting best action
            u = node.sampleUCB()

            #state transition
            x = env.step(x0, u)
            v = self.search(x)

            #update state-action visit count
            node.N[u] += 1

            #update Q value of node
            node.updateQ(u,v)

            return v

        elif self.env.isTerminal(x0):
            #checking if x0 is terminal
            return 0 #no more steps needed to finish

        else:
            #leaf node
            node = Node(self.num_action)

            #getting policy/value
            probs = self.policy(x0)
            v = self.value(x0)

            #TODO enforcing probs are nonzero for legal actions (we don't need to do this but it would help)

            #updating node probs
            node.P = probs

            return v
