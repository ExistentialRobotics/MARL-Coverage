import numpy as np

class ReplayBuffer(object):
    '''
    Stores state transitions (state, action, reward, next state)
    '''
    def __init__(self, obs_dim, act_dim, size, graph_dim=None):
        super().__init__()
        self._maxsize = size
        self._size = 0

        #creating all arrays
        self._state = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self._action = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self._reward = np.zeros(size, dtype=np.float32)
        self._nextstate = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self._done = np.zeros(size, dtype=np.float32)

        self._store_graph = False
        if graph_dim is not None:
            self._graph = np.zeros(combined_shape(size, graph_dim), dtype=np.float32)
            self._next_graph = np.zeros(combined_shape(size, graph_dim), dtype=np.float32)
            self._store_graph = True

        #creating episode tracking lists
        self._start_lis = []
        self._len_lis = []

        #tracking where we need to write next to array
        self._ptr = 0

    def addtransition(self, state, action, reward, next_state, done,
                      graph=None, next_graph=None):
        #adding new data
        self._state[self._ptr] = state
        self._action[self._ptr] = action
        self._reward[self._ptr] = reward
        self._nextstate[self._ptr] = next_state
        self._done[self._ptr] = done
        if graph is not None:
            self._graph[self._ptr] = graph
            self._next_graph[self._ptr] = next_graph

        #incrementing size
        self._ptr += 1
        self._ptr = self._ptr % self._maxsize
        self._size = min(self._size + 1, self._maxsize)

    def addepisode(self, episode):
        #tracking episode information
        self._start_lis.append(self._ptr)
        self._len_lis.append(len(episode))

        #adding all transitions
        for i in range(len(episode)):
            if self._store_graph:
                self.addtransition(episode[i][0][0], episode[i][1], episode[i][2], episode[i][3][0], episode[i][4], episode[i][0][1], episode[i][3][1])
            else:
                self.addtransition(episode[i][0], episode[i][1], episode[i][2], episode[i][3], episode[i][4])
        #reducing the episode information list to only contain complete episodes
        while sum(self._len_lis) > self._maxsize:
            self._len_lis.pop(0)
            self._start_lis.pop(0)

    def samplebatch(self, N):
        indices = np.random.randint(0, self._size, size=N)
        if self._store_graph:
            return self._state[indices], self._action[indices], self._reward[indices], self._nextstate[indices], \
                   self._done[indices], self._graph[indices], self._next_graph[indices]
        else:
            return self._state[indices], self._action[indices], self._reward[indices], self._nextstate[indices], self._done[indices]

    def samplesequential(self, N):
        #checking for valid input
        assert N > 0, "you need to sample at least 1 step"

        #picking an episode at random (this is an approximately incorrect
        #assumption but it simplifies the coding a lot)
        ep_num = np.random.randint(0, len(self._start_lis))

        #choosing the start increment(within the episode) for sequential sampling
        start_incr = np.random.randint(0, self._len_lis[ep_num] - N + 1)
        start_ind = self._start_lis[ep_num] + start_incr

        #making the indices, mod to handle the overlap
        indices = np.remainder(np.arange(start_ind, start_ind + N), self._maxsize)

        #returning the correct samples
        if self._store_graph:
            return self._state[indices], self._action[indices], self._reward[indices], self._nextstate[indices], \
                   self._done[indices], self._graph[indices], self._next_graph[indices]
        else:
            return self._state[indices], self._action[indices], self._reward[indices], self._nextstate[indices], self._done[indices]

    def samplebatchsequential(self, batchsize, N):
        #checking for valid input
        assert N > 0, "you need to sample at least 1 step"

        #storing intermediate results
        statelis = []
        actionlis = []
        rewardlis = []
        nextstatelis = []
        donelis = []

        if self._store_graph:
            graphlis = []
            nextgraphlis = []

        #getting all the batches
        for i in range(batchsize):
            #picking an episode at random (this is an approximately incorrect
            #assumption but it simplifies the coding a lot)
            ep_num = np.random.randint(0, len(self._start_lis))

            #choosing the start increment(within the episode) for sequential sampling
            start_incr = np.random.randint(0, self._len_lis[ep_num] - N + 1)
            start_ind = self._start_lis[ep_num] + start_incr

            #making the indices, mod to handle the overlap
            indices = np.remainder(np.arange(start_ind, start_ind + N), self._maxsize)
            statelis.append(self._state[indices])
            actionlis.append(self._action[indices])
            rewardlis.append(self._reward[indices])
            nextstatelis.append(self._nextstate[indices])
            donelis.append(self._done[indices])

            if self._store_graph:
                graphlis.append(self._graph[indices])
                nextgraphlis.append(self._next_graph[indices])

        statelis = np.swapaxes(statelis, 0, 1)
        actionlis = np.swapaxes(actionlis, 0, 1)
        rewardlis = np.swapaxes(rewardlis, 0, 1)
        nextstatelis = np.swapaxes(nextstatelis, 0, 1)
        donelis = np.swapaxes(donelis, 0, 1)

        if self._store_graph:
            graphlis = np.swapaxes(graphlis, 0, 1)
            nextgraphlis = np.swapaxes(nextgraphlis, 0, 1)

        # print(statelis.shape)
        # print(actionlis.shape)
        # print(rewardlis.shape)
        # print(nextstatelis.shape)
        # print(donelis.shape)
        # print(graphlis.shape)
        # print(nextgraphlis.shape)

        if(self._store_graph):
            return statelis, actionlis, rewardlis, nextstatelis, donelis, graphlis, nextgraphlis
        return statelis, actionlis, rewardlis, nextstatelis, donelis

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)
