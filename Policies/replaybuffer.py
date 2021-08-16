import numpy as np

class ReplayBuffer(object):
    '''
    Stores state transitions (state, action, reward, next state)
    '''
    def __init__(self, obs_dim, act_dim, size):
        super().__init__()
        self._maxsize = size
        self._size = 0

        #creating all arrays
        self._state = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self._action = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self._reward = np.zeros(size, dtype=np.float32)
        self._nextstate = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self._done = np.zeros(size, dtype=np.float32)

        #tracking where we need to write next to array
        self._ptr = 0

    def addtransition(self, state, action, reward, next_state, done):
        #adding new data
        self._state[self._ptr] = state
        self._action[self._ptr] = action
        self._reward[self._ptr] = reward
        self._nextstate[self._ptr] = next_state
        self._done[self._ptr] = done

        #incrementing size
        self._ptr += 1
        self._ptr = self._ptr % self._maxsize
        self._size = min(self._size + 1, self._maxsize)

    def addepisode(self, episode):
        for i in range(len(episode)):
            self.addtransition(episode[i][0], episode[i][1], episode[i][2], episode[i][3], episode[i][4])
    def samplebatch(self, N):
        indices = np.random.randint(0, self._size, size=N)
        return self._state[indices], self._action[indices], self._reward[indices], self._nextstate[indices], self._done[indices]

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)
