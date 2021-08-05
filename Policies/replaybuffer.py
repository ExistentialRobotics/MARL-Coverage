import numpy as np

class ReplayBuffer(object):
    '''
    Stores state transitions (state, action, reward, next state)
    '''
    def __init__(self, maxsize):
        super().__init__()
        self.resetbuffer()
        self._maxsize = maxsize
        self._size = 0

    def resetbuffer(self):
        self._state = []
        self._action = []
        self._reward = []
        self._nextstate = []

    def addtransition(self, state, action, reward, next_state):
        #checking if buffer is full and removing first element
        if self._size == self._maxsize:
            self._state.pop(0)
            self._action.pop(0)
            self._reward.pop(0)
            self._nextstate.pop(0)

            self._size -= 1

        #adding new data
        self._state.append(state)
        self._action.append(action)
        self._reward.append(reward)
        self._nextstate.append(next_state)

        #incrementing size
        self._size += 1

    def addepisode(self, episode):
        for i in range(len(episode)):
            self.addtransition(episode[i][0], episode[i][1], episode[i][2], episode[i][3])

    def sampletransition(self):
        #returns a random transition in the replay buffer
        index = np.random.randint(self._size)
        return self._state[index], self._action[index], self._reward[index], self._nextstate[index]

    def sampleepisode(self, steps):
        episode = []
        for i in range(steps):
            state, action, reward, next_state = self.sampletransition()
            episode.append((state, action, reward, next_state))
        return episode
