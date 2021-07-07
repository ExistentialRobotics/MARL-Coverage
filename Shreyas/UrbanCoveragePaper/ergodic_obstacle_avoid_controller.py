import numpy as np

#TODO
class ErgodicObstacleAvoidController(object):
    def __init__(self, qlis, qcoor, res, gain):
        super().__init__()
        self._qlis = qlis
        self._numrobot = len(qlis)
        self._res = res
        self._qcoor = qcoor

        self._K = gain*np.eye(2)

    def step(self, dt):

        #apply control input and update state
        for i in range(self._numrobot):
            # u_i = self._K @ (self._CV[i]-self._qlis[i])
            self._qlis[i] += u_i*dt

        #returning the current state
        return self._qlis


    def grid2World(self, x, y):
        '''
        we are assuming x and y are not in the image coordinate system, just
        array coordinates with the same standard orientation as R^2
        '''
        newx = self._qcoor[0][0] + (float(x)/self._res[0])*self._qcoor[1][0]
        newy = self._qcoor[0][1] + (float(y)/self._res[1])*self._qcoor[1][1]
        return np.array([[newx], [newy]])
