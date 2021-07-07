import numpy as np

class GridController(object):
    def __init__(self, qlis, qcoor, res, gain):
        super().__init__()
        self._qlis = qlis
        self._numrobot = len(qlis)
        self._res = res
        self._qcoor = qcoor

        self._K = gain*np.eye(2)

        #creating a list of targets
        self._targets = []

        for i in range(self._numrobot):
            #we could use task assignment to assign region based on proximity as
            #to minimize total distance traveled to static position
            xtarg = qcoor[0][0] + (i+0.5)*qcoor[1][0]/float(self._numrobot)
            ytarg = qcoor[0][1] + 0.5*qcoor[1][1]
            self._targets.append(np.array([[xtarg], [ytarg]]))

    def step(self, dt):
        #apply control input and update state
        for i in range(self._numrobot):
            u_i = self._K @ (self._targets[i]-self._qlis[i])
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
