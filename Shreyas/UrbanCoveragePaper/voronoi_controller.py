import numpy as np
from scipy.spatial import KDTree

class VoronoiController(object):
    def __init__(self, qlis, qcoor, res, gain):
        super().__init__()
        self._qlis = qlis
        self._numrobot = len(qlis)
        self._res = res
        self._qcoor = qcoor

        self._K = gain*np.eye(2)

        #creating list of arrays to store intermediate computations needed for update/control
        self._CV = []
        self._MV = []

        for i in range(self._numrobot):
            self._CV.append(np.zeros((2,1)))
            self._MV.append(0)

    def step(self, dt):
        qlis_modshape = np.array(self._qlis).reshape(self._numrobot, 2)
        self._kdtree = KDTree(qlis_modshape)

        self.computeVoronoiIntegrals()

        #apply control input and update state
        for i in range(self._numrobot):
            u_i = self._K @ (self._CV[i]-self._qlis[i])
            self._qlis[i] += u_i*dt

        #returning the current state
        return self._qlis

    def computeVoronoiIntegrals(self):
        '''
        computing CV, MV
        '''
        #zeroing all intermediate stores
        for i in range(self._numrobot):
            self._CV[i] = np.zeros((2,1))
            self._MV[i] = 0

        #looping over all squares in Q
        for i in range(self._res[0]):
            for j in range(self._res[1]):
                #converting the grid coordinate to world coordinate
                pos = self.grid2World(i,j)

                #some functions want vector not matrix so we convert
                reshapedpos = np.reshape(pos, (2,))

                #deciding which voronoi region it belongs to
                region = self._kdtree.query(reshapedpos)[1]

                #incrementing M and L
                self._MV[region] += 1
                self._CV[region] += pos

        #computing all C_V based on M's and L's
        for i in range(self._numrobot):
            self._CV[i] = self._CV[i]/self._MV[i]

    def grid2World(self, x, y):
        '''
        we are assuming x and y are not in the image coordinate system, just
        array coordinates with the same standard orientation as R^2
        '''
        newx = self._qcoor[0][0] + (float(x)/self._res[0])*self._qcoor[1][0]
        newy = self._qcoor[0][1] + (float(y)/self._res[1])*self._qcoor[1][1]
        return np.array([[newx], [newy]])
