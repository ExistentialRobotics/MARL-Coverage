import numpy as np
from scipy.spatial import KDTree
from controller import Controller

class VoronoiController(Controller):
    """
    Voronoi Controller takes the list of robot positions as the observation,
    and splits the region defined by qcoor into grid cells specified by res,
    and integrates over these cells to find the centroid of each voronoi
    region. It then sends each robot toward the centroid of its voronoi
    region.
    """
    def __init__(self, numrobot, qcoor, res, gain):
        super().__init__(numrobot)

        #gain matrix
        self._K = gain*np.eye(2)

        #res[0] determines how many cells to split x-axis into
        #res[1] determines how many cells to split y-axis into
        self._res = res

        #q[0] is the coordinates of the lower left hand corner of the rectangular region
        #q[1] is the delta to get the coordinates of the upper right hand corner
        self._qcoor = qcoor

        #creating list of arrays to store intermediate computations needed for update/control
        self._CV = []
        self._MV = []

        for i in range(self._numrobot):
            self._CV.append(np.zeros((2,1)))
            self._MV.append(0)

    def getControls(self, observation):
        #observation should be a list of robot positions
        qlis_modshape = np.array(observation).reshape(self._numrobot, 2)
        self._kdtree = KDTree(qlis_modshape)

        self.computeVoronoiIntegrals()

        #compute all control input
        U = []
        for i in range(self._numrobot):
            u_i = self._K @ (self._CV[i]-observation[i])
            U.append(u_i)

        #returning the list of controls
        return U

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
