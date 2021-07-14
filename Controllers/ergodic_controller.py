import numpy as np
from controller import Controller

class ErgodicController(Controller):
    """
    ErgodicController takes the list of robot positions as the observation,
    uses this to update the weights of the fourier basis of the
    visitation distribution. It then compares this to the precomputed
    fourier coefficients of the given coverage distribution, which is assumed
    to be uniform if none is given. Using this we can compute the controls of each agent.

    Equation References and Notation are from the paper:
    'Multi-Agent Ergodic Coverage with Obstacle Avoidance'

    """
    #TODO fix: right now xcoor needs to be based at (0,0) because we aren't
    #          translating the fourier basis functions

    def __init__(self, numrobot, xcoor, res, numbasis, dt, coverage_dist=None,  klis=None, umax=10):
        super().__init__(numrobot)

        #defining region and grid for fourier projection
        self._xcoor = xcoor
        self._res = res

        #Assuming X is in R^2
        self._n = 2

        #fourier coefficients
        self._mu_k = np.zeros((numbasis, 1))
        #the c_klis is not normalized by N*t
        self._c_k = np.zeros((numbasis, 1))

        #list of wave-number vectors of the fourier basis
        self._klis = klis

        #coverage_dist should be a function taking state to probability
        self._coverage_dist = coverage_dist
        self._numbasis = numbasis

        self._dt = dt
        self._totaltime = 0

        self._umax = umax

        #if the wave-number list isn't provided we will generate one
        if(self._klis == None):
            self._klis = []
            for i in range(numbasis):
                self._klis.append(i*np.array([[1.], [1.]]))

        #computing mu fourier coefficients
        self.projectCoverageDist()


    def getControls(self, observation):
        #incrementing time
        self._totaltime += self._dt

        #update c_klis
        for i in range(self._numbasis):
            for j in  range(self._numrobot):
                self._c_k[i][0] += self.computeBasis(self._klis[i], observation[j])*self._dt

        #creating Blist to store control directions
        B = []
        for i in range(self._numrobot):
            B.append(np.zeros((2,1)))

        #computing S_k
        S_k = self._c_k - self._numrobot*self._totaltime*self._mu_k

        #computing the controls
        for i in range(self._numbasis):
            b = self.calculateLambda(self._klis[i]) * S_k[i][0]
            for j in range(self._numrobot):
                B[j] += b*self.computeBasisGradient(self._klis[i], observation[j])

        #normalizing B and scaling by -umax
        for i in range(self._numrobot):
            B[i] = -self._umax/np.sqrt(B[i].dot(B[i])) * B[i]

        #B is now the list of controls 
        return B


    def computeBasisGradient(self, k, x):
        """
        returns the gradient of the computeBasis function
        """

        #finding the h_k constant that makes the basis orthonormal
        #we don't need to check the zero case because gradient of constant is 0
        h_k = np.sqrt(res[0]*res[1])/2

        #calculating each term of the gradient
        r1 = -k[0][0] / xcoor[1][0] * np.pi *np.sin(k[0][0] * np.pi * x[0][0] / xcoor[1][0]) * np.cos(k[1][0] * np.pi * x[1][0] / xcoor[1][1])
        r2 = -k[1][0] / xcoor[1][1] * np.pi * np.cos(k[0][0] * np.pi * x[0][0] / xcoor[1][0]) * np.sin(k[1][0] * np.pi * x[1][0] / xcoor[1][1])

        result = 1.0/h_k * np.array([[r1], [r2]])
        return result

    def computeBasis(self, k, x):
        #finding the h_k constant that makes the basis orthonormal
        h_k = np.sqrt(res[0]*res[1])
        if(k[0][0] != 0 or k[1][0] != 0):
            h_k = h_k/2

        result = np.cos(k[0][0] * np.pi * x[0][0] / xcoor[1][0]) * np.cos(
            k[1][0] * np.pi * x[1][0] / xcoor[1][1])
        return 1.0/h_k * result

    def projectCoverageDist(self):
        A = self._xcoor[1][0] * self._xcoor[1][1]
        dA = A/(res[0]*res[1])

        #looping over all squares in X
        for i in range(self._res[0]):
            for j in range(self._res[1]):
                #converting the grid coordinate to world coordinate
                pos = self.grid2World(i,j)

                #evaluating coverage distribution
                mu = 1.0/(A)
                if(self._coverage_dist != None):
                    mu = self._coverage_dist(pos)

                #looping over all basis functions
                for k in range(self._numbasis):
                    #integrating: eq 5
                    self._mu_k[k][0] += mu*self.computeBasis(self._klis[k], pos)*dA

    def calculateLambda(self, k):
        #computes the weighting based on lambda
        Lambda = (1+k.dot(k))**((self._n + 1)/2)
        Lambda = 1/Lambda
        return Lambda

    def grid2World(self, x, y):
        '''
        we are assuming x and y are not in the image coordinate system, just
        array coordinates with the same standard orientation as R^2
        '''
        newx = self._qcoor[0][0] + (float(x)/self._res[0])*self._qcoor[1][0]
        newy = self._qcoor[0][1] + (float(y)/self._res[1])*self._qcoor[1][1]
        return np.array([[newx], [newy]])

