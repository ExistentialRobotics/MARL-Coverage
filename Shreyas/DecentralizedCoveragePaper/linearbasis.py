import numpy as np
from scipy.stats import multivariate_normal
import time

class GaussianBasis(object):
    def __init__(self, mulis, sigmalis):
        super().__init__()
        self._basislen = len(mulis)
        self._a = np.random.rand(self._basislen, 1)
        self._kappa = np.zeros((self._basislen,1))
        self._mulis = mulis
        self._sigmalis = sigmalis

    def eval(self, position):
        sum = np.transpose(self.evalBasis(position)) @ self._a
        return sum[0]

    def evalBasis(self, position):
        for i in range(self._basislen):
            self._kappa[i][0] = multivariate_normal(self._mulis[i],
                                           self._sigmalis[i]).pdf(position)
        return self._kappa

    def getparam(self):
        return self._a
    def updateparam(self, newa):
        self._a = newa
