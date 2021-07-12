import numpy as np
from controller import Controller

class GridController(Controller):
    def __init__(self, numrobot, qcoor, res, gain):
        super().__init__(numrobot)

        self._K = gain*np.eye(2)
        self._res = res
        self._qcoor = qcoor

        #creating a list of targets
        self._targets = []

        for i in range(self._numrobot):
            #we could use task assignment to assign region based on proximity as
            #to minimize total distance traveled to static position
            xtarg = qcoor[0][0] + (i+0.5)*qcoor[1][0]/float(self._numrobot)
            ytarg = qcoor[0][1] + 0.5*qcoor[1][1]
            self._targets.append(np.array([[xtarg], [ytarg]]))

    def getControls(self, observation):
        #compute all controls
        U = []
        for i in range(self._numrobot):
            u_i = self._K @ (self._targets[i]-observation[i])
            U.append(u_i)

        #returning list of controls
        return U
