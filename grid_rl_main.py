import numpy as np
import matplotlib.pyplot as plt

from Environments.super_grid_rl import SuperGridRL
from Controllers.grid_rl_random_controller import GridRLRandomController


'''Environment Parameters'''
numrobot = 6
gridwidth = 25
gridlen = 25
seed = 420

'''Making the Controller for the Swarm Agent'''
c = GridRLRandomController(numrobot)

'''Making the Environment'''
e = SuperGridRL(c, numrobot, gridlen, gridwidth, seed=seed)


#tracking rewards
rewardlis = []

numsteps = 200
for i in range(numsteps):
    r = e.step()
    rewardlis.append(r)
    e.render()

plt.clf()
plt.plot(np.array(rewardlis))
plt.show()
