import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data

class Graph_Data(object):

    def __init__(self, numfeatures, xinds, yinds, commradius):
        self._commradius = commradius
        self._numfeatures = numfeatures # number of features per robot
        self._xinds = xinds
        self._yinds = yinds
        self.features = None
        self.edge_index = None



    def set_data(self, xinds, yinds):
        self._xinds = xinds
        self._yinds = yinds
        edges = []
        f = []
        for i in range(xinds.shape[0]):
            for j in range(i, xinds.shape[0]):
                if np.linalg.norm(np.array([(xinds[i] - xinds[j]), (yinds[i] - yinds[j])])) <= self._commradius:
                    edges.append([i, j])
                    if [i] not in f:
                        f.append([i for _ in range(self._numfeatures)])
                    if [j] not in f:
                        f.append([j for _ in range(self._numfeatures)])
        self.edge_index = torch.tensor(edges)
        self.features = torch.tensor(f)

    def get_data(self):
        return Data(x=self.features, edge_index=self.edge_index.t().contiguous())

    def display_connections(self):
        for edge in self.edge_index:
            plt.plot([self._xinds[edge[0]] + 0.5, self._xinds[edge[1]] + 0.5], [self._yinds[edge[0]] + 0.5, self._yinds[edge[1]] + 0.5])
        # plt.show()
