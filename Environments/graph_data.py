import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data

class Graph_Data(object):

    def __init__(self, xinds, yinds, commradius):
        self._commradius = commradius
        self.edge_index = None
        self._xinds = xinds
        self._yinds = yinds
        self.x = None


    def set_data(self, xinds, yinds):
        self._xinds = xinds
        self._yinds = yinds
        edges = []
        x = []
        for i in range(xinds.shape[0]):
            for j in range(i, xinds.shape[0]):
                if np.linalg.norm(np.array([(xinds[i] - xinds[j]), (yinds[i] - yinds[j])])) <= self._commradius:
                    edges.append([i, j])
                    if [i] not in x:
                        x.append([i])
                    if [j] not in x:
                        x.append([j])
        self.edge_index = torch.tensor(edges)
        self.x = torch.tensor(x)

    def get_data(self):
        return Data(x=self.x, edge_index=self.edge_index.t().contiguous())

    def display_connections(self):
        for edge in self.edge_index:
            plt.plot([self._xinds[edge[0]] + 0.5, self._xinds[edge[1]] + 0.5], [self._yinds[edge[0]] + 0.5, self._yinds[edge[1]] + 0.5])
        # plt.show()
