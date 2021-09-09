import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNN(torch.nn.Module):
    def __init__(self, conv_out_dim, num_features, out_dim, num_layers):
        super().__init__()

        layers = []

        for i in range(num_layers):
            if i == 0:
                layers += [GCNConv(conv_out_dim, num_features[i])]
            elif i == num_layers - 1:
                layers += [GCNConv(num_features[i - 1], out_dim)]
            else:
                layers += [GCNConv(num_features[i - 1], num_features[i])]

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
