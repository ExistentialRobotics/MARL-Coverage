import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn
from . graph_utils import LSIGF, GraphFilterBatch

class GNN(torch.nn.Module):
    def __init__(self, action_dim, obs_dim, model_config):
        super(GNN, self).__init__()

        """ Init the conv layers """
        # getting all conv layer config params
        conv_channels = model_config['conv_channels']
        conv_filters = model_config['conv_filters']
        for i in range(len(conv_filters)):
            conv_filters[i] = (conv_filters[i], conv_filters[i])
        if model_config['conv_activation'] == 'relu':
            conv_activation = nn.ReLU

        output_activation = None

        # ensure that the number of channels and filters match
        assert len(conv_channels) == len(conv_filters)

        conv_layers = []

        conv_output_size = np.array(obs_dim)

        # add conv layers to network
        for i in range(len(conv_channels)):
            # set padding so that img dims remain the same thru each conv layer
            padding = np.floor((conv_filters[i][0] - 1) / 2).astype(int)

            if i == 0:
                # add stride on first conv layer to reduce img dims
                stride = 2
                conv_layers += [nn.Conv2d(obs_dim[0], conv_channels[i],
                                          conv_filters[i], padding=padding,
                                          stride=stride),
                                nn.BatchNorm2d(conv_channels[i], affine=False),
                                conv_activation()]
            else:
                stride = 1
                conv_layers += [nn.Conv2d(conv_channels[i - 1],
                                          conv_channels[i],
                                          conv_filters[i],
                                          padding=padding,
                                          stride=stride),
                                nn.BatchNorm2d(conv_channels[i],
                                               affine=False),
                                conv_activation()]
                # calculate the output of the current layer based on the output of the last layer

            conv_output_size[1:] = np.floor((conv_output_size[1:] + 2 * padding
                                             - np.array(conv_filters[i])) / stride + 1)
            conv_output_size[0] = conv_channels[i]

        # add flatten layer to made conv output 1D for the gnn layers
        conv_layers += [nn.Flatten()]
        # insert conv layers into nn.sequential, now they're ready for forward()
        self.conv_layers = nn.Sequential(*conv_layers)


        """ Init MLP to convert CNN output to GNN input """
        hidden_MLP_layers = []

        # get MLP layer config params
        hidden_sizes = model_config['hidden_mlp_sizes']
        if model_config['hidden_mlp_activation'] == 'relu':
            hidden_activation = nn.ReLU
        hidden_mlp_output = model_config["node_feature_size"]

        # add hidden mlp layers to network
        for i in range(len(hidden_sizes)):
            if i == 0:
                # num in features of first layer input is flat output dim of the last conv layer
                hidden_MLP_layers += [nn.Linear(int(np.prod(conv_output_size)), hidden_sizes[i]),
                           hidden_activation()]
            else:
                hidden_MLP_layers += [nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]),
                           hidden_activation()]

        # save the output of the hidden mlp, will be needed later
        self.num_compressed_features = hidden_sizes[-1]

        # insert hidden mlp layers into nn.sequential, now they're ready for forward()
        self.hidden_MLP_layers = nn.Sequential(*hidden_MLP_layers)
        self.hidden_MLP_layers.apply(init_weights)


        """ Init the GNN layers """
        gfl = []  # Graph Filtering Layers

        # getting all gnn layer config params
        nGraphFilterTaps = [model_config["num_gf_taps"]]
        node_feature_size = [model_config["node_feature_size"]]

        self.L = len(nGraphFilterTaps)  # Number of graph filtering layers
        self.F = [self.num_compressed_features] + node_feature_size  # Features
        self.K = nGraphFilterTaps  # nFilterTaps # Filter taps
        self.E = 1  # Number of edge features
        self.bias = True

        for l in range(self.L):
            # \\ Graph filtering stage:
            gfl.append(GraphFilterBatch(self.F[l], self.F[l + 1], self.K[l], self.E, self.bias))
            # There is a 2*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            gfl.append(nn.ReLU(inplace=True))

        # insert gnn layers into nn.sequential, now they're ready for forward()
        self.GFL = nn.Sequential(*gfl)  # Graph Filtering Layers


        """ Init MLP layers to convert GNN output to q vals """
        qval_MLP_layers = []

        # get qval mlp config params
        qval_mlp_size = model_config["qval_mlp_size"]
        if model_config["qval_mlp_activation"] == 'relu':
            qval_mlp_activation = nn.ReLU


        # in features is GNN output, out features is num actions
        qval_MLP_layers += [nn.Linear(self.F[-1], qval_mlp_size), qval_mlp_activation()]
        qval_MLP_layers += [nn.Linear(qval_mlp_size, action_dim)]

        # insert qval mlp layers into nn.sequential, now they're ready for forward()
        self.qval_MLP_layers = nn.Sequential(*qval_MLP_layers)
        self.qval_MLP_layers.apply(init_weights)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0)
