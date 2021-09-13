import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from . graph_utils import LSIGF, GraphFilterBatch, BatchLSIGF

class GNN(torch.nn.Module):
    def __init__(self, action_dim, obs_dim, model_config):
        super(GNN, self).__init__()
        #cpu vs gpu code
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_dim = action_dim

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

        # add flatten layer to make conv output 1D for the gnn layers
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

        # save the output of the hidden mlp, will be needed for the forward pass
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


    def addGSO(self, S):
        # We add the GSO on real time, this GSO also depends on time and has
        # shape either B x N x N or B x E x N x N
        if self.E == 1:  # It is B x T x N x N
            assert len(S.shape) == 3
            self.S = S.unsqueeze(1)  # B x E x N x N
        else:
            assert len(S.shape) == 4
            assert S.shape[1] == self.E
            self.S = S


    def forward(self, inputTensor):
        if len(inputTensor.shape) != 5:
            inputTensor = torch.unsqueeze(inputTensor, axis=0)
        # print("input tensor shape: " + str(inputTensor.shape))

        batch_size = inputTensor.shape[0]
        num_robot  = inputTensor.shape[1]

        # B x G x N
        extractFeatureMap = torch.zeros(batch_size, self.num_compressed_features, num_robot).to(self._device)

        # pass thru conv then mlp
        for id_agent in range(num_robot):
            input_currentAgent = inputTensor[:, id_agent]
            featureMapFlatten = self.conv_layers(input_currentAgent)
            compressfeature = self.hidden_MLP_layers(featureMapFlatten)
            extractFeatureMap[:, :, id_agent] = compressfeature # B x F x N

        # adding GSOs
        for l in range(self.L):
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            self.GFL[2 * l].addGSO(self.S) # add GSO for GraphFilter

        # pass thru graph conv layers: B x F x N - > B x G x N,
        sharedFeature = self.GFL(extractFeatureMap)

        # pass thru final mlp to get qvals
        action_predict = torch.zeros((batch_size, num_robot, self.action_dim), dtype=torch.float, device=self._device)
        for id_agent in range(num_robot):
            sharedFeatureFlatten = sharedFeature[:, :, id_agent]
            action_predict[:, id_agent] = self.qval_MLP_layers(sharedFeatureFlatten) # 1 x 5
        return action_predict

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0)
