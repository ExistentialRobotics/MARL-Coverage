import numpy as np
import torch
import torch.nn as nn

from Policies.Networks.Qnet import init_weights

'''
Switched Padding back to what it was earlier to test if that is
breaking the performance, ideally it shouldn't be but who knows.
'''


class Grid_RL_Conv(nn.Module):

    def __init__(self, action_dim, obs_dim, model_config):
        super(Grid_RL_Conv, self).__init__()

        #getting all model config params
        conv_channels = model_config['conv_channels']
        conv_filters = model_config['conv_filters']
        for i in range(len(conv_filters)):
            conv_filters[i] = (conv_filters[i], conv_filters[i])
        if model_config['conv_activation'] == 'relu':
            conv_activation = nn.ReLU
        hidden_sizes = model_config['hidden_sizes']
        if model_config['hidden_activation'] == 'relu':
            hidden_activation = nn.ReLU
        output_activation = None

        # ensure that the number of channels and filters match
        assert len(conv_channels) == len(conv_filters)

        layers = []

        conv_output_size = np.array(obs_dim)

        # add conv layers to network
        for i in range(len(conv_channels)):
            # set padding so that img dims remain the same thru each conv layer
            padding = np.floor((conv_filters[i][0] - 1) / 2).astype(int)

            if i == 0:
                # add stride on first conv layer to reduce img dims
                stride = 2
                layers += [nn.Conv2d(obs_dim[0], conv_channels[i], conv_filters[i], padding=padding,
                                stride=stride), nn.BatchNorm2d(conv_channels[i], affine=False),
                                                conv_activation()]
            else:
                stride = 1
                layers += [nn.Conv2d(conv_channels[i - 1], conv_channels[i], conv_filters[i], padding=padding,
                                                        stride=stride), nn.BatchNorm2d(conv_channels[i], affine=False),
                                                conv_activation()]
            # calculate the output of the current layer based on the output of the last layer

            conv_output_size[1:] = np.floor((conv_output_size[1:] + 2 * padding - np.array(conv_filters[i])) / stride + 1)
            conv_output_size[0] = conv_channels[i]

        # add flatten layer to made conv output 1D for the fc layers
        layers += [nn.Flatten()]

        # add hidden layers to network
        for i in range(len(hidden_sizes)):
            if i == 0:
                # num in features of first layer input is flat output dim of the last conv layer
                layers += [nn.Linear(int(np.prod(conv_output_size)), hidden_sizes[i]),
                           hidden_activation()]
            else:
                layers += [nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]),
                           hidden_activation()]

        # last fc layer output features is the number of actions
        if output_activation == None:
            layers += [nn.Linear(hidden_sizes[-1], action_dim)]
        else:
            layers += [nn.Linear(hidden_sizes[-1], action_dim), output_activation()]
        self.layers = nn.Sequential(*layers)
        self.layers.apply(init_weights)

    def forward(self, x):
        # reshape input if not the right dims
        if len(x.shape) != 4:
            x = torch.unsqueeze(x, axis=0)
        return torch.squeeze(self.layers(x), axis=0)

class Critic(nn.Module):

    def __init__(self, act_dim, obs_dim, conv_channels, conv_filters, conv_activation, hidden_sizes, hidden_activation):
        super().__init__()

        assert len(conv_channels) == len(conv_filters)

        conv_layers = []

        conv_output_size = np.array(obs_dim)

        for i in range(len(conv_channels)):
            padding = np.floor((conv_filters[i][0] - 1) / 2).astype(int)
            if i == 0:
                stride = 2
                conv_layers += [nn.Conv2d(obs_dim[0], conv_channels[i], conv_filters[i], padding=padding,
                                          stride=stride), nn.BatchNorm2d(conv_channels[i], affine=False),
                                conv_activation()]
            else:
                stride = 1
                conv_layers += [nn.Conv2d(conv_channels[i - 1], conv_channels[i], conv_filters[i], padding=padding,
                                          stride=stride), nn.BatchNorm2d(conv_channels[i], affine=False),
                                conv_activation()]

            conv_output_size[1:] = np.floor((conv_output_size[1:] + 2 * padding - np.array(conv_filters[i])) / stride + 1)
            conv_output_size[0] = conv_channels[i]

        conv_layers += [nn.Flatten()]

        self.q_conv = nn.Sequential(*conv_layers)

        fc_layers = []

        for i in range(len(hidden_sizes)):
            if i == 0:
                fc_layers += [nn.Linear(np.prod(conv_output_size) + act_dim, hidden_sizes[i]),
                              nn.BatchNorm1d(hidden_sizes[i], affine=False), hidden_activation()]
            else:
                fc_layers += [nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]),
                              nn.BatchNorm1d(hidden_sizes[i], affine=False), hidden_activation()]

        fc_layers += [nn.Linear(hidden_sizes[-1], 1)]

        self.q_fc = nn.Sequential(*fc_layers)
        self.q_fc.apply(init_weights)

    def forward(self, obs, act):
        q = self.q_fc(torch.cat((self.q_conv(obs), act), dim=-1))
        return torch.squeeze(q, -1)
