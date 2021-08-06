import numpy as np
import torch
import torch.nn as nn

class Grid_RL_Conv(nn.Module):

    def __init__(self, action_dim, obs_dim, conv_channels, conv_filters,
                 conv_activation, hidden_sizes, hidden_activation,
                 output_activation):
        super(Grid_RL_Conv, self).__init__()

        # ensure that the number of channels and filters match
        assert len(conv_channels) == len(conv_filters)

        layers = []

        conv_output_size = np.array(obs_dim)

        # add conv layers to network
        for i in range(len(conv_channels)):
            if i == 0:
                # add stride on first conv layer to reduce img dims
                stride = 2
                layers += [nn.Conv2d(obs_dim[0], conv_channels[i], conv_filters[i],
                                          stride=stride), nn.BatchNorm2d(conv_channels[i], affine=False),
                                conv_activation()]
            else:
                stride = 1
                layers += [nn.Conv2d(conv_channels[i - 1], conv_channels[i], conv_filters[i],
                                          stride=stride), nn.BatchNorm2d(conv_channels[i], affine=False),
                                conv_activation()]

            # calculate the output of the current layer based on the output of the last layer
            conv_output_size[1:] = np.floor((conv_output_size[1:] - np.array(conv_filters[i])) / stride + 1)
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
        layers += [nn.Linear(hidden_sizes[-1], action_dim), output_activation()]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return torch.squeeze(self.layers(x), axis=0)
