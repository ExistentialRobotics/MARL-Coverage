import numpy as np
import torch
import torch.nn as nn

from Policies.Networks.Qnet import init_weights

'''
Switched Padding back to what it was earlier to test if that is
breaking the performance, ideally it shouldn't be but who knows.
'''


class Grid_RL_Recur(nn.Module):

    def __init__(self, action_dim, obs_dim, conv_channels, conv_filters,
                 conv_activation, hidden_sizes, hidden_activation,
                 output_activation=None):
        super(Grid_RL_Recur, self).__init__()

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
                conv_layers += [nn.Conv2d(obs_dim[0], conv_channels[i], conv_filters[i], padding=padding,
                                stride=stride), nn.BatchNorm2d(conv_channels[i], affine=False),
                                                conv_activation()]
            else:
                stride = 1
                conv_layers += [nn.Conv2d(conv_channels[i - 1], conv_channels[i], conv_filters[i], padding=padding,
                                                        stride=stride), nn.BatchNorm2d(conv_channels[i], affine=False),
                                                conv_activation()]
            # calculate the output of the current layer based on the output of the last layer

            conv_output_size[1:] = np.floor((conv_output_size[1:] + 2 * padding - np.array(conv_filters[i])) / stride + 1)
            conv_output_size[0] = conv_channels[i]

        # add flatten layer to made conv output 1D for the fc layers
        conv_layers += [nn.Flatten()]

        # add hidden layers to network
        lin_layers = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                # num in features of first layer input is flat output dim of the last conv layer
                lin_layers += [nn.Linear(int(np.prod(conv_output_size)), hidden_sizes[i]),
                           hidden_activation()]
            else:
                lin_layers += [nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]),
                           hidden_activation()]

        self.conv_layers = nn.Sequential(*conv_layers)
        self.conv_layers.apply(init_weights)
        self.lin_layers = nn.Sequential(*lin_layers)
        self.lin_layers.apply(init_weights)

        # add LSTM layer between last fc layer and action output
        self.lstm = nn.LSTM(hidden_sizes[-1], action_dim, 1, batch_first=False)

    def forward(self, x, hidden):
        # reshape input if not the right dims
        if len(x.shape) != 4:
            x = torch.unsqueeze(x, axis=0)

        x = self.conv_layers(x)
        x = self.lin_layers(x)
        x, hidden = self.lstm(torch.unsqueeze(x, axis=0), hidden)
        return torch.squeeze(x, axis=0), hidden
