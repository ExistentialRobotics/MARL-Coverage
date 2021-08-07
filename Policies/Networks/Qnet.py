import numpy as np

import torch
import torch.nn as nn


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0)


class Critic(nn.Module):

    def __init__(self, obs_dim, act_dim, conv_channels, conv_filters, conv_activation, hidden_sizes, hidden_activation):
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