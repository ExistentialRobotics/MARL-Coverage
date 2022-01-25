import numpy as np
import torch
import torch.nn as nn

'''
Switched Padding back to what it was earlier to test if that is
breaking the performance, ideally it shouldn't be but who knows.
'''


class Alpha_Net(nn.Module):

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
        self.layers = nn.Sequential(*layers)

        self.output_layers = [nn.Linear(hidden_sizes[-1], action_dim), nn.Linear(hidden_sizes[-1], 1)]
        self.output_layers = nn.Sequential(*output_layers)

        self.layers.apply(init_weights)
        self.output_layers.apply(init_weights)


    def forward(self, x):
        # multiagent = False
        # # reshape input if not the right dims
        # if len(x.shape) == 3:
        #     x = torch.unsqueeze(x, axis=0)
        # elif len(x.shape) == 5:
        #     batch = x.shape[0]
        #     numrobot = x.shape[1]
        #     x = torch.reshape(x, (batch*numrobot,) + x.shape[2:5])
        #     multiagent = True

        x = torch.squeeze(self.layers(x), axis=0)
        if multiagent:
            x = torch.reshape(x,(batch, numrobot, -1))
        return x

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0)
