import numpy as np
import torch
import torch.nn as nn

from Policies.Networks.Qnet import init_weights

'''
Switched Padding back to what it was earlier to test if that is
breaking the performance, ideally it shouldn't be but who knows.
'''
class Grid_RL_Recur(nn.Module):

    def __init__(self, action_dim, obs_dim, model_config):
        super(Grid_RL_Recur, self).__init__()

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

        lstm_cell_size = model_config['lstm_cell_size']
        num_recurr_layers = model_config['num_recurr_layers']

        #if recurr_end, recurrent layer will be at end, otherwise it will be
        #after the conv layer
        self._recurr_end = model_config['recurr_end']

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

        # add flatten layer to made conv output 1D for the fc layers
        conv_layers += [nn.Flatten()]

        # add hidden layers to network
        lin_layers = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                # num in features of first layer input is flat output dim of the last conv layer
                if self._recurr_end:
                    lin_layers += [nn.Linear(int(np.prod(conv_output_size)),
                                            hidden_sizes[i]), hidden_activation()]
                else:
                    lin_layers += [nn.Linear(lstm_cell_size, hidden_sizes[i]),
                            hidden_activation()]
            else:
                lin_layers += [nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]),
                           hidden_activation()]

        # add LSTM layer between conv and fc layers
        if self._recurr_end:
            self.lstm = nn.LSTM(hidden_sizes[-1], lstm_cell_size,
                                num_recurr_layers, batch_first=True)
        else:
            self.lstm = nn.LSTM(int(np.prod(conv_output_size)), lstm_cell_size,
                                num_recurr_layers, batch_first=True)


        # last fc layer output features is the number of actions
        if not self._recurr_end:
            lin_layers += [nn.Linear(hidden_sizes[-1], action_dim)]
        else:
            final_lin = [nn.Linear(lstm_cell_size, action_dim)]

        #output activation if specified
        if output_activation and not self._recurr_end:
            lin_layers += output_activation()


        self.conv_layers = nn.Sequential(*conv_layers)
        self.conv_layers.apply(init_weights)
        self.lin_layers = nn.Sequential(*lin_layers)
        self.lin_layers.apply(init_weights)

        if self._recurr_end:
            self.final_lin = nn.Sequential(*final_lin)
            self.final_lin.apply(init_weights)



    def forward(self, x, hidden):
        # reshape input if not the right dims
        if len(x.shape) != 4:
            x = torch.unsqueeze(x, axis=0)
        if self._recurr_end:
            x = self.conv_layers(x)
            x = self.lin_layers(x)
            x = torch.unsqueeze(x, axis=1)
            x, hidden = self.lstm(x , hidden)
            x = torch.squeeze(x, axis=1)
            x = self.final_lin(x)
            return x, hidden
        else:
            x = torch.unsqueeze(self.conv_layers(x), axis=1)
            x, hidden = self.lstm(x, hidden)
            x = self.lin_layers(torch.squeeze(x, axis=1))
            return torch.squeeze(x, axis=0), hidden
