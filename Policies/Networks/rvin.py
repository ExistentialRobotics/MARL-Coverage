import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class RVIN(nn.Module):

    def __init__(self, action_dim, obs_dim, model_config):
        super(RVIN, self).__init__()
        # get vi module params
        vi_config = model_config["vin_module"]

        # number of vi iterations
        self.k = vi_config["k"]

        conv_output_size = np.array(obs_dim)

        # conv layer to convert input to first reward image
        self.h_net = nn.Conv2d(in_channels=obs_dim[0],
                                out_channels=vi_config["hout_channels"],
                                kernel_size=vi_config["hconv_filter"],
                                stride=1,
                                padding=(vi_config["hconv_filter"] - 1) // 2,
                                bias=True)

        conv_output_size[1:] = np.floor((conv_output_size[1:] + 2 * ((vi_config["hconv_filter"] - 1) // 2)
                                         - np.array(vi_config["hconv_filter"])) + 1)
        conv_output_size[0] = vi_config["hout_channels"]

        # conv layer to generate reward image
        self.r_net = nn.Conv2d(in_channels=vi_config["hout_channels"],
                               out_channels=1,
                               kernel_size=vi_config["rconv_filter"],
                               padding=(vi_config["rconv_filter"] - 1) // 2,
                               stride=1)

        # conv layer used to generate q vals inside of vi module
        self.q_net = nn.Conv2d(in_channels=2,
                               out_channels=vi_config["qout_channels"],
                               kernel_size=vi_config["qconv_filter"],
                               padding=(vi_config["qconv_filter"] - 1) // 2,
                               stride=1)

        conv_output_size[1:] = np.floor((conv_output_size[1:] + 2 * ((vi_config["qconv_filter"] - 1) // 2)
                                         - np.array(vi_config["qconv_filter"])) + 1)
        conv_output_size[0] = vi_config["qout_channels"]

        # convert qnet output to a lower dimension
        # q_to_recurr = [nn.Flatten(), nn.Linear(int(np.prod(conv_output_size)), model_config["lstm_output"], bias=False), nn.ReLU()]
        # self.q_to_recurr = nn.Sequential(*q_to_recurr)

        # lstm to use history of past qvals to influence current ones
        self.lstm = nn.LSTM(vi_config["qout_channels"], model_config['lstm_cell_size'],
                            model_config['num_recurr_layers'], batch_first=True)

        # fc to map output of vi module to number of actions
        fc = [nn.ReLU(), nn.Linear(model_config['lstm_cell_size'], action_dim, bias=False)]
        self.fc = nn.Sequential(*fc)


    def forward(self, x, hidden, record_images=False):
        # reshape input if not the right dims
        # print(x.shape)
        if len(x.shape) != 4:
            x = torch.unsqueeze(x, axis=0)

        # TODO: vectorize this
        p_x = []
        p_y = []
        for i in range(x.shape[0]):
            pos_maps = torch.squeeze(x[i, 0, :, :], axis=0)
            # print(pos_maps)
            inds = np.argwhere(pos_maps.cpu().numpy()>0)
            # print(inds)
            p_x.append(inds[0, 0])
            p_y.append(inds[0, 1])
            # print(str(pos_x) + " " + str(pos_y))

        # Get reward image from observation image
        h = self.h_net(x)
        r = self.r_net(h)

        if record_images: # TODO: Currently only support single input image
            # Save grid image in Numpy array
            self.grid_image = x.data[0].cpu().numpy() # cpu() works both GPU/CPU mode
            # Save reward image in Numpy array
            self.reward_image = r.data[0].cpu().numpy() # cpu() works both GPU/CPU mode

        # Initialize value map (zero everywhere)
        v = torch.zeros(r.size())
        if torch.cuda.is_available():
            v = v.cuda()

        # Wrap to autograd.Variable
        v = Variable(v)

        # K-iterations of Value Iteration module
        for _ in range(self.k):
            # print("---------------K-iterations of Value Iteration module---------------")
            # print(r.shape)
            # print(v.shape)
            rv = torch.cat([r, v], 1) # [batch_size, 2, imsize, imsize]
            q = self.q_net(rv)
            v, _ = torch.max(q, 1) # torch.max returns (values, indices)
            # print(v.shape)
            v = torch.unsqueeze(v, axis=1)

            if record_images:
                # Save single value image in Numpy array for each VI step
                self.value_images.append(v.data[0].cpu().numpy()) # cpu() works both GPU/CPU mode

        # Do one last convolution
        rv = torch.cat([r, v], 1) # [batch_size, 2, imsize, imsize]
        q = self.q_net(rv)

        # Attention model
        # q_out = attention(q, pos_x, pos_y)
        q_out = torch.zeros((q.shape[0], q.shape[1]))
        for i in range(len(p_x)):
            # print(q[i, :, p_x[i], p_y[i]].shape)
            q_out[i] = q[i, :, p_x[i], p_y[i]]

        # lstm pass
        q_out = torch.unsqueeze(q_out, axis=1)
        q_out, hidden = self.lstm(q_out, hidden)

        # get final q values
        q_out = self.fc(torch.squeeze(q_out, axis=1))

        return q_out, hidden
