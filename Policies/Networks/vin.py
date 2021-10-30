import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def attention(tensor, x_pos, y_pos):
    """Attention model for grid world
    """
    num_data = tensor.size()[0]

    # Slicing x_pos positions
    slice_x_pos = x_pos.expand(self.imsize, 1, self.qout_c, num_data)
    slice_x_pos = slice_x_pos.permute(3, 2, 1, 0)
    q_out = tensor.gather(2, slice_x_pos).squeeze(2)

    # Slicing y_pos positions
    slice_y_pos = y_pos.expand(1, self.qout_c, num_data)
    slice_y_pos = slice_y_pos.permute(2, 1, 0)
    q_out = q_out.gather(2, slice_y_pos).squeeze(2)

    return q_out


class VIN(nn.Module):

    def __init__(self, action_dim, obs_dim, model_config):
        super(VIN, self).__init__()
        # get vi module params
        vi_config = model_config["vin_module"]

        # number of vi iterations
        self.k = vi_config["k"]
        self.qout_c = vi_config["qout_channels"]
        self.imsize = obs_dim[1]

        # conv layer to convert input to first reward image
        self.h_net = nn.Conv2d(in_channels=obs_dim[0],
                                out_channels=vi_config["hout_channels"],
                                kernel_size=vi_config["hconv_filter"],
                                stride=1,
                                padding=(vi_config["hconv_filter"] - 1) // 2,
                                bias=True)

        # conv layer to generate reward image
        self.r_net = nn.Conv2d(in_channels=vi_config["hout_channels"],
                               out_channels=1,
                               kernel_size=vi_config["rconv_filter"],
                               padding=(vi_config["rconv_filter"] - 1) // 2,
                               stride=1)

        # conv layer used to generate q vals inside of vi module
        self.q_net = nn.Conv2d(in_channels=2,
                               out_channels=self.qout_c,
                               kernel_size=vi_config["qconv_filter"],
                               padding=(vi_config["qconv_filter"] - 1) // 2,
                               stride=1)

        # fc to map output of vi module to number of actions
        self.fc = nn.Linear(in_features=self.qout_c,
                            out_features=action_dim,
                            bias=False)

    def forward(self, x, record_images=False):
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
        v = torch.zeros(r.size()).cuda()
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
        q_out = q[:, :, pos_x, pos_y]

        # get final q values
        q_out = self.fc(q_out)

        return q_out
