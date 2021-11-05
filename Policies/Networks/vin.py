import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def attention(tensor, S1, S2, qout_channels):
    """Attention model for grid world
    """
    num_data = tensor.size()[0]
    imsize = tensor.size()[2]

    # Slicing S1 positions
    slice_s1 = S1.expand(imsize, 1, qout_channels, num_data)
    slice_s1 = slice_s1.permute(3, 2, 1, 0)
    q_out = tensor.gather(2, slice_s1)
    q_out = q_out.squeeze(2)

    # Slicing S2 positions
    slice_s2 = S2.expand(1, qout_channels, num_data)
    slice_s2 = slice_s2.permute(2, 1, 0)
    q_out = q_out.gather(2, slice_s2).squeeze(2)

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
        self.fc = nn.Linear(in_features=self.qout_c + obs_dim[0] ** 2,
                            out_features=action_dim,
                            bias=False)

    def forward(self, x, record_images=False):
        # reshape input if not the right dims
        if len(x.shape) != 4:
            x = torch.unsqueeze(x, axis=0)

        # get indicies of robot positions
        inds = torch.nonzero(x[:, 0, :, :])
        inds = inds[:, 1:]

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
            rv = torch.cat([r, v], 1) # [batch_size, 2, imsize, imsize]
            q = self.q_net(rv)
            v, _ = torch.max(q, 1) # torch.max returns (values, indices)
            v = torch.unsqueeze(v, axis=1)

            if record_images:
                # Save single value image in Numpy array for each VI step
                self.value_images.append(v.data[0].cpu().numpy()) # cpu() works both GPU/CPU mode

        # Do one last convolution
        rv = torch.cat([r, v], 1) # [batch_size, 2, imsize, imsize]
        q = self.q_net(rv)

        # Attention model
        q_out = attention(q, inds[:, 0], inds[:, 1], self.qout_c)

        # get final q values
        q_out = self.fc(q_out)

        return q_out
