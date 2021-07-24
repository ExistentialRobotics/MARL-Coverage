import numpy as np
import torch
import torch.nn as nn

class Grid_RL_Conv(nn.Module):

    def __init__(self, classes):
        super(Grid_RL_Conv, self).__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(2, 10, 3),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True)
        )
        self.lin1 = nn.Linear(10, 10)
        self.re1 = nn.ReLU(inplace=True)
        self.lin2 = nn.Linear(10, classes)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.c1(x)
        x = self.re1(self.lin1(x))
        x = self.sig(self.lin2(x))
        return x
