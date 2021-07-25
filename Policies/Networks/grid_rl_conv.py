import numpy as np
import torch
import torch.nn as nn

class Grid_RL_Conv(nn.Module):

    def __init__(self, classes):
        super(Grid_RL_Conv, self).__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(2, 10, 5),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(10, 10, 5),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True)
        )
        self.lin1 = nn.Linear(2890, 500)
        self.re1 = nn.ReLU(inplace=True)
        self.lin2 = nn.Linear(500, 100)
        self.re2 = nn.ReLU(inplace=True)
        self.lin3 = nn.Linear(100, classes)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.re1(self.lin1(x.flatten()))
        x = self.re2(self.lin2(x))
        x = self.sig(self.lin3(x))
        return x
