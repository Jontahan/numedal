import torch
import torch.nn as nn
import torch.nn.functional as F

class LinRegNet(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(LinRegNet, self).__init__()
        self.linreg = nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        x = torch.flatten(x)
        return self.linreg(x)