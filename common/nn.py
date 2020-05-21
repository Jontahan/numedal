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

class LinRegNet(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(LinRegNet, self).__init__()
        self.linreg = nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        x = torch.flatten(x)
        return self.linreg(x)

class NeuralNet(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 32)
        self.fc2 = nn.Linear(32, n_outputs)

    def forward(self, x):
        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)