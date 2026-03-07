import torch
import torch.nn as nn
from torch_geometric.nn import SGConv

# Code from the Tutorial 2 of the Basiralab's repository:
class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, input_dim, output_dim=35778):
        super(SGC, self).__init__()

        self.W = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.W(x)