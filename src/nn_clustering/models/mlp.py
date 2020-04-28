import numpy as np
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, out_dim, n_hidden_layers=0, hidden_dim=None):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim or out_dim
        self.n_hidden_layers = n_hidden_layers

        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden_layer = self.construct_hidden_layer()
        self.output_layer = nn.Linear(self.hidden_dim, self.out_dim)

    def construct_hidden_layer(self):
        hidden_layer = nn.ModuleList([
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
        ])
        for _ in np.arange(self.n_hidden_layers):
            self.hidden_layer.extend([
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
            ])
        return hidden_layer

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x
