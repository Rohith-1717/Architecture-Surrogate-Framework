import torch.nn as nn

from .Encoder import GraphEncoder


class ZeroGraph(nn.Module):
    def __init__(self, hidden_dim=128, num_targets=5):
        super().__init__()

        self.encoder = GraphEncoder(hidden_dim=hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_targets)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.encoder(x, edge_index, edge_attr, batch)
        return self.head(x)