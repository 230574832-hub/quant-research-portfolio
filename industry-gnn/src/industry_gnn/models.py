from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATConv, GCNConv


class FeatureEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, dropout: float):
        super().__init__()
        blocks = []
        current = input_dim
        for _ in range(max(1, layers)):
            blocks.extend([nn.Linear(current, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            current = hidden_dim
        self.network = nn.Sequential(*blocks)

    def forward(self, x):
        return self.network(x)


class SharedMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int = 1, dropout: float = 0.15, **_):
        super().__init__()
        self.encoder = FeatureEncoder(input_dim, hidden_dim, layers, dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        return self.head(self.encoder(data.x)).squeeze(-1)


class IndustryAggregationMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int = 1, dropout: float = 0.15, **_):
        super().__init__()
        # A shared encoder isolates the incremental effect of peer information.
        self.encoder = FeatureEncoder(input_dim, hidden_dim, layers, dropout)
        self.head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, data):
        own = self.encoder(data.x)
        peer = self.encoder(data.agg_x)
        return self.head(torch.cat([own, peer], dim=-1)).squeeze(-1)


class GCNRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layers: int = 1,
        graph_layers: int = 1,
        dropout: float = 0.15,
        **_,
    ):
        super().__init__()
        self.encoder = FeatureEncoder(input_dim, hidden_dim, layers, dropout)
        self.convs = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim, add_self_loops=False, normalize=True) for _ in range(max(1, graph_layers))]
        )
        self.dropout = nn.Dropout(dropout)
        # A complete industry clique gives every node the same one-layer GCN
        # message.  Keep a direct stock-specific path so that information is not
        # destroyed by aggregation.
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        own = self.encoder(data.x)
        message = own
        for conv in self.convs:
            message = self.dropout(torch.relu(conv(message, data.edge_index)))
        return self.head(torch.cat([own, message, own - message], dim=-1)).squeeze(-1)


class GATRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layers: int = 1,
        graph_layers: int = 1,
        gat_heads: int = 2,
        dropout: float = 0.15,
        **_,
    ):
        super().__init__()
        self.encoder = FeatureEncoder(input_dim, hidden_dim, layers, dropout)
        self.convs = nn.ModuleList(
            [
                GATConv(
                    hidden_dim,
                    hidden_dim,
                    heads=gat_heads,
                    concat=False,
                    dropout=dropout,
                    add_self_loops=False,
                )
                for _ in range(max(1, graph_layers))
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        own = self.encoder(data.x)
        message = own
        for conv in self.convs:
            message = self.dropout(torch.relu(conv(message, data.edge_index)))
        return self.head(torch.cat([own, message, own - message], dim=-1)).squeeze(-1)


MODEL_CLASSES = {
    "shared_mlp": SharedMLP,
    "industry_mlp": IndustryAggregationMLP,
    "gcn": GCNRegressor,
    "gcn_random": GCNRegressor,
    "gcn_label_random": GCNRegressor,
    "gat": GATRegressor,
    "gat_random": GATRegressor,
    "gat_label_random": GATRegressor,
    "gat_self_only": GATRegressor,
}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODEL_CLASSES:
        raise ValueError(f"Unknown neural model: {name}")
    return MODEL_CLASSES[name](**kwargs)

