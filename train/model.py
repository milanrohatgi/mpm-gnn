import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GraphAttentionNetwork(nn.Module):
    def __init__(self, input_dim=17, edge_dim=1, hidden_dim=128, num_layers=3, num_heads=4, dropout=0.1):
        super(GraphAttentionNetwork, self).__init__()
        
        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, edge_dim=hidden_dim))
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Output 2D positions
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            x = F.elu(x)
        
        return self.decoder(x)
