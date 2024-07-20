import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.nn import GAT
from torch_geometric.nn import global_mean_pool

from DefaultElement import DEFAULT_ELEMENTS

num_atom_type = len(DEFAULT_ELEMENTS)
num_bond_type = 100


class GATLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads):
        super(GATLayer, self).__init__(aggr='add')  # 使用加法聚合
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.per_head_dim = out_channels // heads

        self.W = nn.Linear(in_channels, self.per_head_dim * heads, bias=False)
        self.W_edge = nn.Linear(in_channels, self.per_head_dim * heads, bias=False)
        self.a = nn.Parameter(torch.Tensor(heads, 2 * self.per_head_dim))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W.weight)
        torch.nn.init.xavier_uniform_(self.W_edge.weight)
        torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, x, edge_index, edge_attr):
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, fill_value=0)
        x = self.W(x)
        edge_attr = self.W_edge(edge_attr)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)

    def message(self, edge_index_i, x_i, x_j, edge_attr, size_i):
        x_ij = torch.cat([x_i, x_j + edge_attr], dim=1)

        a = self.a.view(-1, self.heads, 2 * self.per_head_dim)
        x_ij = x_ij.view(-1, self.heads, 2 * self.per_head_dim)
        alpha = (a * x_ij).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        return (x_j.view(-1, self.heads, self.per_head_dim) * alpha.view(-1, self.heads, 1)).sum(dim=1)

    def update(self, aggr_out):
        return aggr_out


class GATModel(nn.Module):
    def __init__(self, hidden_dim=100, out_dim=64, num_layers=6, dropout=0.1, heads=4):
        super(GATModel, self).__init__()
        self.x_embedding = nn.Embedding(num_atom_type, hidden_dim)
        nn.init.xavier_uniform_(self.x_embedding.weight)

        self.gat_layers = nn.ModuleList([
            GATLayer(hidden_dim, hidden_dim*heads, heads=heads) for _ in range(num_layers)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()

        self.linear1 = nn.Linear(hidden_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.x_embedding(x)
        for gat_layer, norm_layer in zip(self.gat_layers, self.norm_layers):
            x_residual = x
            x = gat_layer(x, edge_index, edge_attr)
            x = norm_layer(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = x + x_residual
        x = global_mean_pool(x, batch)
        h = self.linear1(x)
        out = self.linear2(h)
        return out, h


