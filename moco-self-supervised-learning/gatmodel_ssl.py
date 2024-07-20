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


# self.gatlayer = GAT(in_channels=100, out_channels=100, num_layers=1, dropout=0.1)
class GATLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GATLayer, self).__init__(aggr='add')  # "Add" aggregation.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.W_edge = torch.nn.Linear(in_channels, out_channels, bias=False)  ###
        self.a = torch.nn.Parameter(torch.Tensor(1, 2 * out_channels))
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W.weight)
        torch.nn.init.xavier_uniform_(self.W_edge.weight)  ###
        torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)
    def forward(self, x, edge_index, edge_attr):
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, fill_value=0)
        x = self.W(x)
        edge_attr = self.W_edge(edge_attr)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)

    def message(self, edge_index_i, x_i, x_j, edge_attr, size_i):
        x_ij = torch.cat([x_i, x_j + edge_attr], dim=1)
        alpha = self.a @ x_ij.t()
        alpha = alpha.squeeze()
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        return x_j * alpha.view(-1, 1)
    def update(self, aggr_out):
        return aggr_out


class GATModel(nn.Module):
    def __init__(self, hidden_dim=100, out_dim=64, num_layers=6, dropout=0.1):
        super(GATModel, self).__init__()
        self.out_dim = out_dim
        self.x_embedding = nn.Embedding(num_atom_type, hidden_dim)
        nn.init.xavier_uniform_(self.x_embedding.weight)

        # 初始化多个GAT层
        self.gat_layers = nn.ModuleList([
            GATLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        # 初始化多个归一化层
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        # Dropout 和 激活函数可以复用
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()
        # self.out_lin = nn.Linear(hidden_dim, 1)  # 最后的输出线性层
        # self.out = nn.Sequential(nn.Linear(hidden_dim, out_dim),
        #                          nn.ELU(),
        #                          nn.Linear(out_dim, 1))
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.x_embedding(x)
        # 遍历所有GAT层
        for gat_layer, norm_layer in zip(self.gat_layers, self.norm_layers):
            x_residual = x
            x = gat_layer(x, edge_index, edge_attr)
            x = norm_layer(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = x + x_residual  # residual connection
        # 全局平均池化作为readout
        x = global_mean_pool(x, batch)
        # 输出层
        x = self.fc(x)
        return x


