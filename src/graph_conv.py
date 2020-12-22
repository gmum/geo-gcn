import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class SpatialGraphConv(MessagePassing):
    def __init__(self, coors, in_channels, out_channels, hidden_size, dropout=0):
        """
        coors - dimension of positional descriptors (e.g. 2 for 2D images)
        in_channels - number of the input channels (node features)
        out_channels - number of the output channels (node features)
        hidden_size - number of the inner convolutions
        dropout - dropout rate after the layer
        """
        super(SpatialGraphConv, self).__init__(aggr='add')
        self.dropout = dropout
        self.lin_in = torch.nn.Linear(coors, hidden_size * in_channels)
        self.lin_out = torch.nn.Linear(hidden_size * in_channels, out_channels)
        self.in_channels = in_channels

    def forward(self, x, pos, edge_index):
        """
        x - feature matrix of the whole graph [num_nodes, label_dim]
        pos - node position matrix [num_nodes, coors]
        edge_index - graph connectivity [2, num_edges]
        """
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))  # num_edges = num_edges + num_nodes

        return self.propagate(edge_index=edge_index, x=x, pos=pos, aggr='add')  # [N, out_channels, label_dim]

    def message(self, pos_i, pos_j, x_j):
        """
        pos_i [num_edges, coors]
        pos_j [num_edges, coors]
        x_j [num_edges, label_dim]
        """

        relative_pos = pos_j - pos_i  # [n_edges, hidden_size * in_channels]
        spatial_scaling = F.relu(self.lin_in(relative_pos))  # [n_edges, hidden_size * in_channels]

        n_edges = spatial_scaling.size(0)
        # [n_edges, in_channels, ...] * [n_edges, in_channels, 1]
        result = spatial_scaling.reshape(n_edges, self.in_channels, -1) * x_j.unsqueeze(-1)
        return result.view(n_edges, -1)

    def update(self, aggr_out):
        """
        aggr_out [num_nodes, label_dim, out_channels]
        """
        aggr_out = self.lin_out(aggr_out)  # [num_nodes, label_dim, out_features]
        aggr_out = F.relu(aggr_out)
        aggr_out = F.dropout(aggr_out, p=self.dropout, training=self.training)

        return aggr_out
