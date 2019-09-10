import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class GraphConv(MessagePassing):
    def __init__(self, coors, out_channels_1, out_features, label_dim=1, dropout=0):
        """
        label_dim - dimention of node reprezentaion
        coors - dimension of position (for MNIST 2)
        out_channels_1 - dimension of convolution on each reprezentation chanal 
                        * autput will have dimention label_dim * out_channels_1
        out_features - dimension of node representation after graphConv
        """
        super(GraphConv, self).__init__(aggr='add')
        self.lin_in = torch.nn.Linear(coors, label_dim * out_channels_1)
        self.lin_out = torch.nn.Linear(label_dim * out_channels_1, out_features)
        self.dropout = dropout

    def forward(self, x, pos, edge_index):
        """
        x - feature matrix of the whole graph [num_nodes, label_dim]
        pos - node position matrix [num_nodes, coors]
        edge_index - graph connectivity [2, num_edges]
        """

        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))  # num_edges = num_edges + num_nodes

        return self.propagate(edge_index=edge_index, x=x, pos=pos, aggr='add')  # [N, out_channels, label_dim]

    def message(self, pos_i, pos_j, x_j):
        """
        pos_i [num_edges, coors]
        pos_j [num_edges, coors]
        x_j [num_edges, label_dim]
        """

        tmp = pos_j - pos_i
        L = self.lin_in(tmp)  # [num_edges, out_channels]
        num_nodes, label_dim = list(x_j.size())
        label_dim_out_channels_1 = list(L.size())[1]

        X = F.relu(L)
        Y = x_j
        X = torch.t(X)
        X = F.dropout(X, p=self.dropout, training=self.training)
        result = torch.t(
            (X.view(label_dim, -1, num_nodes) * torch.t(Y).unsqueeze(1)).reshape(label_dim_out_channels_1, num_nodes))
        return result

    def update(self, aggr_out):
        """
        aggr_out [num_nodes, label_dim, out_channels]
        """
        aggr_out = self.lin_out(aggr_out)  # [num_nodes, label_dim, out_features]
        aggr_out = F.relu(aggr_out)
        aggr_out = F.dropout(aggr_out, p=self.dropout, training=self.training)

        return aggr_out
