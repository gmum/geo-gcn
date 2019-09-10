import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from graph_conv import GraphConv
from normalized_cut_2d import normalized_cut_2d
from torch_geometric.nn import graclus, max_pool, global_mean_pool


class GeoGC(torch.nn.Module):
    def __init__(self, dim_coor, out_dim, input_features,
                 layers_num, model_dim, out_channels_1, dropout,
                 use_cluster_pooling):
        super(GeoGC, self).__init__()
        self.layers_num = layers_num
        self.use_cluster_pooling = use_cluster_pooling

        self.conv_layers = [GraphConv(coors=dim_coor,
                                      out_channels_1=out_channels_1,
                                      out_features=model_dim,
                                      label_dim=input_features,
                                      dropout=dropout)] + \
                           [GraphConv(coors=dim_coor,
                                      out_channels_1=out_channels_1,
                                      out_features=model_dim,
                                      label_dim=model_dim,
                                      dropout=dropout) for _ in range(layers_num - 1)]

        self.conv_layers = torch.nn.ModuleList(self.conv_layers)

        self.fc1 = torch.nn.Linear(model_dim, out_dim)

    def forward(self, data):
        for i in range(self.layers_num):
            data.x = self.conv_layers[i](data.x, data.pos, data.edge_index)

            if self.use_cluster_pooling:
                weight = normalized_cut_2d(data.edge_index, data.pos)
                cluster = graclus(data.edge_index, weight, data.x.size(0))
                data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data.x = global_mean_pool(data.x, data.batch)
        x = self.fc1(data.x)

        return F.log_softmax(x, dim=1)
