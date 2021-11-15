import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PointConv, fps, radius
from torch_scatter import scatter
from torch_geometric.nn import knn_graph, knn_interpolate
from torch_geometric.nn import GATConv


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn, aggr='max')

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],max_num_neighbors=16)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index) ## Change to normal message passing of mlp over edges and max pool
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GATConvModel(nn.Module):
    def __init__(self, k, edge_index, in_features=32):
        super(GATConvModel, self).__init__()
        
        self.k = k
        self.edge_index = edge_index
        self.in_features = in_features

        self.block1_1     = GATConv(in_channels=self.in_features, out_channels=64, heads=1)
        self.block1_2     = GATConv(in_channels=64, out_channels=256, heads=1)
        self.block1_3     = GATConv(in_channels=256, out_channels=1024, heads=1)
        self.block1_4     = GATConv(in_channels=1024, out_channels=256, heads=1)
        self.block1_5     = GATConv(in_channels=256, out_channels=64, heads=1)
        self.block1_6     = GATConv(in_channels=64, out_channels=3, heads=1)


    def forward(self, points):
        points_shape = points.shape
        
        points = points.reshape(points.shape[0], -1, self.in_features)
        points = points.reshape(-1, points.shape[-1])

        edge_index_1 = self.edge_index
        edge_index = []
        if points_shape[0] > 1:
            for i in range(points_shape[0]):
                edge_index.append(edge_index_1 + ((torch.max(edge_index_1)+1)*i))
            edge_index = torch.cat(edge_index, axis=1)
        else:
            edge_index = edge_index_1

        b1 = points
        b1 = F.relu(self.block1_1(b1, edge_index))
        b1 = F.relu(self.block1_2(b1, edge_index))
        b1 = F.relu(self.block1_3(b1, edge_index))
        b1 = F.relu(self.block1_4(b1, edge_index))
        b1 = F.relu(self.block1_5(b1, edge_index))
        output = self.block1_6(b1, edge_index)
        output = output.reshape(points_shape[0], -1, output.shape[-1])
        return output
    
    
