import numpy as np
import torch
import torch.nn as nn
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
        
        
#         edge_index_i = []
#         edge_index_j = []
# #         num_nodes = 954
#         num_nodes = 106
#         for i in range(num_nodes):
#             for j in range(i-int(k/2), i+int(k/2)):
#                 edge_index_i.append(i)
#                 if j < 0:
#                     edge_index_j.append(num_nodes+j)
#                 elif j > num_nodes-1:
#                     edge_index_j.append(j-num_nodes)
#                 else:
#                     edge_index_j.append(j)
        
#         edge_index_i = torch.from_numpy(np.asarray(edge_index_i))
#         edge_index_j = torch.from_numpy(np.asarray(edge_index_j))
#         self.edge_index_106 = torch.stack([edge_index_i,edge_index_j]).to('cuda')
        
#         edge_index_i = []
#         edge_index_j = []
#         num_nodes = 318
#         for i in range(num_nodes):
#             for j in range(i-int(k/2), i+int(k/2)):
#                 edge_index_i.append(i)
#                 if j < 0:
#                     edge_index_j.append(num_nodes+j)
#                 elif j > num_nodes-1:
#                     edge_index_j.append(j-num_nodes)
#                 else:
#                     edge_index_j.append(j)
        
#         edge_index_i = torch.from_numpy(np.asarray(edge_index_i))
#         edge_index_j = torch.from_numpy(np.asarray(edge_index_j))
#         self.edge_index_318 = torch.stack([edge_index_i,edge_index_j]).to('cuda')
        
#         edge_index_i = []
#         edge_index_j = []
#         num_nodes = 954
#         for i in range(num_nodes):
#             for j in range(i-int(k/2), i+int(k/2)):
#                 edge_index_i.append(i)
#                 if j < 0:
#                     edge_index_j.append(num_nodes+j)
#                 elif j > num_nodes-1:
#                     edge_index_j.append(j-num_nodes)
#                 else:
#                     edge_index_j.append(j)
        
#         edge_index_i = torch.from_numpy(np.asarray(edge_index_i))
#         edge_index_j = torch.from_numpy(np.asarray(edge_index_j))
#         self.edge_index_954 = torch.stack([edge_index_i,edge_index_j]).to('cuda')
        


#         self.bn_relu_64  = nn.Sequential(
#                                 # Input 1024x32
#                                 nn.BatchNorm1d(64),
#                                 nn.LeakyReLU(0.2, inplace=True),
#                                 # 1024x32
#                                 )
        
#         self.block1_in  = nn.Sequential(
#                                 # Input 1024x32
#                                 nn.Linear(in_features=64, out_features=64, bias=True),
#                                 nn.BatchNorm1d(64),
#                                 nn.LeakyReLU(0.2, inplace=True),
#                                 # 1024x32
#                                 )

#         self.block1     = PointTransformerLayer(in_features=64, out_features=64, k=16, bias=True, use_eta=self.use_eta)
        self.block1_1     = GATConv(in_channels=self.in_features, out_channels=64, heads=1)
    
        self.bn_relu_32_1  = nn.Sequential(
                                # Input 1024x32
#                                 nn.BatchNorm1d(32),
                                nn.LeakyReLU(0.2, inplace=True),
                                # 1024x32
                                )
        
        self.block1_2     = GATConv(in_channels=64, out_channels=64, heads=1)
        self.block1_3     = GATConv(in_channels=64, out_channels=3, heads=1)
                                # 1024x32

#         self.block1_out = nn.Sequential(
#                                 nn.LeakyReLU(0.2, inplace=True),
#                                 nn.Linear(in_features=64, out_features=64, bias=True),
#                                 nn.BatchNorm1d(64),
#                                 nn.LeakyReLU(0.2, inplace=True),
#                                 # 1024x32
#                                 )
        
#         self.transition_up1 = torch.nn.Upsample(scale_factor=3, mode='linear', align_corners=True)
#         self.transition_up1 = torch.nn.Upsample(scale_factor=3, mode='nearest')

#         self.lin1     = GATConv(in_channels=64, out_channels=32, heads=1)
        
#         self.lin1 = nn.Sequential(
#                                 # Input 256x32
#                                 nn.Linear(in_features=64, out_features=32, bias=True),
#                                 nn.BatchNorm1d(32),
#                                 nn.LeakyReLU(0.2, inplace=True),
#                                 # 256x64
#                                 )
        
#         self.block2_in  = nn.Sequential(
#                                 # Input 256x64
#                                 nn.Linear(in_features=32, out_features=32, bias=True),
#                                 nn.LeakyReLU(0.2, inplace=True),
#                                 # 256x64
#                                 )

#         self.block2     = PointTransformerLayer(in_features=32, out_features=32, k=16, bias=True, use_eta=self.use_eta)
#         self.block2_1     = GATConv(in_channels=64, out_channels=32, heads=1)
    
#         self.bn_relu_32_2  = nn.Sequential(
#                                 # Input 1024x32
# #                                 nn.BatchNorm1d(32),
#                                 nn.LeakyReLU(0.2, inplace=True),
#                                 # 1024x32
#                                 )
        
#         self.block2_2     = GATConv(in_channels=32, out_channels=32, heads=1)
        
#         self.block2_out = nn.Sequential(
#                                 nn.LeakyReLU(0.2, inplace=True),
#                                 nn.Linear(in_features=32, out_features=32, bias=True),
#                                 nn.BatchNorm1d(32),
#                                 nn.LeakyReLU(0.2, inplace=True),
#                                 # 256x64
#                                 )
        
#         self.transition_up2 = torch.nn.Upsample(scale_factor=3, mode='linear', align_corners=True)
#         self.transition_up2 = torch.nn.Upsample(scale_factor=3, mode='nearest')
        
#         self.lin2     = GATConv(in_channels=32, out_channels=3, heads=1)
        
#         self.bn_relu_3  = nn.Sequential(
#                                 # Input 1024x32
#                                 nn.BatchNorm1d(3),
#                                 nn.LeakyReLU(0.2, inplace=True),
#                                 # 1024x32
#                                 )
        
#         self.lin2 = nn.Sequential(
#                                 # Input 64x64
#                                 nn.Linear(in_features=32, out_features=3, bias=True),
#                                 nn.BatchNorm1d(3),
#                                 nn.LeakyReLU(0.2, inplace=True),
#                                 # 64x128
#                                 )


#         self.block3_1     = GATConv(in_channels=32, out_channels=32, heads=1)
    
#         self.bn_relu_32_3  = nn.Sequential(
#                                 # Input 1024x32
# #                                 nn.BatchNorm1d(32),
#                                 nn.LeakyReLU(0.2, inplace=True),
#                                 # 1024x32
#                                 )
        
#         self.block3_2     = GATConv(in_channels=32, out_channels=3, heads=1)
            
#         self.block3_out = nn.Sequential(
#                                 nn.LeakyReLU(0.2, inplace=True),
#                                 nn.Linear(in_features=3, out_features=3, bias=True),
# #                                 nn.BatchNorm1d(3),
# #                                 nn.LeakyReLU(0.2, inplace=True),
#                                 # 64x128
#                                 )
        
#         self.edge_block     = PointTransformerLayerNoPos(in_features=3, out_features=3, k=16, bias=True, use_eta=self.use_eta)

#         self.edge_mlp = nn.Sequential(
#                                 nn.Linear(in_features=6, out_features=32, bias=True),
#                                 nn.LeakyReLU(0.2, inplace=True),
#                                 nn.Linear(in_features=32, out_features=1, bias=True),
#                                 )
        

    def forward(self, points):
        points_shape = points.shape
        
#         features = features.reshape(-1, features.shape[-1])
        points = points.reshape(points.shape[0], -1, self.in_features)
        points = points.reshape(-1, points.shape[-1])
#         batch = torch.arange(points_shape[0], dtype=torch.int64, device=points.device).view(-1, 1).repeat(1, points_shape[1]).reshape(-1)
#         edge_index = knn_graph(points, k=self.k, batch=batch, loop=False)
        
#         print(f'features shape : {features.shape}')
        
        
        edge_index_1 = self.edge_index
#         edge_index_1 = self.edge_index_954
        edge_index = []
        if points_shape[0] > 1:
            for i in range(points_shape[0]):
                edge_index.append(edge_index_1 + ((torch.max(edge_index_1)+1)*i))
            edge_index = torch.cat(edge_index, axis=1)
        else:
            edge_index = edge_index_1

        b1 = points
#         print(f'b1 shape : {b1.device}')
#         print(f'edge_index shape : {edge_index.device}')
#         b1 = self.block1_in(features)
#         b1 = self.block1(b1, points, batch)
        b1 = self.block1_1(b1, edge_index)
        b1 = self.bn_relu_32_1(b1)
        b1 = self.block1_2(b1, edge_index)
        b1 = self.bn_relu_32_1(b1)
        output = self.block1_3(b1, edge_index)
#         b1 = self.block1_out(b1)
#         b1 = b1 + points # residual connection
#         b1_out = b1 # output small mesh
#         b1 = self.bn_relu_32_1(b1)

#         print(f'b1 shape : {b1.shape}')
#         print(f'points shape : {points.shape}')
#         print(f'batch shape : {batch.shape}')
        
#         b1 = b1.reshape(points_shape[0], -1, b1.shape[-1])
#         points = points.reshape(points_shape[0], -1, points.shape[-1])
#         edge_index = knn_graph(b1 + torch.normal(torch.zeros(b1.shape), 0.001).to(b1.device), k=2, batch=batch, loop=False, cosine=False)
#         points_sparse_i = b1[edge_index[0,:]]
#         points_sparse_j = b1[edge_index[1,:]]
#         points_sparse_new = (points_sparse_i + points_sparse_j)/2
#         b1 = torch.cat((b1,points_sparse_new), dim=0)

#         print(f'b1 shape : {b1.shape}')
#         b1 = self.transition_up1(b1.permute(0,2,1)).permute(0,2,1)
        
# #         points = self.transition_up1(points.permute(0,2,1)).permute(0,2,1)
#         b1 = b1.reshape(points_shape[0], -1, b1.shape[-1])
#         print(f'b1 shape : {b1.shape}')
#         batch = torch.arange(b1.shape[0], dtype=torch.int64, device=b1.device).view(-1, 1).repeat(1, b1.shape[1]).reshape(-1)
#         b1 = b1.reshape(-1, b1.shape[-1])
#         print(f'b1 shape : {b1.shape}')
        
#         edge_index_1 = self.edge_index_318
#         edge_index = []
#         if points_shape[0] > 1:
#             for i in range(points_shape[0]):
#                 edge_index.append(edge_index_1 + ((torch.max(edge_index_1)+1)*i))
#             edge_index = torch.cat(edge_index, axis=1)
#         else:
#             edge_index = edge_index_1
#         print(f'edge_index shape : {edge_index.shape}')
        
# #         points = points.reshape(-1, points.shape[-1])
#         edge_index = knn_graph(b1, k=self.k, batch=batch, loop=False)
        
#         print(f'b1 shape : {b1.shape}')
#         print(f'points shape : {points.shape}')
#         print(f'batch shape : {batch.shape}')

#         b2 = self.block2_1(b1, edge_index)
#         b2 = self.bn_relu_32_2(b2)
#         b2 = self.block2_2(b2, edge_index)
# #         b2 = self.block2_out(b2)
# #         b2 = b2 + b1 # residual connection
#         b2_out = b2 # output small mesh
#         b2 = self.bn_relu_32_2(b2)
# #         b2 = self.bn_relu_32(b2)
        
#         print(f'b2 shape : {b2.shape}')
#         print(f'points shape : {points.shape}')
#         print(f'batch shape : {batch.shape}')
        
#         b2 = b2.reshape(points_shape[0], -1, b2.shape[-1])
#         points = points.reshape(points_shape[0], -1, points.shape[-1])
        
#         b2 = self.transition_up2(b2.permute(0,2,1)).permute(0,2,1)
#         print(f'b2 shape : {b2.shape}')
#         edge_index = knn_graph(b2 + torch.normal(torch.zeros(b1.shape), 0.001).to(b2.device), k=2, batch=batch, loop=False, cosine=False)
#         points_sparse_i = b2[edge_index[0,:]]
#         points_sparse_j = b2[edge_index[1,:]]
#         points_sparse_new = (points_sparse_i + points_sparse_j)/2
#         b2 = torch.cat((b2,points_sparse_new), dim=0)
# #         print(f'b2 shape : {b2.shape}')
# #         points = self.transition_up2(points.permute(0,2,1)).permute(0,2,1)

# #         print(f'b2 shape : {b2.shape}')
#         b2 = b2.reshape(points_shape[0], -1, b2.shape[-1])
# #         print(f'b2 shape : {b2.shape}')
# #         print(f'b2 shape : {b2.shape}')
#         batch = torch.arange(b2.shape[0], dtype=torch.int64, device=b2.device).view(-1, 1).repeat(1, b2.shape[1]).reshape(-1)
#         b2 = b2.reshape(-1, b2.shape[-1])
#         print(f'b2 shape : {b2.shape}')

#         edge_index_1 = self.edge_index_954
#         edge_index = []
#         if points_shape[0] > 1:
#             for i in range(points_shape[0]):
#                 edge_index.append(edge_index_1 + ((torch.max(edge_index_1)+1)*i))
#             edge_index = torch.cat(edge_index, axis=1)
#         else:
#             edge_index = edge_index_1
#         print(f'edge_index shape : {edge_index.shape}')
        
# #         points = points.reshape(-1, points.shape[-1])
#         edge_index = knn_graph(b2, k=self.k, batch=batch, loop=False)
        
#         print(f'b2 shape : {b2.shape}')
#         print(f'points shape : {points.shape}')
#         print(f'batch shape : {batch.shape}')
        

#         b3 = self.block3_1(b2, edge_index)
#         b3 = self.bn_relu_32_3(b3)
#         b3 = self.block3_2(b3, edge_index)
# #         b3 = self.block3_out(b3)
#         output = b3# + b2 # residual connection
        
#         print(f'output shape : {output.shape}')
        
#         edge_in = self.edge_block(output, batch)
        
#         edge_index = knn_graph(output, k=16, batch=batch, loop=False)
# #         edge_attr = torch.zeros((edge_index.shape[1],1))+0.5 ## These features are initialised to 0.5 (unsure if part of the mesh). Then the model can learn if edges are part of the mesh or not.
#         src, dst = edge_index
#         edge_attr = torch.cat([edge_in[src], edge_in[dst]], dim=-1)
#         edge_output = self.edge_mlp(edge_attr)

        output = output.reshape(points_shape[0], -1, output.shape[-1])
#         b1_out = b1_out.reshape(points_shape[0], -1, b1_out.shape[-1])
#         b2_out = b2_out.reshape(points_shape[0], -1, b2_out.shape[-1])
        
#         print(f'output shape : {output.shape}')
#         print(f'points shape : {points.shape}')
#         print(f'batch shape : {batch.shape}')
        
        return output#, b2_out, b1_out
#         return output#, edge_index, edge_output
    
    
