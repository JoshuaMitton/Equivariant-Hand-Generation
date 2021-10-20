import torch
import torch.nn as nn
from torch_geometric.nn import PointConv, fps, radius
from torch_scatter import scatter
from torch_geometric.nn import knn_graph, knn_interpolate

from hand_shape_pose.model.layers.transformer_layers import PointTransformerLayer
from hand_shape_pose.model.layers.transformer_layers import PointTransformerLayerNoPos

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


class PointTransformer(nn.Module):
    def __init__(self, ngpu, batch_size, ratio=0.25, use_eta=False):
        super(PointTransformer, self).__init__()
        
        self.ngpu = ngpu
        self.batch_size = batch_size
        self.in_pc_size = 4   
        self.ratio = ratio
        self.r = 1
        self.use_eta = use_eta

        self.bn_relu_64  = nn.Sequential(
                                # Input 1024x32
                                nn.BatchNorm1d(64),
                                nn.LeakyReLU(0.2, inplace=True),
                                # 1024x32
                                )
        
#         self.block1_in  = nn.Sequential(
#                                 # Input 1024x32
#                                 nn.Linear(in_features=64, out_features=64, bias=True),
#                                 nn.BatchNorm1d(64),
#                                 nn.LeakyReLU(0.2, inplace=True),
#                                 # 1024x32
#                                 )

#         self.block1     = PointTransformerLayer(in_features=64, out_features=64, k=16, bias=True, use_eta=self.use_eta)
        self.block1_1     = PointTransformerLayerNoPos(in_features=64, out_features=64, k=16, bias=True, use_eta=self.use_eta)
        self.block1_2     = PointTransformerLayerNoPos(in_features=64, out_features=64, k=16, bias=True, use_eta=self.use_eta)
                                # 1024x32

#         self.block1_out = nn.Sequential(
#                                 nn.LeakyReLU(0.2, inplace=True),
#                                 nn.Linear(in_features=64, out_features=64, bias=True),
#                                 nn.BatchNorm1d(64),
#                                 nn.LeakyReLU(0.2, inplace=True),
#                                 # 1024x32
#                                 )
        
        self.transition_up1 = torch.nn.Upsample(scale_factor=3, mode='linear', align_corners=True)

        self.block2_1     = PointTransformerLayerNoPos(in_features=64, out_features=64, k=16, bias=True, use_eta=self.use_eta)
        
        self.lin1 = nn.Sequential(
                                # Input 256x32
                                nn.Linear(in_features=64, out_features=32, bias=True),
                                nn.BatchNorm1d(32),
                                nn.LeakyReLU(0.2, inplace=True),
                                # 256x64
                                )
        
        self.bn_relu_32  = nn.Sequential(
                                # Input 1024x32
                                nn.BatchNorm1d(32),
                                nn.LeakyReLU(0.2, inplace=True),
                                # 1024x32
                                )
        
#         self.block2_in  = nn.Sequential(
#                                 # Input 256x64
#                                 nn.Linear(in_features=32, out_features=32, bias=True),
#                                 nn.LeakyReLU(0.2, inplace=True),
#                                 # 256x64
#                                 )

#         self.block2     = PointTransformerLayer(in_features=32, out_features=32, k=16, bias=True, use_eta=self.use_eta)
        self.block2_2     = PointTransformerLayerNoPos(in_features=32, out_features=32, k=16, bias=True, use_eta=self.use_eta)
        
#         self.block2_out = nn.Sequential(
#                                 nn.LeakyReLU(0.2, inplace=True),
#                                 nn.Linear(in_features=32, out_features=32, bias=True),
#                                 nn.BatchNorm1d(32),
#                                 nn.LeakyReLU(0.2, inplace=True),
#                                 # 256x64
#                                 )
        
        self.transition_up2 = torch.nn.Upsample(scale_factor=3, mode='linear', align_corners=True)
        
        self.block3_1     = PointTransformerLayerNoPos(in_features=32, out_features=32, k=16, bias=True, use_eta=self.use_eta)
        
        self.lin2 = nn.Sequential(
                                # Input 64x64
                                nn.Linear(in_features=32, out_features=3, bias=True),
                                nn.BatchNorm1d(3),
                                nn.LeakyReLU(0.2, inplace=True),
                                # 64x128
                                )
        
        
        
#         self.block3_in  = nn.Sequential(
#                                 # Input 64x128
#                                 nn.Linear(in_features=3, out_features=3, bias=True),
#                                 nn.BatchNorm1d(3),
#                                 nn.LeakyReLU(0.2, inplace=True),
#                                 # 64x128
#                                 )

#         self.block3     = PointTransformerLayer(in_features=3, out_features=3, k=16, bias=True, use_eta=self.use_eta)
#         self.block3_1     = PointTransformerLayerNoPos(in_features=3, out_features=3, k=16, bias=True, use_eta=self.use_eta)
        self.block3_2     = PointTransformerLayerNoPos(in_features=3, out_features=3, k=16, bias=True, use_eta=self.use_eta)
            
#         self.block3_out = nn.Sequential(
#                                 nn.LeakyReLU(0.2, inplace=True),
#                                 nn.Linear(in_features=3, out_features=3, bias=True),
# #                                 nn.BatchNorm1d(3),
# #                                 nn.LeakyReLU(0.2, inplace=True),
#                                 # 64x128
#                                 )
        
        self.edge_block     = PointTransformerLayerNoPos(in_features=3, out_features=3, k=16, bias=True, use_eta=self.use_eta)

        self.edge_mlp = nn.Sequential(
                                nn.Linear(in_features=6, out_features=32, bias=True),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Linear(in_features=32, out_features=1, bias=True),
                                )
        

    def forward(self, features, points):
        points_shape = points.shape
        
        features = features.reshape(-1, features.shape[-1])
        points = points.reshape(-1, points.shape[-1])
        batch = torch.arange(points_shape[0], dtype=torch.int64, device=points.device).view(-1, 1).repeat(1, points_shape[1]).reshape(-1)
        
#         print(f'features shape : {features.shape}')
#         print(f'points shape : {points.shape}')
#         print(f'batch shape : {batch.shape}')
        
        b1 = features
#         b1 = self.block1_in(features)
#         b1 = self.block1(b1, points, batch)
        b1 = self.block1_1(b1, batch)
        b1 = self.bn_relu_64(b1)
        b1 = self.block1_2(b1, batch)
        b1 = self.bn_relu_64(b1)
#         b1 = self.block1_out(b1)
#         b1 = b1 + features # residual connection

#         print(f'b1 shape : {b1.shape}')
#         print(f'points shape : {points.shape}')
#         print(f'batch shape : {batch.shape}')
        
        b1 = b1.reshape(points_shape[0], -1, b1.shape[-1])
        points = points.reshape(points_shape[0], -1, points.shape[-1])
        
        b1 = self.transition_up1(b1.permute(0,2,1)).permute(0,2,1)
        points = self.transition_up1(points.permute(0,2,1)).permute(0,2,1)
        batch = torch.arange(points.shape[0], dtype=torch.int64, device=points.device).view(-1, 1).repeat(1, points.shape[1]).reshape(-1)
        
        b1 = b1.reshape(-1, b1.shape[-1])
        points = points.reshape(-1, points.shape[-1])
        
#         print(f'b1 shape : {b1.shape}')
#         print(f'points shape : {points.shape}')
#         print(f'batch shape : {batch.shape}')
        
        b2 = self.block2_1(b1, batch)
        b2 = self.bn_relu_64(b2)
        b2 = self.lin1(b2)
#         b2 = self.block2_in(b1)
#         b2 = self.block2(b2, points, batch)
        b2 = self.block2_2(b2, batch)
        b2 = self.bn_relu_32(b2)
#         b2 = self.block2_out(b2)
#         b2 = b2 + b1 # residual connection
        
#         print(f'b2 shape : {b2.shape}')
#         print(f'points shape : {points.shape}')
#         print(f'batch shape : {batch.shape}')
        
        b2 = b2.reshape(points_shape[0], -1, b2.shape[-1])
        points = points.reshape(points_shape[0], -1, points.shape[-1])
        
        b2 = self.transition_up2(b2.permute(0,2,1)).permute(0,2,1)
        points = self.transition_up2(points.permute(0,2,1)).permute(0,2,1)
        batch = torch.arange(points.shape[0], dtype=torch.int64, device=points.device).view(-1, 1).repeat(1, points.shape[1]).reshape(-1)
        
        b2 = b2.reshape(-1, b2.shape[-1])
        points = points.reshape(-1, points.shape[-1])
        
#         print(f'b2 shape : {b2.shape}')
#         print(f'points shape : {points.shape}')
#         print(f'batch shape : {batch.shape}')
        
        b2 = self.block3_1(b2, batch)
        b2 = self.bn_relu_32(b2)
        b3 = self.lin2(b2)
#         b3 = self.block3_in(b2)
#         b3 = self.block3(b3, points, batch)
        b3 = self.block3_2(b3, batch)
#         b3 = self.block3_out(b3)
        output = b3# + b2 # residual connection
        
#         print(f'output shape : {output.shape}')
        
        edge_in = self.edge_block(output, batch)
        
        edge_index = knn_graph(output, k=16, batch=batch, loop=False)
#         edge_attr = torch.zeros((edge_index.shape[1],1))+0.5 ## These features are initialised to 0.5 (unsure if part of the mesh). Then the model can learn if edges are part of the mesh or not.
        src, dst = edge_index
        edge_attr = torch.cat([edge_in[src], edge_in[dst]], dim=-1)
        edge_output = self.edge_mlp(edge_attr)

        output = output.reshape(points_shape[0], -1, output.shape[-1])
        
#         print(f'output shape : {output.shape}')
#         print(f'points shape : {points.shape}')
#         print(f'batch shape : {batch.shape}')
        
        return output, edge_index, edge_output
    
    
class MLP(nn.Module):
    def __init__(self, ngpu, batch_size, ratio=0.25, use_eta=False):
        super(MLP, self).__init__()
        
        self.ngpu = ngpu
        self.batch_size = batch_size
        self.in_pc_size = 4   
        self.ratio = ratio
        self.r = 1
        self.use_eta = use_eta

        self.block1  = nn.Sequential(
                                # Input 1024x32
                                nn.Linear(in_features=64, out_features=64, bias=True),
#                                 nn.BatchNorm1d(64),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Linear(in_features=64, out_features=64, bias=True),
#                                 nn.BatchNorm1d(64),
                                nn.LeakyReLU(0.2, inplace=True),
                                # 1024x32
                                )

        
        self.transition_up1 = torch.nn.Upsample(scale_factor=3)

        
        self.block2in = nn.Sequential(
                                # Input 256x32
                                nn.Linear(in_features=64, out_features=32, bias=True),
#                                 nn.BatchNorm1d(32),
                                nn.LeakyReLU(0.2, inplace=True),
                                )
        
        self.block2 = nn.Sequential(
                                # Input 256x32
                                nn.Linear(in_features=32, out_features=32, bias=True),
#                                 nn.BatchNorm1d(32),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Linear(in_features=32, out_features=32, bias=True),
#                                 nn.BatchNorm1d(32),
                                nn.LeakyReLU(0.2, inplace=True),
                                # 256x64
                                )

        
        self.transition_up2 = torch.nn.Upsample(scale_factor=3)
        
        self.block3in = nn.Sequential(
                                # Input 64x64
                                nn.Linear(in_features=32, out_features=3, bias=True),
#                                 nn.BatchNorm1d(3),
                                nn.LeakyReLU(0.2, inplace=True),
                                )
        
        self.block3 = nn.Sequential(
                                # Input 64x64
                                nn.Linear(in_features=3, out_features=3, bias=True),
#                                 nn.BatchNorm1d(3),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Linear(in_features=3, out_features=3, bias=True),
#                                 nn.BatchNorm1d(3),
                                nn.LeakyReLU(0.2, inplace=True),
                                # 64x128
                                )
        

    def forward(self, features):
        points_shape = features.shape
        
        features = features.reshape(-1, features.shape[-1])

        b1 = self.block1(features)
        b1 = b1 + features # residual connection
        
        b1 = b1.reshape(points_shape[0], -1, b1.shape[-1])        
        b1 = self.transition_up1(b1.permute(0,2,1)).permute(0,2,1)
        b1 = b1.reshape(-1, b1.shape[-1])

        b1 = self.block2in(b1)

        b2 = self.block2(b1)
        b2 = b2 + b1 # residual connection

        b2 = b2.reshape(points_shape[0], -1, b2.shape[-1])        
        b2 = self.transition_up2(b2.permute(0,2,1)).permute(0,2,1)
        b2 = b2.reshape(-1, b1.shape[-1])

        b2 = self.block3in(b2)

        b3 = self.block3(b2)
        output = b3 + b2 # residual connection

        output = output.reshape(points_shape[0], -1, output.shape[-1])
        
        return output
