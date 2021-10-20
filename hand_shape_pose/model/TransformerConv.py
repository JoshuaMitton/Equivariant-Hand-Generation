import torch
import torch.nn as nn
from torch_geometric.nn import PointConv, fps, radius
from torch_scatter import scatter
# from torch_geometric.nn import TransformerConv as TransformerConvLayer
from torch_geometric.nn import knn_graph, knn_interpolate
from hand_shape_pose.model.layers.transformer_layers import TransformerConvLayer


class TransformerConv(nn.Module):
    def __init__(self, k):
        super(TransformerConv, self).__init__()
        
        self.k = k
        self.r = 1

        self.block1_t1  = TransformerConvLayer(in_channels=64, out_channels=64, bias=True)
        self.block1_t2  = TransformerConvLayer(in_channels=64, out_channels=64, bias=True)
        self.block1_t3  = TransformerConvLayer(in_channels=64, out_channels=64, bias=True)
        
        self.block1points  = TransformerConvLayer(in_channels=64+3, out_channels=3, bias=True)
        
        self.transition_up1 = torch.nn.Upsample(scale_factor=3)

        self.block2_t1  = TransformerConvLayer(in_channels=64, out_channels=32, bias=True)
        self.block2_t2  = TransformerConvLayer(in_channels=32, out_channels=32, bias=True)
        self.block2_t3  = TransformerConvLayer(in_channels=32, out_channels=32, bias=True)
        
        self.block2points  = TransformerConvLayer(in_channels=32+3, out_channels=3, bias=True)
        
        self.transition_up2 = torch.nn.Upsample(scale_factor=3)
        
        self.block3_t1 = TransformerConvLayer(in_channels=32, out_channels=3, bias=True)
        self.block3_t2 = TransformerConvLayer(in_channels=3, out_channels=3, bias=True)
        self.block3_t3 = TransformerConvLayer(in_channels=3, out_channels=3, bias=True)
        

    def forward(self, features, points):
        points_shape = points.shape
        
        features = features.reshape(-1, features.shape[-1])
        points = points.reshape(-1, points.shape[-1])
        batch = torch.arange(points_shape[0], dtype=torch.int64, device=points.device).view(-1, 1).repeat(1, points_shape[1]).reshape(-1)
        
        edge_index = knn_graph(points, k=self.k, batch=batch, loop=False)
#         edge_attr = torch.zeros((edge_index.shape[1],1))+0.5 ## These features are initialised to 0.5 (unsure if part of the mesh). Then the model can learn if edges are part of the mesh or not.
        
        print(f'features shape : {features.shape}')
        print(f'edge_index shape : {edge_index.shape}')
        print(f'block1 : {self.block1_t1}')
#         print(f'edge_attr shape : {edge_attr.shape}')
#         print(f'points shape : {points.shape}')
#         print(f'batch shape : {batch.shape}')
#         print(f'edge_index : {edge_index}')
        
        b1 = self.block1_t1(x=features, edge_index=edge_index)
        b1 = self.block1_t2(x=b1, edge_index=edge_index)
        b1 = self.block1_t3(x=b1, edge_index=edge_index)
        points = self.block1points(x=torch.cat((points, b1), dim=-1), edge_index=edge_index)

#         b1 = b1 + features # residual connection

#         print(f'b1 shape : {b1.shape}')
#         print(f'points shape : {points.shape}')
#         print(f'batch shape : {batch.shape}')
            
        b1 = b1.reshape(points_shape[0], -1, b1.shape[-1])
        points = points.reshape(points_shape[0], -1, points.shape[-1])
        
        ## Need to upsample edges and edge attrs
        b1 = self.transition_up1(b1.permute(0,2,1)).permute(0,2,1)
        points = self.transition_up1(points.permute(0,2,1)).permute(0,2,1)
        batch = torch.arange(points.shape[0], dtype=torch.int64, device=points.device).view(-1, 1).repeat(1, points.shape[1]).reshape(-1)
        
        b1 = b1.reshape(-1, b1.shape[-1])
        points = points.reshape(-1, points.shape[-1])
        
#         print(f'b1 shape : {b1.shape}')
#         print(f'points shape : {points.shape}')
#         print(f'batch shape : {batch.shape}')
        
        edge_index = knn_graph(points, k=self.k, batch=batch, loop=False)
        
        b2 = self.block2_t1(x=b1, edge_index=edge_index)
        b2 = self.block2_t2(x=b2, edge_index=edge_index)
        b2 = self.block2_t3(x=b2, edge_index=edge_index)
        points = self.block2points(x=torch.cat((points, b2), dim=-1), edge_index=edge_index)

#         b2 = b2 + b1 # residual connection
        
        b2 = b2.reshape(points_shape[0], -1, b2.shape[-1])
        points = points.reshape(points_shape[0], -1, points.shape[-1])
        
        b2 = self.transition_up2(b2.permute(0,2,1)).permute(0,2,1)
        points = self.transition_up2(points.permute(0,2,1)).permute(0,2,1)
        batch = torch.arange(points.shape[0], dtype=torch.int64, device=points.device).view(-1, 1).repeat(1, points.shape[1]).reshape(-1)
        
        b2 = b2.reshape(-1, b1.shape[-1])
        points = points.reshape(-1, points.shape[-1])
        
#         print(f'b2 shape : {b2.shape}')
#         print(f'points shape : {points.shape}')
#         print(f'batch shape : {batch.shape}')

        edge_index = knn_graph(points, k=self.k, batch=batch, loop=False)
        
        b3 = self.block3_t1(x=b2, edge_index=edge_index)
        b3 = self.block3_t2(x=b3, edge_index=edge_index)
        b3 = self.block3_t3(x=b3, edge_index=edge_index)

#         output = b3 + b2 # residual connection

#         edge_attr = torch.zeros((edge_index.shape[1],1))+0.5 ## These features are initialised to 0.5 (unsure if part of the mesh). Then the model can learn if edges are part of the mesh or not.



        output = b3.reshape(points_shape[0], -1, b3.shape[-1])
        
#         print(f'output shape : {output.shape}')
#         print(f'points shape : {points.shape}')
#         print(f'batch shape : {batch.shape}')
        
        return output
    
    