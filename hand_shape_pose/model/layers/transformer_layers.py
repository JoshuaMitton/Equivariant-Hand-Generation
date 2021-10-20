import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import knn_graph
from torch_geometric.nn.inits import reset
from torch_geometric.data import Data

import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch_geometric.utils import softmax
import math

# import os,sys
# equivariant_attention_source =  os.path.join(os.getcwd(), '../SE3-Transformer-2/se3-transformer-public')
# if equivariant_attention_source not in sys.path:
#     sys.path.append(equivariant_attention_source)

# from equivariant_attention.modules import get_basis_and_r, GSE3Res, GNormBias, GConvSE3, GNormSE3, GMaxPooling, GAvgPooling
# from equivariant_attention.fibers import Fiber

import dgl


class PointTransformerLayer(MessagePassing):
    def __init__(self, in_features, out_features, k, bias=True, use_eta=False):
        super(PointTransformerLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.use_eta = use_eta
        
        self.psi = torch.nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=True)
        self.phi = torch.nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=True)
        self.alpha = torch.nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=True)
        
        self.delta = nn.Sequential(
                                nn.Linear(in_features=3, out_features=self.out_features, bias=True),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=True),
                                )
        
        self.gamma = nn.Sequential(
                                nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=True),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=True),
                                nn.Softmax(),
                                )
        if self.use_eta:
            self.eta = torch.nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=True)
        else:
            self.eta = None
        
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.psi)
        reset(self.phi)
        reset(self.alpha)
        reset(self.delta)
        reset(self.gamma)
        if self.use_eta:
            reset(self.eta)

    def forward(self, x, pos, batch):
        edge_index = knn_graph(pos, k=self.k, batch=batch, loop=False)
        out = self.propagate(edge_index, x=x, pos=pos, size=None)
        if self.use_eta:
            out += self.eta(x)
        return out
    
    def message(self, x_i, x_j, pos_i, pos_j):
#         print(f'pos_i shape : {pos_i.shape}')
#         print(f'pos_j shape : {pos_j.shape}')
        pos_enc = self.delta(pos_i-pos_j)
        psi_i = self.psi(x_i)
        phi_j = self.alpha(x_j)
        alpha_j = self.phi(x_j)
#         print(f'pos_enc shape : {pos_enc.shape}')
#         print(f'psi_i shape : {psi_i.shape}')
#         print(f'phi_j shape : {phi_j.shape}')
#         print(f'alpha_j shape : {alpha_j.shape}')
        feature_trans_branch = self.gamma(psi_i-phi_j+pos_enc)
        attention_gen_branch = alpha_j+pos_enc
        msg = feature_trans_branch * attention_gen_branch
        return msg
    
class PointTransformerLayerNoPos(MessagePassing):
    def __init__(self, in_features, out_features, k, bias=True, use_eta=False):
        super(PointTransformerLayerNoPos, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.use_eta = use_eta
        
        self.psi = torch.nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=True)
        self.phi = torch.nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=True)
        self.alpha = torch.nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=True)
        
#         self.delta = nn.Sequential(
#                                 nn.Linear(in_features=3, out_features=self.out_features, bias=True),
#                                 nn.LeakyReLU(0.2, inplace=True),
#                                 nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=True),
#                                 )
        
        self.gamma = nn.Sequential(
                                nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=True),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=True),
                                nn.Softmax(),
                                )
        if self.use_eta:
            self.eta = torch.nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=True)
        else:
            self.eta = None
        
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.psi)
        reset(self.phi)
        reset(self.alpha)
#         reset(self.delta)
        reset(self.gamma)
        if self.use_eta:
            reset(self.eta)

    def forward(self, x, batch):
        edge_index = knn_graph(x, k=self.k, batch=batch, loop=False)
        out = self.propagate(edge_index, x=x, size=None)
        if self.use_eta:
            out += self.eta(x)
        return out
    
    def message(self, x_i, x_j):
#         print(f'pos_i shape : {pos_i.shape}')
#         print(f'pos_j shape : {pos_j.shape}')
#         pos_enc = self.delta(pos_i-pos_j)
        psi_i = self.psi(x_i)
        phi_j = self.alpha(x_j)
        alpha_j = self.phi(x_j)
#         print(f'pos_enc shape : {pos_enc.shape}')
#         print(f'psi_i shape : {psi_i.shape}')
#         print(f'phi_j shape : {phi_j.shape}')
#         print(f'alpha_j shape : {alpha_j.shape}')
        feature_trans_branch = self.gamma(psi_i-phi_j)
        attention_gen_branch = alpha_j
        msg = feature_trans_branch * attention_gen_branch
        return msg
    
class SE3TransformerLayer(nn.Module):
    def __init__(self, in_features, out_features, k, num_degrees=4, div=4, n_heads=1, si_m='1x1', si_e='att', x_ij='add'):
        """
        Args:
            in_fibers: 
            out_fibers: 
            num_degrees: number of degrees (aka types) in hidden layer, count start from type-0
            div: (int >= 1) keys, queries and values will have (num_channels/div) channels
            n_heads: (int >= 1) for multi-headed attention
            si_m: ['1x1', 'att'] type of self-interaction in hidden layers
            si_e: ['1x1', 'att'] type of self-interaction in final layer
            x_ij: ['add', 'cat'] use relative position as edge feature
        """
        super(SE3TransformerLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        
        self.num_degrees = num_degrees
        self.num_channels = 32
        self.edge_dim = 1
        self.div = div
        self.n_heads = n_heads
        self.si_m, self.si_e = si_m, si_e
        self.x_ij = x_ij

        self.num_layers = 4
        self.pooling = 'max'
        
        self.fibers = {
                       'in': Fiber(1, 3),
                       'mid': Fiber(num_degrees, self.num_channels),
                       'out': Fiber(1, 32)
                      }
        
#         self.transformer = GSE3Res(self.fibers['in'], self.fibers['mid'], edge_dim=self.edge_dim, div=self.div, n_heads=self.n_heads)
# #         self.transformer = GSE3Res(self.in_fibers, self.out_fibers, div=self.div, n_heads=self.n_heads,
# #                                   learnable_skip=True, skip='cat', selfint=self.si_m, x_ij=self.x_ij)
#         self.normbias = GNormBias(self.fibers['mid'])
#         self.transformer2 = GConvSE3(self.fibers['mid'], self.fibers['out'], self_interaction=True, edge_dim=self.edge_dim)
        
#         self.reset_parameters()

        blocks = self._build_gcn(self.fibers, 1)
        self.Gblock, self.FCblock = blocks
        print(self.Gblock)
        print(self.FCblock)

    def reset_parameters(self):
        reset(self.transformer)
        reset(self.normbias)
        
    def _build_gcn(self, fibers, out_dim):
        # Equivariant layers
        Gblock = []
        fin = fibers['in']
        for i in range(self.num_layers):
            Gblock.append(GSE3Res(fin, fibers['mid'], edge_dim=self.edge_dim, 
                                  div=self.div, n_heads=self.n_heads, skip='sum'))
            Gblock.append(GNormSE3(fibers['mid']))
            fin = fibers['mid']
        Gblock.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim))

        # Pooling
        if self.pooling == 'avg':
            Gblock.append(GAvgPooling())
        elif self.pooling == 'max':
            Gblock.append(GMaxPooling())

        # FC layers
        FCblock = []
        FCblock.append(nn.Linear(self.fibers['out'].n_features, self.fibers['out'].n_features))
        FCblock.append(nn.ReLU(inplace=True))
        FCblock.append(nn.Linear(self.fibers['out'].n_features, 512))

        return nn.ModuleList(Gblock), nn.ModuleList(FCblock)

    def forward(self, x, pos, batch):
        edge_index = knn_graph(pos, k=self.k, batch=batch, loop=False)
        
#         G = Data(x=x, edge_index=edge_index, pos=pos)
        # Create graph (connections only, no bond or feature information yet)
        x = torch.unsqueeze(x, dim=-1)
        print(f'edge_index : {edge_index.shape}')
        print(f'x : {x.shape}')
        print(f'pos : {pos.shape}')
        print(f'batch : {batch.shape}')
        G = dgl.DGLGraph((edge_index[0,:],edge_index[1,:]))
        G.ndata['x'] = pos
        G.ndata['f'] = x
        G.edata['d'] = pos[edge_index[0,:]] - pos[edge_index[1,:]]
        G.edata['w'] = torch.ones((edge_index.shape[-1],1)).to('cuda')
        basis, r = get_basis_and_r(G, self.num_degrees-1)
#         print(f'basis : {basis}')
#         print(f'r : {r}')
        h = {'0': G.ndata['f']}
        for layer in self.Gblock:
            h = layer(h, G=G, r=r, basis=basis)

        for layer in self.FCblock:
            h = layer(h)
#         h = self.transformer(h, G=G, r=r, basis=basis)
#         h = self.normbias(h, G=G, r=r, basis=basis)
#         out = self.transformer2(h, G=G, r=r, basis=basis)
        out = h
        print(f'out : {out}')
        return out

    
class TransformerConvLayer(MessagePassing):
    def __init__(self, in_channels, out_channels,
                 heads = 1, concat = True, beta = False,
                 dropout = 0., edge_dim = None,
                 bias = True, root_weight = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(TransformerConvLayer, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()


    def forward(self, x, edge_index, edge_attr = None):
        """"""

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out += x_r

        return out


    def message(self, x_i, x_j, edge_attr,
                index, ptr, size_i):

        query = self.lin_query(x_i).view(-1, self.heads, self.out_channels)
        key = self.lin_key(x_j).view(-1, self.heads, self.out_channels)

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            key += edge_attr

        alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = self.lin_value(x_j).view(-1, self.heads, self.out_channels)
        if edge_attr is not None:
            out += edge_attr

        out *= alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)