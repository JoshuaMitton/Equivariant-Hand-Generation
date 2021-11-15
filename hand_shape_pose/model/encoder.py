import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as TF

from e2cnn import gspaces
from e2cnn import nn as e2nn

import numpy as np
import math

import os
os.umask(0o002)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import emlp.nn.pytorch as emlpnn
from emlp.nn.pytorch import EMLPBlock, Linear
from emlp.reps import Scalar,V,T,Rep
from emlp.groups import SO
import logging

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, padding=2)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, padding=2)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)
        return out

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.block1 = ResidualBlock(in_channels=64, out_channels=64)
        self.block2 = ResidualBlock(in_channels=64, out_channels=64)
        self.block3 = ResidualBlock(in_channels=64, out_channels=64)
        self.block4 = ResidualBlock(in_channels=64, out_channels=64)
        self.pool = torch.nn.MaxPool2d(2)
        self.fc1 = torch.nn.Linear(64 * 16 * 16, 8192)
        self.fc2 = torch.nn.Linear(8192, 8192)
        self.fc3 = torch.nn.Linear(8192, 4096)

    def forward(self, x):
#         print(f'x shape {x.shape}')
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        x = self.pool(self.block3(x))
        x = self.pool(self.block4(x))
#         print(f'x shape {x.shape}')
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
    
class E2ResidualBlock(e2nn.EquivariantModule):
    def __init__(self, type_in, type_out, type_mid=None, padding=2):
        super(E2ResidualBlock, self).__init__()
        if not type_mid:
            type_mid = type_out
        self.conv1 = e2nn.R2Conv(type_in, type_mid, kernel_size=5, padding=padding)
        self.bn1 = e2nn.InnerBatchNorm(type_mid)
        self.relu1 = e2nn.ReLU(type_mid)
        self.conv2 = e2nn.R2Conv(type_mid, type_out, kernel_size=5, padding=padding)
        self.bn2 = e2nn.InnerBatchNorm(type_out)
        self.relu2 = e2nn.ReLU(type_out)


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu2(out)
        return out
    
    def evaluate_output_shape(self, input_shape):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        if self.shortcut is not None:
            return self.shortcut.evaluate_output_shape(input_shape)
        else:
            return input_shape
    
class E2Encoder(torch.nn.Module):
    def __init__(self):
        super(E2Encoder, self).__init__()
        self.r2_act = gspaces.Rot2dOnR2(N=8)
        self.type_in  = e2nn.FieldType(self.r2_act,  3*[self.r2_act.trivial_repr])
        self.type_out = e2nn.FieldType(self.r2_act, 8*[self.r2_act.regular_repr])

        self.conv1 = e2nn.R2Conv(self.type_in, self.type_out, kernel_size=5, padding=2)
        self.bn1 = e2nn.InnerBatchNorm(self.type_out)
        self.relu = e2nn.ReLU(self.type_out)
        self.block1 = E2ResidualBlock(type_in=self.type_out, type_out=self.type_out)
        self.block2 = E2ResidualBlock(type_in=self.type_out, type_out=self.type_out)
        self.block3 = E2ResidualBlock(type_in=self.type_out, type_out=self.type_out)
        self.block4 = E2ResidualBlock(type_in=self.type_out, type_out=self.type_out)
        self.pool = self.maxpool = e2nn.PointwiseMaxPool(self.type_out, 2)
        self.fc1 = torch.nn.Linear(64 * 16 * 16, 8192)
        self.fc2 = torch.nn.Linear(8192, 8192)
        self.fc3 = torch.nn.Linear(8192, 4096)

    def forward(self, x):
        x = e2nn.GeometricTensor(x, self.type_in)
#         print(f'x shape {x.shape}')
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        x = self.pool(self.block3(x))
        x = self.pool(self.block4(x))
        x = x.tensor
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
    
class E2EncoderE2(torch.nn.Module):
    def __init__(self):
        super(E2EncoderE2, self).__init__()
        self.r2_act = gspaces.Rot2dOnR2(N=8)
        self.type_in  = e2nn.FieldType(self.r2_act,  3*[self.r2_act.trivial_repr])
        self.type_out = e2nn.FieldType(self.r2_act, 8*[self.r2_act.regular_repr])
        R = []
        for r in [0,315,270,225,180,135,90,45]:
            R.append(np.array([[np.cos(r*np.pi/180),-np.sin(r*np.pi/180)],[np.sin(r*np.pi/180),np.cos(r*np.pi/180)]]))
        R = np.stack(R)
        R = np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(R, axis=-3), axis=0), axis=0), axis=0)
        self.R = torch.from_numpy(R).float().to('cuda')

        self.conv1 = e2nn.R2Conv(self.type_in, self.type_out, kernel_size=5, padding=2)
        self.bn1 = e2nn.InnerBatchNorm(self.type_out)
        self.relu = e2nn.ReLU(self.type_out)
        self.block1 = E2ResidualBlock(type_in=self.type_out, type_out=self.type_out)
        self.block2 = E2ResidualBlock(type_in=self.type_out, type_out=self.type_out)
        self.block3 = E2ResidualBlock(type_in=self.type_out, type_out=self.type_out)
        self.block4 = E2ResidualBlock(type_in=self.type_out, type_out=self.type_out)
        self.block5 = E2ResidualBlock(type_in=self.type_out, type_out=self.type_out)
        self.pool = self.maxpool = e2nn.PointwiseMaxPool(self.type_out, 2)
        self.convout = e2nn.R2Conv(self.type_out, self.type_out, kernel_size=5, padding=2)


    def forward(self, x):
        x = e2nn.GeometricTensor(x, self.type_in)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        x = self.pool(self.block3(x))
        x = self.pool(self.block4(x))
        x = self.pool(self.block5(x))
        x = self.convout(x)           
        x = x.tensor                  

        ## Reverse group rotation action to stabilise latent representation
        x = torch.reshape(x, (-1,8,8,8,8)) # (batch,feature,group,k,k)
        x_stab = []
        for g in range(8):
            x_stab.append(TF.rotate(x[:,:,g,:,:], ((8-g)%8)*45))
        x = torch.stack(x_stab, dim=2)

        ## Rotate group axis into vector space
        x = x.permute(0,3,4,1,2) #permute from (batch, feature, group, k, k) to (batch, k, k, feature, group) so that emlp considers pairs of feature axis as 2d vectors, it wouldn't make any sense to consider adjoining pixels as 2d vectors.
        x = x.permute(0,1,2,4,3) # (batch,k,k,group,feature)
        x = torch.reshape(x, (-1,8,8,8,4,2)) # (batch,k,k,group,feature,xy)
        x = torch.einsum('bijgfx,bijgfxy->bijgfy', x, self.R)
        x = torch.mean(x, dim=3)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        return x
