import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms

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
        super().__init__()
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
