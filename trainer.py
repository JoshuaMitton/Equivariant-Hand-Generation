import argparse
import os
import os.path as osp
import sys
sys.path.append('data/')
sys.path.append('hand_shape_pose/loss')
sys.path.append('hand_shape_pose/loss/chamfer_distance')

import csv

import os
os.umask(0o002)

# H5 File bug over network file system.
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm

# from torch_geometric.nn.unpool import knn_interpolate
from torch_scatter import scatter_mean

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)

# from hand_shape_pose.config import cfg
from hand_shape_pose.model.shape_pose_network import ShapePoseNetwork
# from hand_shape_pose.model.PointTransformer import PointTransformer, MLP
# from hand_shape_pose.model.TransformerConv import TransformerConv
# from hand_shape_pose.model.GATModel import GATConvModel
from hand_shape_pose.model.encoder import Encoder, E2Encoder
from hand_shape_pose.data.build import build_dataset

import emlp.nn.pytorch as emlpnn
from emlp.nn.pytorch import EMLPBlock, Linear
from emlp.reps import Scalar,V,T,Rep
from emlp.groups import SO
import logging

# Folder permissions for cluster.
os.umask(0o002)
# H5 File bug over network file system.
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# class EMLP(nn.Module):
#         """ Equivariant MultiLayer Perceptron. 
#             If the input ch argument is an int, uses the hands off uniform_rep heuristic.
#             If the ch argument is a representation, uses this representation for the hidden layers.
#             Individual layer representations can be set explicitly by using a list of ints or a list of
#             representations, rather than use the same for each hidden layer.
#             Args:
#                 rep_in (Rep): input representation
#                 rep_out (Rep): output representation
#                 group (Group): symmetry group
#                 ch (int or list[int] or Rep or list[Rep]): number of channels in the hidden layers
#                 num_layers (int): number of hidden layers
#             Returns:
#                 Module: the EMLP objax module."""
#         def __init__(self,rep_in,rep_out,group,ch=384,num_layers=3):
#             super().__init__()
#             logging.info("Initing EMLP (PyTorch)")
#             self.rep_in =rep_in(group)
#             self.rep_out = rep_out(group)

#             self.G=group
#             # Parse ch as a single int, a sequence of ints, a single Rep, a sequence of Reps
# #             if isinstance(ch,int): middle_layers = num_layers*[ch*Scalar(self.G)+ch*V(self.G)+ch*V(self.G)**2]
#             if isinstance(ch,int): middle_layers = [uniform_rep(ch,group) for _ in range(num_layers)]
#             elif isinstance(ch,Rep): middle_layers = num_layers*[ch(group)]
#             else: middle_layers = [(c(group) if isinstance(c,Rep) else uniform_rep(c,group)) for c in ch]
#             #assert all((not rep.G is None) for rep in middle_layers[0].reps)
#             reps = [self.rep_in]+middle_layers
#             #logging.info(f"Reps: {reps}")
#             self.network = nn.Sequential(
#                 *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
#                 Linear(reps[-1],self.rep_out)
#             )
#         def forward(self,x):
#             return self.network(x)

def test(encoder, reshaper, mesh_decoder, data_loader_val, criterion, csvlog, args):
# def test(encoder, lin_projector, mesh_decoder, data_loader_val, chamfer_dist, csvlog, args):
    encoder.eval()
    reshaper.eval()
    mesh_decoder.eval()
    tot_loss = 0

    for i, batch in enumerate(data_loader_val):
        ## Get data
        images, cam_params, bboxes, pose_roots, pose_scales, mesh_pts_gt, mesh_normal_gt, mesh_tri_idx, image_ids = batch
        images = images.permute(0,3,1,2).float().to(args.device)
        mesh_pts_gt = mesh_pts_gt.float().to(args.device)
        
        if args.dataset == 'real_world_testset':
            # Normalise mesh points to 0.5-0.5 for real world data
            mesh_pts_gt += 16.8738
            mesh_pts_gt /= 67.1409
            mesh_pts_gt -= 0.5

        elif args.dataset == 'synthetic_train_val':
            # Normalise mesh points to 0.5-0.5 for synthetic data
            mesh_pts_gt += 10.2785
            mesh_pts_gt /= 36.9131
            mesh_pts_gt -= 0.5

        ## Create latent space
        latent = encoder(images)

        ## Get some initial point estimate for point transformer layers
        points = reshaper(latent)

        ## Mesh decoder
        mesh = mesh_decoder(points)
        mesh = mesh.view(mesh.shape[0], -1, 3)

        loss = criterion(mesh, mesh_pts_gt)

        tot_loss += loss.item()

    return tot_loss

def test_plot(encoder, reshaper, mesh_decoder, batch, criterion, csvlog, epoch, args):
# def test(encoder, lin_projector, mesh_decoder, data_loader_val, chamfer_dist, csvlog, args):
    encoder.eval()
    reshaper.eval()
    mesh_decoder.eval()
    tot_loss = 0

    ## Get data
    images, cam_params, bboxes, pose_roots, pose_scales, mesh_pts_gt, mesh_normal_gt, mesh_tri_idx, image_ids = batch
#     print(images)
    images = torch.unsqueeze(images.permute(2,0,1).float().to(args.device),axis=0)
#     print(mesh_pts_gt)
#     mesh_pts_gt = torch.from_numpy(mesh_pts_gt).float().to(args.device)
#     print(mesh_pts_gt)

    if args.dataset == 'real_world_testset':
        # Normalise mesh points to 0.5-0.5 for real world data
        mesh_pts_gt += 16.8738
        mesh_pts_gt /= 67.1409
        mesh_pts_gt -= 0.5

    elif args.dataset == 'synthetic_train_val':
        # Normalise mesh points to 0.5-0.5 for synthetic data
        mesh_pts_gt += 10.2785
        mesh_pts_gt /= 36.9131
        mesh_pts_gt -= 0.5

    ## Create latent space
    latent = encoder(images)

    ## Get some initial point estimate for point transformer layers
    points = reshaper(latent)

    ## Mesh decoder
    mesh = mesh_decoder(points)
    mesh = mesh.view(mesh.shape[0], -1, 3)

#     print(f'mesh_pts_gt shape : {mesh_pts_gt.shape}')
#     print(f'mesh_tri_idx shape : {mesh_tri_idx.shape}')
#     print(f'mesh shape : {mesh.shape}')
    
    fig = plt.figure(figsize=plt.figaspect(0.2))
    
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    mesh_2d = np.squeeze(mesh_pts_gt)
    mesh_tri_idx = np.squeeze(mesh_tri_idx)
    ax.plot_trisurf(mesh_2d[:,0], mesh_2d[:,1], mesh_2d[:,2], triangles=mesh_tri_idx, color='grey', alpha=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=305, azim=75)
    ax.set_title(f"Original mesh")
    
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    mesh_2d = torch.squeeze(mesh).detach().cpu().numpy()
    mesh_tri_idx = np.squeeze(mesh_tri_idx)
    ax.plot_trisurf(mesh_2d[:,0], mesh_2d[:,1], mesh_2d[:,2], triangles=mesh_tri_idx, color='grey', alpha=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=305, azim=75)
    ax.set_title(f"Predicted mesh")

    plt.savefig(f'outputs/{args.savedir}/meshs_{epoch}.png')
#     plt.show()
    plt.close()

    return tot_loss

def main():
    
    parser = argparse.ArgumentParser(description='PathologyGAN trainer.')
    parser.add_argument('--device',          dest='device',          type=str,            default='cuda',           help='cuda or cpu.')
    parser.add_argument('--encoder',         dest='encoder',         type=str,            default='Conv', help='Type of encoder model - options: (1)Conv=Conv ResNet (2)EConv=E2 Equivariant Conv ResNet.')
    parser.add_argument('--decoder',         dest='decoder',         type=str,            default='MLP', help='Type of decoder model - options: (1)GNN=GATConv (2)MLP=MLP (3)EMLP=SO(3) Equivariant MLP.')
    parser.add_argument('--num_workers',     dest='num_workers',     type=int,            default=0,                help='Number of workers for the dataloader.')
    parser.add_argument('--batch_size',      dest='batch_size',      type=int,            default=8,                help='Batch size for dataloader.')
    parser.add_argument('--num_joints',      dest='num_joints',      type=int,            default=21,               help='Number of joints.')
    parser.add_argument('--dataset',         dest='dataset',         type=str,            default='real_world_testset',               help='Dataset name.')
    parser.add_argument('--savedir',         dest='savedir',         type=str,            default='run0',               help='Dataset name.')
    parser.add_argument('--epochs',          dest='epochs',          type=int,            default=20,               help='Dataset name.')
    parser.add_argument('--lr',              dest='lr',              type=float,          default=0.001,               help='Dataset name.')
    args             = parser.parse_args()
    print(args)


    # 1. Load data
    dataset = build_dataset(args.dataset)
    print(len(dataset))

    if args.dataset == 'real_world_testset':
        dataset_train, dataset_val = torch.utils.data.random_split(dataset, [500, 83], generator=torch.Generator().manual_seed(0))
#         dataset_train, dataset_val = torch.utils.data.random_split(dataset, [1, 582], generator=torch.Generator().manual_seed(0))
    elif args.dataset == 'synthetic_train_val':
        train_len = int(len(dataset) * 0.8) #40000
        dataset_train, dataset_val = torch.utils.data.random_split(dataset, [train_len, len(dataset)-train_len], generator=torch.Generator().manual_seed(0))
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    ## Get an adjacency matrix to use for all graph
    batch = dataset_train[0]
    images, cam_params, bboxes, pose_roots, pose_scales, mesh_pts_gt, mesh_normal_gt, mesh_tri_idx, image_ids = batch
    edge_index = torch.from_numpy(np.transpose(np.concatenate([mesh_tri_idx[:,[0,1]],mesh_tri_idx[:,[1,2]],mesh_tri_idx[:,[0,2]]], axis=0))).type(torch.LongTensor).to(args.device)
    
    # 2. Create models
    if args.encoder == 'Conv':
        encoder = Encoder()
        encoder = encoder.to(args.device)
        print(encoder)
    elif args.encoder == 'EConv':
        encoder = E2Encoder()
        encoder = encoder.to(args.device)
        print(encoder)

    ## Pull out points from CNN latent space and reshape to be a hand point cloud.
    reshaper = torch.nn.Sequential(
#                     torch.nn.Linear(4096, 30528), # 954x32
#                     torch.nn.Linear(4096, 15264), # 954x32
                    torch.nn.Linear(4096, 954), # 954x32
                    torch.nn.ReLU()
                    )
    reshaper = reshaper.to(args.device)
    print(reshaper)

    if args.decoder == 'GNN':
        ## GNN
        mesh_decoder = GATConvModel(16, edge_index)
        mesh_decoder = mesh_decoder.to(args.device)
        print(mesh_decoder)
        
    elif args.decoder == 'MLP':
        ## Point estimator
        mesh_decoder = torch.nn.Sequential(
#                         torch.nn.Linear(30528, 8000),
                        torch.nn.Linear(15264, 8000),
                        torch.nn.ReLU(),
                        torch.nn.Linear(8000, 2862) #954x3
                        )
        mesh_decoder = mesh_decoder.to(args.device)
        print(mesh_decoder)
        
    elif args.decoder == 'EMLP':
        repin= 318*V # Setup some example data representations
        repmid = 318*V+10*V**2
        repout = 954*V
        G = SO(3) # The lorentz group
        
        mesh_decoder = emlpnn.EMLP(repin,repout,G,num_layers=2,ch=repmid)
#         mesh_decoder = EMLP(repin,repout,G,num_layers=3,ch=repmid)

        ## Point estimator
#         mesh_decoder = torch.nn.Sequential(
# #                         torch.nn.Linear(30528, 8000),
#                         torch.nn.Linear(15264, 8000),
#                         torch.nn.ReLU(),
#                         torch.nn.Linear(8000, 2862) #954x3
#                         )
        mesh_decoder = mesh_decoder.to(args.device)
        print(mesh_decoder)
        
    print(f'trainable params for decoder : {sum(p.numel() for p in mesh_decoder.parameters() if p.requires_grad)}')


    
    # 3. Create optimiser and loss
    criterion = torch.nn.MSELoss()
    
    def laplace_loss(output, target, edge_index):
        edge_index = edge_index.detach().cpu().numpy()
        edge_index = torch.from_numpy(edge_index).to('cuda')
        edge_index_src = edge_index[0,:].view(-1)
        edge_index_des = edge_index[1,:].view(-1)
        delta_k_pred = output[:,edge_index_des,:]
        delta_k_true = target[:,edge_index_des,:]
        delta_k = delta_k_pred - delta_k_true
        delta_k_mean = scatter_mean(delta_k, edge_index_src, dim=1)
        
        delta_i = output - target
        
        loss = torch.mean((delta_i - delta_k_mean)**2)
        return loss
    
    optimizer = torch.optim.Adam(list(encoder.parameters())+list(reshaper.parameters())+list(mesh_decoder.parameters()), lr=args.lr)
    
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
    
    # 4. Run training
    if not os.path.isdir(f'outputs/{args.savedir}'):
        os.makedirs(f'outputs/{args.savedir}')
    
    csvfile =  open(f'outputs/{args.savedir}/log.csv', 'w', newline='')
    csvlog = csv.writer(csvfile)
        
    for epoch in range(args.epochs):
        encoder.train()
        reshaper.train()
        mesh_decoder.train()
        tot_loss = 0
        tot_loss_mse = 0
        tot_loss_laplace = 0

        for i, batch in enumerate(data_loader_train):
            ## Get data
            images, cam_params, bboxes, pose_roots, pose_scales, mesh_pts_gt, mesh_normal_gt, mesh_tri_idx, image_ids = batch
            images = images.permute(0,3,1,2).float().to(args.device)
            mesh_pts_gt = mesh_pts_gt.float().to(args.device)

            if args.dataset == 'real_world_testset':
                # Normalise mesh points to 0.5-0.5 for real world data
                mesh_pts_gt += 16.8738
                mesh_pts_gt /= 67.1409
                mesh_pts_gt -= 0.5
            
            elif args.dataset == 'synthetic_train_val':
                # Normalise mesh points to 0.5-0.5 for synthetic data
                mesh_pts_gt += 10.2785
                mesh_pts_gt /= 36.9131
                mesh_pts_gt -= 0.5

            # zero the parameter gradients
            optimizer.zero_grad()

            ## Create latent space
            latent = encoder(images)

            ## Get some initial point estimate for point transformer layers
            points = reshaper(latent)

            ## Mesh decoder
            mesh = mesh_decoder(points)
            mesh = mesh.view(mesh.shape[0], -1, 3)
            
            loss_mse = criterion(mesh, mesh_pts_gt)
            loss_laplace = 10*laplace_loss(mesh, mesh_pts_gt, edge_index)

            if epoch < 50:
                loss = loss_mse
            else:
                loss = loss_mse + loss_laplace

            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
            tot_loss_mse += loss_mse.item()
            tot_loss_laplace += loss_laplace.item()


        ## Optional run tests at each epoch or some subset
        test_loss = test(encoder, reshaper, mesh_decoder, data_loader_val, criterion, csvlog, args)

        _ = test_plot(encoder, reshaper, mesh_decoder, dataset_val[0], criterion, csvlog, epoch, args)
        
        scheduler.step()

        print(f'[{epoch+1}] - loss: {tot_loss/len(data_loader_train):.5f} - test loss: {test_loss/len(data_loader_val):.6f}')
        print(f'[{epoch+1}] - loss mse: {tot_loss_mse/len(data_loader_train):.6f} - loss lapace: {tot_loss_laplace/len(data_loader_train):.6f}')

        csvlog.writerows([[f'{epoch+1}', f'{tot_loss/len(data_loader_train):.5f}', f'{test_loss/len(data_loader_val):.6f}']])
        
        # 5. Save models
        if (epoch % 10) == 9:
            if not os.path.isdir(f'outputs/{args.savedir}'):
                os.makedirs(f'outputs/{args.savedir}')
            torch.save(encoder.state_dict(), f'outputs/{args.savedir}/encoder_{epoch}.pt')
            torch.save(reshaper.state_dict(), f'outputs/{args.savedir}/reshaper_{epoch}.pt')
            torch.save(mesh_decoder.state_dict(), f'outputs/{args.savedir}/mesh_decoder_{epoch}.pt')
        
    # 5. Save models
    if not os.path.isdir(f'outputs/{args.savedir}'):
        os.makedirs(f'outputs/{args.savedir}')
    torch.save(encoder.state_dict(), f'outputs/{args.savedir}/encoder.pt')
    torch.save(reshaper.state_dict(), f'outputs/{args.savedir}/reshaper.pt')
    torch.save(mesh_decoder.state_dict(), f'outputs/{args.savedir}/mesh_decoder.pt')


if __name__ == "__main__":
    main()