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

import torch
import torchvision
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm

from torch_geometric.nn.unpool import knn_interpolate
from torch_scatter import scatter_mean

import matplotlib.pyplot as plt
import numpy as np


# from hand_shape_pose.config import cfg
from hand_shape_pose.model.shape_pose_network import ShapePoseNetwork
from hand_shape_pose.model.PointTransformer import PointTransformer, MLP
from hand_shape_pose.model.TransformerConv import TransformerConv
from hand_shape_pose.model.GATModel import GATConvModel
from hand_shape_pose.model.encoder import Encoder
from hand_shape_pose.data.build import build_dataset

from chamfer_distance import ChamferDistance

# from pytorch3d import loss as pt3_loss
# import pytorch3d
# print(dir(pytorch3d))
# from pytorch3d.loss import (
#     chamfer_distance, 
#     mesh_edge_loss, 
#     mesh_laplacian_smoothing, 
#     mesh_normal_consistency,
# )

# from hand_shape_pose.util.logger import setup_logger, get_logger_filename
# from hand_shape_pose.util.miscellaneous import mkdir
# from hand_shape_pose.util.vis import save_batch_image_with_mesh_joints
# from hand_shape_pose.util import renderer

# Folder permissions for cluster.
os.umask(0o002)
# H5 File bug over network file system.
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

def test(encoder, reshaper, mesh_decoder, data_loader_val, criterion, csvlog, args):
# def test(encoder, lin_projector, mesh_decoder, data_loader_val, chamfer_dist, csvlog, args):
    encoder.eval()
    reshaper.eval()
    mesh_decoder.eval()
    tot_loss = 0
    tot_loss_mse = 0
    tot_loss_mse_2 = 0
    tot_loss_mse_1 = 0
    tot_loss_chamf = 0
    tot_loss_chamf_2 = 0
    tot_loss_chamf_1 = 0
    tot_loss_edge = 0
    for i, batch in enumerate(data_loader_val):
        ## Get data
        images, cam_params, bboxes, pose_roots, pose_scales, mesh_pts_gt, mesh_normal_gt, mesh_tri_idx, image_ids = batch
#         images, cam_params, mesh_pts_gt, mesh_normal_gt, mesh_tri_idx, image_ids = batch
#         images, mesh_pts_gt = batch
        images = images.permute(0,3,1,2).float().to(args.device)
        mesh_pts_gt = mesh_pts_gt.float().to(args.device)
#         mesh_tri_idx = mesh_tri_idx.to(args.device)
        
        # Normalise mesh points to 0-1
#         mesh_pts_gt[:,:,0] += 16.87377
#         mesh_pts_gt[:,:,1] += 10.663967
#         mesh_pts_gt[:,:,2] -= 27.635679

#         mesh_pts_gt[:,:,0] /= 41.762589
#         mesh_pts_gt[:,:,1] /= 25.609482
#         mesh_pts_gt[:,:,2] /= 22.631415
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
        
        m2 = F.interpolate(mesh_pts_gt.permute(0,2,1), scale_factor=0.333334, mode='linear', align_corners=True).permute(0,2,1)
        m1 = F.interpolate(m2.permute(0,2,1), scale_factor=0.333334, mode='linear', align_corners=True).permute(0,2,1)

#         pose_points = []
#         for image_id in image_ids:
#             pose_points.append(dataset_val.pose_gts[image_id])
#         pose_points = torch.stack(pose_points, dim=0)

        ## Create latent space
        latent = encoder(images)

        ## Get some initial point estimate for point transformer layers
        points = reshaper(latent)

        ## Mesh decoder
        mesh = mesh_decoder(points)
        mesh = mesh.view(mesh.shape[0], -1, 3)
#         mesh = points
#         mesh_2 = points
#         mesh_1 = points

#         true_edges0 = torch.cat((torch.unsqueeze(mesh_tri_idx[:,:,0], dim=-1),torch.unsqueeze(mesh_tri_idx[:,:,1], dim=-1)), dim=-1)
#         true_edges1 = torch.cat((torch.unsqueeze(mesh_tri_idx[:,:,0], dim=-1),torch.unsqueeze(mesh_tri_idx[:,:,2], dim=-1)), dim=-1)
#         true_edges2 = torch.cat((torch.unsqueeze(mesh_tri_idx[:,:,1], dim=-1),torch.unsqueeze(mesh_tri_idx[:,:,2], dim=-1)), dim=-1)
#         true_edges = torch.cat((true_edges0, true_edges1, true_edges2), dim=1)
        
#         dist1, dist2 = chamfer_dist(mesh, mesh_pts_gt)
#         loss_chamf = (torch.mean(dist1)) + (torch.mean(dist2))

        loss_mse = criterion(mesh, mesh_pts_gt)
#         loss_mse_2 = criterion(mesh_2, m2)
#         loss_mse_1 = criterion(mesh_1, m1)
#         loss_edge = edge_loss(mesh, true_edges)
#         loss_edge = mesh_edge_loss(mesh, mesh_pts_gt, true_edges)
#         loss = loss_chamf + loss_mse + loss_edge
        loss = loss_mse# + loss_edge
#         loss = loss_chamf# + loss_chamf_2 + loss_chamf_1# + loss_mse + loss_mse_2 + loss_mse_1 + loss_edge

        tot_loss += loss.item()
#         tot_loss_mse += loss_mse.item()
#         tot_loss_mse_2 += loss_mse_2.item()
#         tot_loss_mse_1 += loss_mse_1.item()
#         tot_loss_chamf += loss_chamf.item()
#         tot_loss_chamf_2 += loss_chamf_2.item()
#         tot_loss_chamf_1 += loss_chamf_1.item()
#         tot_loss_edge += loss_edge.item()


    # print statistics
#     print(f'[Test] loss: {tot_loss:.3f}')
#     print(f'[Test] loss: {tot_loss:.3f}, loss_mse: {tot_loss_mse:.3f}, loss_mesh: {tot_loss_ml:.3f}')
#     csvlog.writerows([[f'Test', f'{tot_loss:.3f}']])
    
    return tot_loss#, tot_loss_mse, tot_loss_mse_2, tot_loss_mse_1, tot_loss_chamf, tot_loss_chamf_2, tot_loss_chamf_1, tot_loss_edge
    
class MeshLoss(torch.nn.Module):
    
    def __init__(self):
        super(MeshLoss, self).__init__()

    def forward(self, output, target, mesh_idx):
        loss = 0
        for i in mesh_idx:
            for j in [(0,1), (0,2), (1,2)]:
#                 print(f'output 0 : {output[:,i[j[0]],:].shape}')
#                 print(f'output 1 : {output[:,i[j[1]],:].shape}')
#                 print(f'target 0 : {target[:,i[j[0]],:].shape}')
#                 print(f'target 1 : {target[:,i[j[1]],:].shape}')
                loss += torch.sum(((output[:,i[j[0]],:] - output[:,i[j[1]],:]) - (target[:,i[j[0]],:] - target[:,i[j[1]],:]))**2)
        return loss

def main():
    
    parser = argparse.ArgumentParser(description='PathologyGAN trainer.')
    parser.add_argument('--device',          dest='device',          type=str,            default='cuda',           help='cuda or cpu.')
    parser.add_argument('--use_gnn',         dest='use_gnn',         action='store_true',             help='Whether to use the GNN or an MLP.')
    parser.add_argument('--num_workers',     dest='num_workers',     type=int,            default=0,                help='Number of workers for the dataloader.')
    parser.add_argument('--batch_size',      dest='batch_size',      type=int,            default=8,                help='Batch size for dataloader.')
    parser.add_argument('--num_joints',      dest='num_joints',      type=int,            default=21,               help='Number of joints.')
    parser.add_argument('--dataset',         dest='dataset',         type=str,            default='real_world_testset',               help='Dataset name.')
    parser.add_argument('--savedir',         dest='savedir',         type=str,            default='run0',               help='Dataset name.')
    parser.add_argument('--epochs',          dest='epochs',          type=int,            default=20,               help='Dataset name.')
    parser.add_argument('--lr',              dest='lr',              type=float,          default=0.001,               help='Dataset name.')
    args             = parser.parse_args()
    print(args)
#     device           = args.device
#     num_workers      = args.num_workers
#     batch_size       = args.batch_size
#     num_joints       = args.num_joints


    # 1. Load data
    dataset = build_dataset(args.dataset)
    print(len(dataset))

#     images = np.load(f'data/real_world_testset_normalised/images.npy')
#     print(images.shape)
#     labels = np.load(f'data/real_world_testset_normalised/labels.npy')
#     print(labels.shape)
    
#     class CustomDataset(torch.utils.data.Dataset):
#         def __init__(self, images, labels):
#             self.images = images
#             self.labels = labels
#         def __len__(self):
#             return len(self.images)
#         def __getitem__(self, idx):
#             img_tensor = torch.from_numpy(self.images[idx])
#             label_tensor = torch.from_numpy(self.labels[idx])
#             return img_tensor, label_tensor
    
#     dataset = CustomDataset(images, labels)
#     print(len(dataset))
    

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
#     encoder = torchvision.models.resnet18(pretrained=True)
#     for param in encoder.parameters():
#         param.requires_grad = False
#     encoder.fc = torch.nn.Linear(512, 512)
#     encoder = encoder.to(args.device)
    
    encoder = Encoder()
    encoder = encoder.to(args.device)
    print(encoder)
    
#     ## Linear projector
#     lin_projector = torch.nn.Sequential(
#                     torch.nn.Linear(16384, 64000),
#                     torch.nn.ReLU(),
#                     torch.nn.Linear(64000, 61056)
#                     )
#     lin_projector = lin_projector.to(args.device)
#     print(lin_projector)

    ## Pull out points from CNN latent space and reshape to be a hand point cloud.
    reshaper = torch.nn.Sequential(
                    torch.nn.Linear(4096, 30528), # 954x32
                    torch.nn.ReLU()
                    )
    reshaper = reshaper.to(args.device)
    print(reshaper)

    print(f'use gnn : {args.use_gnn}')
    if args.use_gnn:
        mesh_decoder = GATConvModel(16, edge_index)
        mesh_decoder = mesh_decoder.to(args.device)
        print(mesh_decoder)
        
    else:
        ## Point estimator
        mesh_decoder = torch.nn.Sequential(
                        torch.nn.Linear(30528, 8000),
                        torch.nn.ReLU(),
                        torch.nn.Linear(8000, 2862) #954x3
    #                     torch.nn.Linear(8000, 61056) #954x64
    #                     torch.nn.Linear(8000, 6784) #106x64
                        )
        mesh_decoder = mesh_decoder.to(args.device)
        print(mesh_decoder)

    
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

#     chamfer_dist = ChamferDistance()
#     def edge_loss (mesh, edges):
#         edge_dist = 0
#         for i in range(mesh.shape[0]):
#             meshi = mesh[i,:,:]
#             edgesi = edges[i,:,:]
#             edge_points0 = meshi[edgesi[:,0]]
#             edge_points1 = meshi[edgesi[:,1]]
#             edge_dist += torch.mean((edge_points0-edge_points1)**2)
# #         loss = torch.mean((output - target)**2)
#         edge_dist = edge_dist / mesh.shape[0]
#         return edge_dist
#     def mesh_edge_loss (mesh, true_mesh, edges):
#         edge_dist = 0
#         for i in range(mesh.shape[0]):
#             meshi = mesh[i,:,:]
#             true_meshi = true_mesh[i,:,:]
#             edgesi = edges[i,:,:]
#             edge_points0 = meshi[edgesi[:,0]]
#             edge_points1 = meshi[edgesi[:,1]]
#             true_edge_points0 = true_meshi[edgesi[:,0]]
#             true_edge_points1 = true_meshi[edgesi[:,1]]
#             edge_dist += torch.mean(torch.abs((edge_points0-edge_points1) - (true_edge_points0-true_edge_points1)))
# #         loss = torch.mean((output - target)**2)
#         edge_dist = edge_dist / mesh.shape[0]
#         return edge_dist
#     criterion_ml = MeshLoss()
#     optimizer = torch.optim.Adam(list(encoder.parameters())+list(lin_projector.parameters())+list(point_estimator.parameters())+list(mesh_decoder.parameters()), lr=args.lr)
#     optimizer = torch.optim.Adam(list(encoder.parameters())+list(lin_projector.parameters())+list(mesh_decoder.parameters()), lr=args.lr)
    optimizer = torch.optim.Adam(list(encoder.parameters())+list(reshaper.parameters())+list(mesh_decoder.parameters()), lr=args.lr)
#     optimizer = torch.optim.Adam(list(encoder.parameters())+list(point_estimator.parameters()), lr=args.lr)
    
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
        tot_loss_mse_1 = 0
        tot_loss_chamf = 0
        tot_loss_chamf_2 = 0
        tot_loss_chamf_1 = 0
        tot_loss_edge = 0
#         for i, batch in tqdm(enumerate(data_loader_train)):
        for i, batch in enumerate(data_loader_train):
            ## Get data
            images, cam_params, bboxes, pose_roots, pose_scales, mesh_pts_gt, mesh_normal_gt, mesh_tri_idx, image_ids = batch
#             images, cam_params, mesh_pts_gt, mesh_normal_gt, mesh_tri_idx, image_ids = batch
#             images, mesh_pts_gt = batch
            images = images.permute(0,3,1,2).float().to(args.device)
            mesh_pts_gt = mesh_pts_gt.float().to(args.device)
#             mesh_tri_idx = mesh_tri_idx.to(args.device)
#             print(f'mesh_pts_gt shape : {mesh_pts_gt.shape}')
            
            # Normalise mesh points to 0-1
#             mesh_pts_gt[:,:,0] += 16.87377
#             mesh_pts_gt[:,:,1] += 10.663967
#             mesh_pts_gt[:,:,2] -= 27.635679
            
#             mesh_pts_gt[:,:,0] /= 41.762589
#             mesh_pts_gt[:,:,1] /= 25.609482
#             mesh_pts_gt[:,:,2] /= 22.631415
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
            
            m2 = F.interpolate(mesh_pts_gt.permute(0,2,1), scale_factor=0.333334, mode='linear', align_corners=True).permute(0,2,1)
            m1 = F.interpolate(m2.permute(0,2,1), scale_factor=0.333334, mode='linear', align_corners=True).permute(0,2,1)

            # zero the parameter gradients
            optimizer.zero_grad()

            ## Create latent space
            latent = encoder(images)

            ## Get some initial point estimate for point transformer layers
            points = reshaper(latent)

            ## Mesh decoder
            mesh = mesh_decoder(points)
            mesh = mesh.view(mesh.shape[0], -1, 3)
            
            
#             print(f'mesh_pts_gt {mesh_pts_gt}')
#             print(f'mesh_normal_gt {mesh_normal_gt}')
#             print(f'mesh_tri_idx {mesh_tri_idx}')
#             true_edges0 = torch.cat((torch.unsqueeze(mesh_tri_idx[:,:,0], dim=-1),torch.unsqueeze(mesh_tri_idx[:,:,1], dim=-1)), dim=-1)
#             true_edges1 = torch.cat((torch.unsqueeze(mesh_tri_idx[:,:,0], dim=-1),torch.unsqueeze(mesh_tri_idx[:,:,2], dim=-1)), dim=-1)
#             true_edges2 = torch.cat((torch.unsqueeze(mesh_tri_idx[:,:,1], dim=-1),torch.unsqueeze(mesh_tri_idx[:,:,2], dim=-1)), dim=-1)
#             true_edges = torch.cat((true_edges0, true_edges1, true_edges2), dim=1)
#             print(f'true_edges shape {true_edges.shape}')
            
#             points0 = mesh_pts_gt[0,:,:]
#             print(f'batch 0 points shape {points0.shape}')
            
#             true_edges0 = true_edges[0,:,:]
#             print(f'batch 0 true_edges shape {true_edges0.shape}')
            
#             edge_points0 = points0[true_edges0[:,0]]
#             edge_points1 = points0[true_edges0[:,1]]
#             print(f'batch 0 edge_points0 shape {edge_points0.shape}')
#             print(f'batch 0 edge_points1 shape {edge_points1.shape}')
        
#             print(f'mesh shape {mesh.shape}')
#             print(f'mesh type : {mesh.dtype}')
#             print(f'mesh_pts_gt shape {mesh_pts_gt.shape}')
#             print(f'mesh_pts_gt type : {mesh_pts_gt.dtype}')
#             loss_mse, loss_ml = criterion(mesh, mesh_pts_gt, x_normals=mesh, y_normals=mesh_pts_gt)
            loss_mse = criterion(mesh, mesh_pts_gt)
            loss_laplace = 10*laplace_loss(mesh, mesh_pts_gt, edge_index)
#             loss_mse_2 = criterion(mesh_2, m2)
#             loss_mse_1 = criterion(mesh_1, m1)
#             dist1, dist2 = chamfer_dist(mesh, mesh_pts_gt)
#             loss_chamf = (torch.mean(dist1)) + (torch.mean(dist2))
#             loss_edge = edge_loss(mesh, true_edges)
#             loss_edge = mesh_edge_loss(mesh, mesh_pts_gt, true_edges)
#             loss = loss_chamf + loss_mse + loss_edge
            if epoch < 50:
                loss = loss_mse
            else:
                loss = loss_mse + loss_laplace
#             loss = loss_chamf# + loss_chamf_2 + loss_chamf_1# + loss_mse + loss_mse_2 + loss_mse_1 + loss_edge
#             loss_ml = criterion_ml(mesh, mesh_pts_gt, mesh_tri_idx)
#             print(f'loss type : {loss.dtype}')
#             loss = loss_mse# + loss_ml
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
            tot_loss_mse += loss_mse.item()
            tot_loss_laplace += loss_laplace.item()
#             tot_loss_mse_2 += loss_mse_2.item()
#             tot_loss_mse_1 += loss_mse_1.item()
#             tot_loss_chamf += loss_chamf.item()
#             tot_loss_chamf_2 += loss_chamf_2.item()
#             tot_loss_chamf_1 += loss_chamf_1.item()
#             tot_loss_edge += loss_edge.item()

        ## Optional run tests at each epoch or some subset
#         test_loss, test_loss_mse, test_loss_mse_2, test_loss_mse_1, test_loss_chamf, test_loss_chamf_2, test_loss_chamf_1, test_loss_edge = test(encoder, point_estimator, mesh_decoder, data_loader_val, chamfer_dist, criterion, mesh_edge_loss, csvlog, args)
        test_loss = test(encoder, reshaper, mesh_decoder, data_loader_val, criterion, csvlog, args)
#         test_loss = test(encoder, lin_projector, mesh_decoder, data_loader_val, chamfer_dist, csvlog, args)

        scheduler.step()

#         print(f'[{epoch+1}] - loss mse  : {tot_loss_mse:.3f} - test loss mse  : {test_loss_mse:.3f}')
#         print(f'[{epoch+1}] - loss mse 2: {tot_loss_mse_2:.3f} - test loss mse 2: {test_loss_mse_2:.3f}')
#         print(f'[{epoch+1}] - loss mse 1: {tot_loss_mse_1:.3f} - test loss mse 1: {test_loss_mse_1:.3f}')
#         print(f'[{epoch+1}] - loss chamf  : {tot_loss_chamf:.3f} - test loss chamf  : {test_loss_chamf:.3f}')
#         print(f'[{epoch+1}] - loss chamf 2: {tot_loss_chamf_2:.3f} - test loss chamf 2: {test_loss_chamf_2:.3f}')
#         print(f'[{epoch+1}] - loss chamf 1: {tot_loss_chamf_1:.3f} - test loss chamf 1: {test_loss_chamf_1:.3f}')
#         print(f'[{epoch+1}] - loss edge: {tot_loss_edge:.3f} - test loss edge: {test_loss_edge:.3f}')
        print(f'[{epoch+1}] - loss: {tot_loss/len(data_loader_train):.6f} - test loss: {test_loss/len(data_loader_val):.6f}')
        print(f'[{epoch+1}] - loss mse: {tot_loss_mse/len(data_loader_train):.6f} - loss lapace: {tot_loss_laplace/len(data_loader_train):.6f}')

        csvlog.writerows([[f'{epoch+1}', f'{tot_loss/len(data_loader_train):.6f}', f'{test_loss/len(data_loader_val):.6f}']])
        
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

    
    # 6. Run final tests
#     test(encoder, lin_projector, point_estimator, mesh_decoder, data_loader_val, chamfer_dist, criterion, mesh_edge_loss, csvlog, args)
#     test(encoder, lin_projector, mesh_decoder, data_loader_val, chamfer_dist, csvlog, args)
    
#     # 2. Load network model
#     model = ShapePoseNetwork(cfg, output_dir)
#     device = cfg.MODEL.DEVICE
#     model.to(device)
#     model.load_model(cfg)

#     mesh_renderer = renderer.MeshRenderer(model.hand_tri.astype('uint32'))

#     # 3. Inference
#     model.eval()
#     results_pose_cam_xyz = {}
#     cpu_device = torch.device("cpu")
#     logger.info("Evaluate on {} frames:".format(len(dataset_val)))
#     for i, batch in enumerate(data_loader_val):
#         images, cam_params, bboxes, pose_roots, pose_scales, image_ids = batch
#         images, cam_params, bboxes, pose_roots, pose_scales = \
#             images.to(device), cam_params.to(device), bboxes.to(device), pose_roots.to(device), pose_scales.to(device)
#         with torch.no_grad():
#             est_mesh_cam_xyz, est_pose_uv, est_pose_cam_xyz = \
#                 model(images, cam_params, bboxes, pose_roots, pose_scales)

#             est_mesh_cam_xyz = [o.to(cpu_device) for o in est_mesh_cam_xyz]
#             est_pose_uv = [o.to(cpu_device) for o in est_pose_uv]
#             est_pose_cam_xyz = [o.to(cpu_device) for o in est_pose_cam_xyz]

#         results_pose_cam_xyz.update({img_id.item(): result for img_id, result in zip(image_ids, est_pose_cam_xyz)})

#         if i % cfg.EVAL.PRINT_FREQ == 0:
#             # 4. evaluate pose estimation
#             avg_est_error = dataset_val.evaluate_pose(results_pose_cam_xyz, save_results=False)  # cm
#             msg = 'Evaluate: [{0}/{1}]\t' 'Average pose estimation error: {2:.2f} (mm)'.format(
#                 len(results_pose_cam_xyz), len(dataset_val), avg_est_error * 10.0)
#             logger.info(msg)

#             # 5. visualize mesh and pose estimation
#             if cfg.EVAL.SAVE_BATCH_IMAGES_PRED:
#                 file_name = '{}_{}.jpg'.format(osp.join(output_dir, 'pred'), i)
#                 logger.info("Saving image: {}".format(file_name))
#                 save_batch_image_with_mesh_joints(mesh_renderer, images.to(cpu_device), cam_params.to(cpu_device),
#                                                   bboxes.to(cpu_device), est_mesh_cam_xyz, est_pose_uv,
#                                                   est_pose_cam_xyz, file_name)

#     # overall evaluate pose estimation
#     assert len(results_pose_cam_xyz) == len(dataset_val), \
#         "The number of estimation results (%d) is inconsistent with that of the ground truth (%d)." % \
#         (len(results_pose_cam_xyz), len(dataset_val))

#     avg_est_error = dataset_val.evaluate_pose(results_pose_cam_xyz, cfg.EVAL.SAVE_POSE_ESTIMATION, output_dir)  # cm
#     logger.info("Overall:\tAverage pose estimation error: {0:.2f} (mm)".format(avg_est_error * 10.0))


if __name__ == "__main__":
    main()