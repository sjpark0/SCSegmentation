import numpy as np
import os
import time
import torch
import torch.nn.functional as F
from misc import *

class MPIRendererTorch:
    def __init__(self, device = 'cuda'):
        self.device = device

    def exp_weight_fn(self, dists, k):
        return torch.exp(-k * dists)

    def mpi_rendering(self, scenedir, render_poses, use_N):
        poses, pts3d, perm, w2c, c2w, hwf = load_colmap_data(scenedir)
        cdepth, idepth = computecloseInfinity(poses, pts3d, perm)
        close_depth = np.min(cdepth) * 0.9
        inf_depth = np.max(idepth) * 2.0
        focal = poses[2, 4, :]
        mpi = []
        for i in range(len(perm)):
            basedir = os.path.join(scenedir, 'mpis_360\\mpi{:02d}'.format(i))
            mpi.append(load_mpi(basedir))
            
        num_level = mpi[0].shape[-2]
        scaling_factor = focal[0]/close_depth
        k_scale = scaling_factor / num_level
        k_scale = 1.0
        factor = hwf[0] / mpi[0].shape[0]
        
        w2c[:,...] = w2c[perm,...]
        c2w[:,...] = c2w[perm,...]
        focal[:] = focal[perm] / factor
        poses[...,:] = poses[...,perm]

        render_poses[:,:, 1] = -render_poses[:,:, 1]
        render_poses[:,:, 2] = -render_poses[:,:, 2]

        factor = torch.tensor(factor, dtype=torch.float32, device=self.device)
        poses = torch.tensor(poses, dtype=torch.float32, device=self.device)
        render_poses = torch.tensor(render_poses, dtype=torch.float32, device=self.device)
        w2c = torch.tensor(w2c, dtype=torch.float32, device=self.device)
        focal = torch.tensor(focal, dtype=torch.float32, device=self.device)
        mpi = torch.stack([torch.tensor(m, dtype=torch.float32, device=self.device) for m in mpi], dim=0)
        renderImg = torch.zeros((render_poses.shape[0], mpi.shape[1], mpi.shape[2], 3), device=self.device)
        start = time.time()
        for i, m in enumerate(render_poses):        
            p = torch.cat((m[:,:4], torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=self.device).view(1, -1)), 0)
            dists = torch.sqrt(torch.sum(torch.square(p[:3, 3:4] - poses[:3, 3, :]), 0))
            inds = torch.argsort(dists)[:use_N]
            weights = self.exp_weight_fn(dists[inds], k_scale).reshape([-1])
            
            w2cs = torch.stack([w2c[ind] for ind in inds], 0)
            focals = torch.stack([focal[ind] for ind in inds], 0)
            mpis = torch.stack([mpi[ind] for ind in inds], 0)
            
            renderImg[i,...] = self.rendering(mpis.shape[2], mpis.shape[1], p, w2cs, m[2,4]/factor, focals, close_depth, inf_depth, num_level, mpis, weights)
        end = time.time()
        print("Time taken : ", end - start)

        return renderImg
        
    def rendering(self, w, h, refC2W, w2cs, refFocal, focals, close_depth, inf_depth, level, mpis, weights):    
        centerX = w / 2
        centerY = h / 2
        
        zstep = (1.0 / (close_depth * (level - 1))) - (1.0 / (inf_depth * (level - 1)))
        #zstep = (1.0 / (inf_depth * (level - 1))) - (1.0 / (close_depth * (level - 1)))
        
        mapY, mapX = torch.meshgrid(torch.arange(h, device=self.device), torch.arange(w, device=self.device), indexing='ij')
        mapX = mapX.float()
        mapY = mapY.float()
        
        coords = torch.stack([(mapX - centerX) / refFocal, (mapY - centerY) / refFocal, torch.ones_like(mapX)], dim=0)  # (3, H, W)
        coords = coords.view(3, -1)

        origin = refC2W[:,3]
        dir = refC2W[:,0:3].matmul(coords)
            
        for i in range(len(w2cs)):
            t_origin = w2cs[i].matmul(origin)
            t_dir = w2cs[i].matmul(dir)
            dst_color = torch.zeros((h, w, 3), device=self.device)
            dst_alpha = torch.zeros((h, w, 1), device=self.device)

            for z in range(level):
                zValue = 1.0 / (zstep * z + 1.0 / inf_depth)
                tr = (zValue - t_origin[2]) / t_dir[2,:]
                trans = t_origin.view(4, 1) + tr.view(1, -1) * t_dir
                newX = (trans[0, :] / trans[2, :] * focals[i] + centerX + 0.5) / (w - 1) * 2 - 1
                newY = (trans[1, :] / trans[2, :] * focals[i] + centerY + 0.5) / (h - 1) * 2 - 1

                
                grid = torch.stack((newX, newY), dim=-1).view(h, w, 2).unsqueeze(0)  # (1, H, W, 2)

                mpi_layer = mpis[i, :, :, z, :].permute(2, 0, 1).unsqueeze(0)  # (1, 4, H, W)

                resImg = F.grid_sample(mpi_layer, grid, mode='bilinear', padding_mode='border', align_corners=True)
                resImg = resImg.squeeze(0).permute(1, 2, 0)  # (H, W, 4)

                mpi_alpha = resImg[..., 3:4] + 1e-8
                mpi_color = resImg[..., 0:3]

                if z == 0:
                    dst_color = mpi_color * mpi_alpha
                    dst_alpha = mpi_alpha
                else:
                    dst_color = dst_color * (1 - mpi_alpha) + mpi_color * mpi_alpha
                    dst_alpha = dst_alpha * (1 - mpi_alpha) + mpi_alpha
                #dst_color = dst_color * (1 - mpi_alpha) + mpi_color * mpi_alpha
                #dst_alpha = dst_alpha * (1 - mpi_alpha) + mpi_alpha
                
                
            if i == 0:
                dst_color_final = dst_color * weights[i]
                dst_alpha_final = dst_alpha * weights[i]
            else:
                dst_color_final = dst_color_final + dst_color * weights[i]
                dst_alpha_final = dst_alpha_final + dst_alpha * weights[i]
            #dst_color_final = dst_color_final + dst_color * weights[i]
            #dst_alpha_final = dst_alpha_final + dst_alpha * weights[i]

        dst_color_final = dst_color_final / dst_alpha_final
        renderImg = torch.clamp(dst_color_final, 0, 1.0)
        return renderImg

