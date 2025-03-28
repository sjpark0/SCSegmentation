import numpy as np
import os
import cv2
import time
from misc import *

class MPIRenderer:
    def __init__(self):
        pass
        
    def exp_weight_fn(self, dists, k):
        return np.exp(-k * dists)

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
            #print(perm[i], c2w[i])
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

        renderImg = np.zeros((render_poses.shape[0], mpi[0].shape[0], mpi[0].shape[1], 3))
        start = time.time()
            
        for i, m in enumerate(render_poses):        
            p = np.concat((m[:,:4], np.reshape(np.array([0, 0, 0, 1]), (1, -1))), 0)
            dists = np.sqrt(np.sum(np.square(p[:3, 3:4] - poses[:3, 3, :]), 0))
            inds = np.argsort(dists)[:use_N]
            weights = self.exp_weight_fn(dists[inds], k_scale).reshape([-1])
            w2cs = np.stack([w2c[ind] for ind in inds], 0)
            focals = np.stack([focal[ind] for ind in inds], 0)
            mpis = np.stack([mpi[ind] for ind in inds], 0)
            renderImg[i,...] = self.rendering(mpis.shape[2], mpis.shape[1], p, w2cs, m[2,4]/factor, focals, close_depth, inf_depth, num_level, mpis, weights)
        end = time.time()
        print("Time taken : ", end - start)

        return renderImg
    
    def rendering(self, w, h, refC2W, w2cs, refFocal, focals, close_depth, inf_depth, level, mpis, weights):
        centerX = w / 2
        centerY = h / 2
        
        zstep = (1.0 / (close_depth * (level - 1))) - (1.0 / (inf_depth * (level - 1)))
        #zstep = (1.0 / (inf_depth * (level - 1))) - (1.0 / (close_depth * (level - 1)))
        
        map2, map1 = np.indices((h, w), dtype=np.float32)
        coords = np.asarray(np.concat([np.reshape(map1, (1, -1)), np.reshape(map2, (1, -1))], 0), np.float32)
        tm = np.stack([(coords[0] - centerX) / refFocal, (coords[1] - centerY) / refFocal, np.ones((coords[0].shape))], 0)
        origin = refC2W[:,3]
        dir = refC2W[:,0:3].dot(tm)

        #dst_color_final = np.zeros((h, w, 3))
        #dst_alpha_final = np.zeros((h, w, 1))
            
        for i in range(len(w2cs)):
            t_origin = w2cs[i].dot(origin)
            t_dir = w2cs[i].dot(dir)
            dst_color = np.zeros((h, w, 3))
            dst_alpha = np.zeros((h, w, 1))
            for z in range(level):
                zValue = 1.0 / (zstep * z + 1.0 / inf_depth)
                tr = (zValue - t_origin[2]) / t_dir[2,:]
                trans = np.reshape(t_origin, (4, -1)) + np.reshape(tr, (1, -1)) * t_dir
                newX = (trans[0,:] / trans[2,:] * focals[i] + centerX + 0.5).astype(np.float32)
                newY = (trans[1,:] / trans[2,:] * focals[i] + centerY + 0.5).astype(np.float32)
                #newX = (trans[0,:] / trans[2,:] * focals[i] + centerX + 0.5).astype(np.int32)
                #newY = (trans[1,:] / trans[2,:] * focals[i] + centerY + 0.5).astype(np.int32)
                
                newX = np.clip(newX, 0, w - 1)
                newY = np.clip(newY, 0, h - 1)

                #resImg = mpis[i, np.reshape(newY, (h, w)), np.reshape(newX, (h, w)), z, :]
                resImg = cv2.remap(mpis[i, :, :, z, :], np.reshape(newX, (h, w)), np.reshape(newY, (h, w)), cv2.INTER_LINEAR)
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
        renderImg = np.clip(dst_color_final, 0, 1.0)
        #renderImg = np.clip(dst_color_final, 0, 1.0)
        return renderImg




