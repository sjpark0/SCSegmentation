import numpy as np
import torch
import torch.nn.functional as F
from rendering.MPIRendererTorch import mpi_rendering, generate_render_path_param
import matplotlib.pyplot as plt
import os
import json
import math

class LKGMuxing:
    def __init__(self):
        self.device = 'cuda'

    def LoadJson(self, path):
        with open(path, 'r') as f:
            calibration = json.load(f)
            self.center = calibration['center']['value']
            self.muxingWidth = int(calibration['screenW']['value'])
            self.muxingHeight = int(calibration['screenH']['value'])
            self.cell_pattern = int(calibration['CellPatternMode']['value'])
            key_order = ['ROffsetX', 'ROffsetY', 'GOffsetX', 'GOffsetY', 'BOffsetX', 'BOffsetY']
            self.cells = torch.tensor([[d[k] for k in key_order] for d in calibration['subpixelCells']], dtype=torch.float32).to('cuda')
            t = self.cells.to('cpu')
            print(t[0, 0], t[0, 2], t[0, 4])
            print(self.cells.to('cpu'))
            #self.cells = calibration['subpixelCells']
            
            self.subpixelCellCount = len(self.cells)
            self.subpixelSize = 1.0 / (3.0 * self.muxingWidth) * (-1 if calibration['flipImageX']['value'] >= 0.5 else 1)

            self.pitch = calibration['pitch']['value'] * self.muxingWidth / calibration['DPI']['value'] * math.cos(math.atan(1.0 / calibration['slope']['value']))
            self.slope = self.muxingHeight / (self.muxingWidth * calibration['slope']['value']) * (-1 if calibration['flipImageX']['value'] >= 0.5 else 1)
            self.muxingImg = torch.zeros([self.muxingHeight, self.muxingWidth, 3])
            
    def GetPixelShift(self, val, subp, axis, cell):
        #return val
        if axis == 0:
            if subp == 0:
                return val + self.cells[cell, 0] / self.muxingWidth
            if subp == 1:
                return val + self.cells[cell, 2] / self.muxingWidth
            if subp == 2:
                return val + self.cells[cell, 4] / self.muxingWidth
        else:
            if subp == 0:
                return val + self.cells[cell, 1] / self.muxingHeight
            if subp == 1:
                return val + self.cells[cell, 3] / self.muxingHeight
            if subp == 2:
                return val + self.cells[cell, 5] / self.muxingHeight
        return val
    
    def GetCellForPixel(self, x, y):
        xPos = x * self.muxingWidth
        yPos = y * self.muxingHeight
        if self.cell_pattern == 0:
            cell = torch.zeros_like(x)
        elif self.cell_pattern == 2:
            cell = xPos % 2

        return cell
                    
    def GetSubPixelViews(self, x, y):   
        #self.subpixelCellCount = 0  
        if self.subpixelCellCount <= 0:   
            views = [x, x + self.subpixelSize, x + 2 * self.subpixelSize]
            views[0] += (y * self.slope)
            views[1] += (y * self.slope)
            views[2] += (y * self.slope)


        else:
            #cell = self.GetCellForPixel(x, y).int().cpu().numpy()
            cell = self.GetCellForPixel(x, y).int()
            views = [self.GetPixelShift(x, 0, 0, cell), self.GetPixelShift(x, 1, 0, cell), self.GetPixelShift(x, 2, 0, cell)]
            
            views[0] += self.GetPixelShift(y, 0, 1, cell) * self.slope
            views[1] += self.GetPixelShift(y, 1, 1, cell) * self.slope
            views[2] += self.GetPixelShift(y, 2, 1, cell) * self.slope
        
        views[0] *= self.pitch
        views[1] *= self.pitch
        views[2] *= self.pitch

        views[0] -= self.center
        views[1] -= self.center
        views[2] -= self.center

        views[0] = 1.0 - (views[0] - torch.floor(views[0]))
        views[1] = 1.0 - (views[1] - torch.floor(views[1]))
        views[2] = 1.0 - (views[2] - torch.floor(views[2]))

        views[0] = torch.clamp(views[0], 0, 1)
        views[1] = torch.clamp(views[1], 0, 1)
        views[2] = torch.clamp(views[2], 0, 1)
        
        return views
    
    def Muxing(self, renderImg):
        print(self.muxingWidth, self.muxingHeight, self.pitch, self.slope, self.center, self.subpixelSize, self.cell_pattern)
        numView = renderImg.shape[0]

        mapY, mapX = torch.meshgrid(torch.arange(self.muxingHeight, device=self.device), torch.arange(self.muxingWidth, device=self.device), indexing='ij')
        mapX1 = mapX.float() / self.muxingWidth
        mapY1 = (self.muxingHeight - mapY.float() - 1) / self.muxingHeight

        newX = torch.clamp((mapX1 * renderImg.shape[2]).int(), 0, renderImg.shape[2] - 1)        
        newY = torch.clamp(((1.0 - mapY1) * renderImg.shape[1]).int(), 0, renderImg.shape[1] - 1)

        views = self.GetSubPixelViews(mapX1, mapY1)
        muxingImg = torch.zeros([self.muxingHeight, self.muxingWidth, 3], device=self.device)
        for i in range(3):
            print(views[i].type)
            view1 = torch.floor(views[i] * numView).long()
            view1 = torch.clamp(view1, 0, numView - 1)
            view2 = view1 + 1
            view2 = torch.clamp(view2, 0, numView - 1)
            viewFilter = views[i] * numView - view1
            muxingImg[mapY, mapX, i] = (1 - viewFilter) * renderImg[view1, newY, newX, i] + viewFilter * renderImg[view2, newY, newX, i]

        return muxingImg
            
def LoadQuiltImage(path, row, col):
    img = plt.imread(path)
    width = int(img.shape[1] / col)
    height = int(img.shape[0] / row)

    renderImg = torch.zeros([row * col, height, width, 3], device='cuda')
    for i in range(row):
        for j in range(col):
            img1 = img[(row - i - 1) * height:(row - i) * height, j * width:(j+1) * width,:]
            renderImg[j + i * col,...] = torch.from_numpy(img1).to('cuda')
    return renderImg

with open('visual.json', 'r') as f:
    data = json.load(f)

print(data)

basedir = "../../Data/07/"
poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
bds = poses_arr[:, -2:].transpose([1,0])
render_poses = generate_render_path_param(poses, bds, 0.2, 3.0, comps=[True, False, False], N=49)
render_poses = np.concatenate([render_poses[...,1:2], -render_poses[...,0:1], render_poses[...,2:]], -1)


muxing = LKGMuxing()
muxing.LoadJson('visual.json')


#renderImg = mpi_rendering(basedir, render_poses, 5)
renderImg = LoadQuiltImage("1.png", 9, 5)
muxingImg = muxing.Muxing(renderImg)
plt.imsave("test.png", muxingImg.cpu().numpy())
#plt.imshow(muxingImg.cpu().numpy())
#plt.show()
