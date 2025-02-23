import numpy as np
import colmap_read_model as read_model
import os

def load_colmap_data(realdir):
    
    camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)
    
    # cam = camdata[camdata.keys()[0]]
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print( 'Cameras', len(cam))

    h, w, f = cam.height, cam.width, cam.params[0]
    # w, h, f = factor * w, factor * h, factor * f
    hwf = np.array([h,w,f]).reshape([3,1])
    
    imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
    imdata = read_model.read_images_binary(imagesfile)
    
    w2c_mats = []
    bottom = np.array([0,0,0,1.]).reshape([1,4])
    
    names = [imdata[k].name for k in imdata]
    print(names)
    print( 'Images #', len(names))
    perm = np.argsort(names)
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)
    
    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)
    
    poses = c2w_mats[:, :3, :4].transpose([1,2,0])
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)
    
    points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')
    pts3d = read_model.read_points3d_binary(points3dfile)
    
    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
    
    return poses, pts3d, perm, w2c_mats, c2w_mats

def computecloseInfinity(poses, pts3d, perm):
    pts_arr = []
    vis_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
        cams = [0] * poses.shape[-1]
        for ind in pts3d[k].image_ids:
            cams[ind-1] = 1
        vis_arr.append(cams)

    pts_arr = np.array(pts_arr)
    vis_arr = np.array(vis_arr)
    
    zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2,0,1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
    valid_z = zvals[vis_arr==1]
    
    cdepth = []
    idepth = []
    for i in perm:
        vis = vis_arr[:, i]
        zs = zvals[:, i]
        zs = zs[vis==1]
        close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)
        cdepth.append(close_depth)
        idepth.append(inf_depth)
    
    return cdepth, idepth

def computeOffsetByZValue(w, h, refC2W, fltW2C, refFocal, fltFocal, zValue, ptX, ptY):
    centerX = w / 2
    centerY = h / 2
    
    origin = refC2W[:,3]
    #print(origin)
    #print(ptX, ptY, centerX, centerY, refFocal, (ptX - centerX) / refFocal, (ptY - centerY) / refFocal)
    dir = refC2W[:,0:3].dot([(ptX - centerX) / refFocal, (ptY - centerY) / refFocal, 1])
    #print(dir)
    t_origin = fltW2C.dot(origin)
    #print(t_origin)
    t_dir = fltW2C.dot(dir)
    #print(t_dir)
    tr = (zValue - t_origin[2]) / t_dir[2]    
    trans = t_origin + tr * t_dir
    #print(tr, trans[0], trans[1], trans[2])

    offsetX = trans[0] / trans[2] * fltFocal + centerX - ptX
    offsetY = trans[1] / trans[2] * fltFocal + centerY - ptY
    #return int(offsetX + 0.5), int(offsetY + 0.5)
    return offsetX, offsetY

def computeSimilarity(img1, img2, boundingBox1, offsetX, offsetY):
    w = img1.shape[1]
    h = img1.shape[0]
    boundingBox2 = [boundingBox1[0] + int(offsetX), boundingBox1[1] + int(offsetY), boundingBox1[2] + int(offsetX), boundingBox1[3] + int(offsetY)]
    #print(boundingBox2)
    boundingBox2[0] = max(0, min(w - 1, boundingBox2[0]))
    boundingBox2[1] = max(0, min(h - 1, boundingBox2[1]))
    boundingBox2[2] = max(0, min(w - 1, boundingBox2[2]))
    boundingBox2[3] = max(0, min(h - 1, boundingBox2[3]))
    #print(boundingBox2)
    num = ((boundingBox2[3] - boundingBox2[1]) * (boundingBox2[2] - boundingBox2[0]))
    if num > 0:
        mean = np.sum((img1[boundingBox2[1] - int(offsetY):boundingBox2[3] - int(offsetY), boundingBox2[0] - int(offsetX):boundingBox2[2] - int(offsetX),:] - img2[boundingBox2[1]:boundingBox2[3],boundingBox2[0]:boundingBox2[2],:])**2) / num
    else:
        mean = -1
    return mean

def ComputeDepth(img, boundingBox, c2w, w2c, focals, refCamID, close_depth, inf_depth, perms):
    zstep = ((1.0 / close_depth) - (1.0 / inf_depth))
    optZ = -1.0
    optMean = 100000000.0
    
    for z in np.linspace(0, 10.0, 101):
        zValueCurrent = 1.0 / (zstep * z + 1.0 / inf_depth)
        mean = 0.0
        numAvailableCam = 0
        for i in range(len(img)):
            if perms[i] == perms[refCamID]:
                continue
            offsetX, offsetY = computeOffsetByZValue(img[perms[i]].shape[1], img[perms[i]].shape[0], c2w[perms[refCamID],:,:], w2c[perms[i],:,:], focals[perms[refCamID]], focals[perms[i]], zValueCurrent, (boundingBox[0] + boundingBox[2]) / 2, (boundingBox[1] + boundingBox[3]) / 2)
            #print(zValueCurrent, offsetX, offsetY)
            similarity = computeSimilarity(img[refCamID], img[i], boundingBox, offsetX, offsetY)
            #print(zValueCurrent, similarity)
            if similarity >= 0:
                mean += similarity
                numAvailableCam+= 1
        if numAvailableCam > 0:
            if optMean > mean / numAvailableCam:
                optMean = mean / numAvailableCam
                optZ = zValueCurrent
    return optZ

def computeOffset(img, boundingBox, c2w, w2c, focals, refCamID, close_depth, inf_depth, perms):
    offsetX = []
    offsetY = []
    optZ = ComputeDepth(img, boundingBox, c2w, w2c, focals, refCamID, close_depth, inf_depth, perms)
    for i in range(len(img)):
        off_x, off_y = computeOffsetByZValue(img[i].shape[1], img[i].shape[0], c2w[perms[refCamID],:,:], w2c[perms[i],:,:], focals[perms[refCamID]], focals[perms[i]], optZ, (boundingBox[0] + boundingBox[2]) / 2, (boundingBox[1] + boundingBox[3]) / 2)
        offsetX.append(off_x)
        offsetY.append(off_y)
    
    return optZ, offsetX, offsetY