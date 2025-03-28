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
    
    return poses, pts3d, perm, w2c_mats, c2w_mats, hwf
      
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

def load_mpi(basedir):
    metadata = os.path.join(basedir, 'metadata.txt')
    mpibinary = os.path.join(basedir, 'mpi.b')
    lines = open(metadata, 'r').read().split('\n')
    h, w, d = [int(x) for x in lines[0].split(' ')[:3]]
    focal = float(lines[0].split(' ')[-1])
    data = np.frombuffer(open(mpibinary, 'rb').read(), dtype=np.uint8)/255.
    data = data.reshape([d,h,w,4]).transpose([1,2,0,3])
    #data.reshape([d, h, w, 4])
    
    data[...,-1] = np.minimum(1., data[...,-1]+1e-8)
    
    pose = np.array([[float(x) for x in l.split(' ')] for l in lines[1:5]]).T
    pose = np.concatenate([pose, np.array([h,w,focal]).reshape([3,1])], -1)
    pose = np.concatenate([-pose[:,1:2], pose[:,0:1], pose[:,2:]], 1)
    idepth, cdepth = [float(x) for x in lines[5].split(' ')[:2]]
    
    return data


def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec0_avg = up
    vec1 = normalize(np.cross(vec2, vec0_avg))
    vec0 = normalize(np.cross(vec1, vec2))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):
    hwf = poses[:3, -1:, 0]
    center = poses[:3, 3, :].mean(-1)
    vec2 = normalize(poses[:3, 2, :].sum(-1))
    vec0_avg = poses[:3, 0, :].sum(-1)
    c2w = np.concatenate([viewmatrix(vec2, vec0_avg, center), hwf], 1)
    return c2w
  
def render_path_axis_param(c2w, up, ax, rad, focal, view_range, N):
    render_poses = []
    center = c2w[:,3]
    hwf = c2w[:,4:5]
    v = c2w[:,ax] * rad
    
    for t in np.linspace(-view_range,view_range,N+1)[:-1]:
        c = center + t * v
        #z = normalize(c - (c - focal * c2w[:,2]))
        z = normalize(c - (center - focal * c2w[:,2]))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses
def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def generate_render_path_param(poses, bds, view_range, focal, comps=None, N=30):
    if comps is None:
        comps = [True]*5
    
    close_depth, inf_depth = bds[0, :].min()*.9, bds[1, :].max()*5.
    dt = .90
    mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
    #focal = mean_dz
    
    #focal = 30 
    
    shrink_factor = .8
    zdelta = close_depth * .2
    
    c2w = poses_avg(poses)
    
    #c2w = poses[:,:,0]
    up = normalize(poses[:3, 0, :].sum(-1))
    
    tt = ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(tt), 90, -1)
    render_poses = []
    print("Focal : ", focal)
    if comps[0]:
        render_poses += render_path_axis_param(c2w, up, 1, shrink_factor*rads[1], focal, view_range, N)
    if comps[1]:
        render_poses += render_path_axis_param(c2w, up, 0, shrink_factor*rads[0], focal, view_range, N)
    if comps[2]:
        render_poses += render_path_axis_param(c2w, up, 2, shrink_factor*zdelta, focal, view_range, N)
    
    rads[2] = zdelta
    
    render_poses = np.array(render_poses)
    
    return render_poses