from MPIRenderer import MPIRenderer
from MPIRendererTorch import MPIRendererTorch
from misc import *
import matplotlib.pyplot as plt

rendererGPU = MPIRendererTorch("cpu")
basedir = "../../Data/07/"
poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
bds = poses_arr[:, -2:].transpose([1,0])
render_poses = generate_render_path_param(poses, bds, 0.2, 3.0, comps=[True, False, False], N=49)
render_poses = np.concatenate([render_poses[...,1:2], -render_poses[...,0:1], render_poses[...,2:]], -1)
img = rendererGPU.mpi_rendering(basedir, render_poses, 5).cpu().numpy()
for i in range(img.shape[0]):
    plt.imshow(img[i,...])
    plt.show()


#renderer = MPIRenderer()
#basedir = "../../Data/07/"
#poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
#poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
#bds = poses_arr[:, -2:].transpose([1,0])
#render_poses = generate_render_path_param(poses, bds, 0.2, 3.0, comps=[True, False, False], N=49)
#render_poses = np.concatenate([render_poses[...,1:2], -render_poses[...,0:1], render_poses[...,2:]], -1)
#img = renderer.mpi_rendering(basedir, render_poses, 5)
#for i in range(img.shape[0]):
#    plt.imshow(img[i,...])
#    plt.show()
