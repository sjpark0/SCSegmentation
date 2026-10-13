"""S2-R6 reference-camera static snap as an in-process step (runMVSeg.py --ref-snap, folder suffix Rs).

The rule of MVSeed/s2r6_snap.py (docs/stage2-R6-prereg.md section 2, adopted in docs/stage2-R6.md), applied
in place to the run's own output folder after the crop pass: on the reference camera (MUVOD c_ini, whose
frame 0 is the given ground-truth prompt) an object's frame-t PNG (t > start) is replaced by its frame-0
PNG when

    IoU(mask_t, mask_0) >= TAU   and   mean |I_t - I_0| over dilate(mask_0, 31 px disk) <= DELTA

(I = the grayscale video frame).  Same constants, same decision function, same bytes as the post-process
(s2r6_snap.py stays as the registered script; tests/test_ref_snap.py pins the two together).

Disclosure rule (prereg section 3): a post-process designed on the test ground truth; report the gain with
LOSO +0.387 and as "reusing the given prompt on static objects", not as better segmentation.
"""
import os
import shutil

TAU, DELTA, DILATE = 0.8, 4.0, 31
SUFFIX = "Rs"


def decide(m0, mt, i0, it, np, cv2):
    """(snap?, iou, imgdiff) for one object at one frame; m0/mt bool masks, i0/it int16 images."""
    if not m0.any():
        return False, None, None
    u = np.count_nonzero(m0 | mt)
    iou = np.count_nonzero(m0 & mt) / u if u else 1.0
    reg = cv2.dilate(m0.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE, DILATE))) > 0
    diff = float(np.abs(it - i0)[reg].mean())
    return (iou >= TAU and diff <= DELTA), iou, diff


def snap_in_place(out_dir, video_dir, ref_cam, start_frame, np, cv2, keep_dir=None):
    """Apply the rule to <out_dir>/<ref_cam>/.  A snapped PNG is replaced by a fresh copy of the frame-0
    PNG (unlink + copy, so a hard link to the old file keeps the old bytes).  keep_dir: hard-link the
    whole out_dir there first (--save-stages: the stage-2 output before the snap).  Returns the stats
    s2r6_snap.py records."""
    if keep_dir is not None:
        shutil.copytree(out_dir, keep_dir, copy_function=os.link)
    cam_dir = os.path.join(out_dir, ref_cam)
    stats = dict(ref=ref_cam, rule=dict(tau=TAU, delta=DELTA, dilate=DILATE), frames=0, objects_frames=0,
                 snapped=0, no_prompt=0, missing=0, per_obj={}, snapped_files=[])
    frames = sorted((x for x in os.listdir(cam_dir) if x.isdigit()), key=int)
    if not frames or frames[0] != str(start_frame):
        raise ValueError(f"{cam_dir}: first frame folder {frames[:1]} is not start_frame {start_frame}")
    objs = sorted(os.listdir(os.path.join(cam_dir, frames[0])))
    m0 = {o: cv2.imread(os.path.join(cam_dir, frames[0], o), cv2.IMREAD_GRAYSCALE) > 127 for o in objs}
    shape = next(iter(m0.values())).shape

    def img(t):
        p = os.path.join(video_dir, "%06d.jpg" % int(t))
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise FileNotFoundError(p)
        if im.shape != shape:
            im = cv2.resize(im, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
        return im.astype(np.int16)
    i0 = img(frames[0])
    for t in frames[1:]:
        stats["frames"] += 1
        it = img(t)
        names = sorted(os.listdir(os.path.join(cam_dir, t)))
        for o in names:
            if o not in m0:
                stats["no_prompt"] += 1
                continue
            stats["objects_frames"] += 1
            p = os.path.join(cam_dir, t, o)
            mt = cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 127
            s, _, _ = decide(m0[o], mt, i0, it, np, cv2)
            po = stats["per_obj"].setdefault(o.split(".")[0], dict(snapped=0, frames=0))
            po["frames"] += 1
            if s:
                os.unlink(p)
                shutil.copyfile(os.path.join(cam_dir, frames[0], o), p)
                stats["snapped"] += 1
                po["snapped"] += 1
                stats["snapped_files"].append(f"{ref_cam}/{t}/{o}")
        stats["missing"] += len(set(objs) - set(names))
    return stats
