#!/usr/bin/env python3
"""S2-R6 reference-camera static snap (docs/stage2-R6-prereg.md section 2).  A post-process on a
finished stage-2 output folder: on the reference camera (MUVOD c_ini, whose frame 0 is the given
ground-truth prompt), an object's frame-t mask (t >= 1) is replaced by its frame-0 mask when

    IoU(mask_t, mask_0) >= TAU   and   mean |I_t - I_0| over dilate(mask_0, 31 px disk) <= DELTA

(I = the grayscale video frame, 0..255).  Uses no ground truth beyond the frame-0 prompt that the
protocol already gives.  Every other PNG (frame 0, other cameras, objects that do not snap) is a
hard link to the source file, so the output is byte-identical to the source wherever the rule is
silent.  Writes <method>Rs next to <method> with a MANIFEST.json recording the rule and counts.
Runs in the scsam3 container (cv2, numpy):

    docker run --rm --user $(id -u):$(id -g) -v /:/host -w /host$PWD scsam3 \
        python MVSeed/s2r6_snap.py --method SegMaskSam3XW1CGPS4MSdgeoset_repFsCs10kG2
"""
import argparse, datetime, hashlib, json, os, shutil, subprocess, sys

TAU, DELTA, DILATE = 0.8, 4.0, 31          # prereg section 2: fixed before the registered run
SUFFIX = "Rs"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "Data", "MVSeg")
CFG = json.load(open(os.path.join(REPO, "SCSam3", "demo", "MVSeg.json")))
OBJ = json.load(open(os.path.join(REPO, "docs", "raw", "muvod_object_sets.json")))


def cam_name(c, d): return f"{d['prefix']}{c:0{d['prefix1']}d}"


def decide(m0, mt, i0, it, np, cv2):
    """(snap?, iou, imgdiff) for one object at one frame; m0/mt bool masks, i0/it int16 images."""
    if not m0.any():
        return False, None, None
    u = np.count_nonzero(m0 | mt)
    iou = np.count_nonzero(m0 & mt) / u if u else 1.0
    reg = cv2.dilate(m0.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE, DILATE))) > 0
    diff = float(np.abs(it - i0)[reg].mean())
    return (iou >= TAU and diff <= DELTA), iou, diff


def link_tree(src, dst):
    shutil.copytree(src, dst, copy_function=os.link)


def snap_scene(scene, method, out, np, cv2):
    d = CFG[scene]; f = d["folder"]; s0 = d["start_frame"]
    src, dst = os.path.join(DATA, f, method), os.path.join(DATA, f, out)
    ref = OBJ[scene]["c_ini"]
    stats = dict(ref=ref, frames=0, objects_frames=0, snapped=0, no_prompt=0, missing=0, per_obj={}, snapped_files=[])
    os.makedirs(dst)
    for cam in sorted(os.listdir(src)):
        if not os.path.isdir(os.path.join(src, cam)): continue
        if cam != ref:
            link_tree(os.path.join(src, cam), os.path.join(dst, cam)); continue
        vdir = os.path.join(DATA, f, "Video", cam)
        frames = sorted((x for x in os.listdir(os.path.join(src, cam)) if x.isdigit()), key=int)
        assert frames[0] == str(s0), (scene, frames[:3])
        link_tree(os.path.join(src, cam, frames[0]), os.path.join(dst, cam, frames[0]))
        objs = sorted(os.listdir(os.path.join(src, cam, frames[0])))
        m0 = {o: cv2.imread(os.path.join(src, cam, frames[0], o), cv2.IMREAD_GRAYSCALE) > 127 for o in objs}
        shape = next(iter(m0.values())).shape

        def img(t):
            im = cv2.imread(os.path.join(vdir, "%06d.jpg" % int(t)), cv2.IMREAD_GRAYSCALE)
            if im is None: sys.exit(f"missing frame {vdir}/{int(t):06d}.jpg")
            if im.shape != shape: im = cv2.resize(im, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
            return im.astype(np.int16)
        i0 = img(frames[0])
        for t in frames[1:]:
            stats["frames"] += 1
            os.makedirs(os.path.join(dst, cam, t)); it = img(t)
            names = sorted(os.listdir(os.path.join(src, cam, t)))
            for o in names:
                a = os.path.join(src, cam, t, o); b = os.path.join(dst, cam, t, o)
                if o not in m0:
                    stats["no_prompt"] += 1; os.link(a, b); continue
                stats["objects_frames"] += 1
                mt = cv2.imread(a, cv2.IMREAD_GRAYSCALE) > 127
                s, iou, diff = decide(m0[o], mt, i0, it, np, cv2)
                po = stats["per_obj"].setdefault(o.split(".")[0], dict(snapped=0, frames=0))
                po["frames"] += 1
                if s:
                    shutil.copyfile(os.path.join(src, cam, frames[0], o), b); stats["snapped"] += 1; po["snapped"] += 1
                    stats["snapped_files"].append(f"{cam}/{t}/{o}")
                else:
                    os.link(a, b)
            stats["missing"] += len(set(objs) - set(names))
    return stats


def main():
    import cv2, numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--scenes", nargs="*", default=list(OBJ))
    a = ap.parse_args()
    out = a.method + SUFFIX
    try:
        head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = len(subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.splitlines())
    except OSError:
        head, dirty = None, None
    for scene in a.scenes:
        d = CFG[scene]
        if os.path.exists(os.path.join(DATA, d["folder"], out)):
            sys.exit(f"{scene}: {out} exists; remove it first")
    me = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    for scene in a.scenes:
        st = snap_scene(scene, a.method, out, np, cv2)
        man = dict(schema="MVSeed s2r6_snap", schema_version=1, prereg="docs/stage2-R6-prereg.md", source_method=a.method,
                   method=out, rule=dict(tau=TAU, delta=DELTA, dilate=DILATE), git_head=head, git_dirty=dirty,
                   script_sha256=me, written_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), stats=st)
        json.dump(man, open(os.path.join(DATA, CFG[scene]["folder"], out, "MANIFEST.json"), "w"), indent=1)
        print(f"{scene:20s} ref {st['ref']:12s} snapped {st['snapped']:4d} / {st['objects_frames']:4d}  no-prompt {st['no_prompt']}  missing {st['missing']}", flush=True)


if __name__ == "__main__":
    main()
