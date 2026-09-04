#!/usr/bin/env python3
"""Per-camera / per-frame breakdown of the MVSeg J&F results for the validity review.

Reuses eval_jf.py's own J/F functions so numbers are on the same footing.
"""
import json, os, sys, math
import numpy as np
import cv2
from multiprocessing import Pool

ROOT = '/home/sjpark/Documents/SCSegmentation/Data/MVSeg'
sys.path.insert(0, ROOT)
import eval_jf as E

CFG = json.load(open('/home/sjpark/Documents/SCSegmentation/SCSam3/demo/MVSeg.json'))
DS12 = ['AlexaMeadeFacePaint', 'Barn', 'Blocks', 'Breakfast', 'Carpark', 'Dog', 'Fencing',
        'Frog', 'MATF', 'Painter', 'PoznanStreet', 'Welder']
DS3 = ['AlexaMeadeExhibit', 'CoffeeMartini', 'FlameSteak']
METHODS = ['SegMaskNew1', 'SegMaskNew2', 'SegMaskNew3', 'SegMaskSam3OneStage', 'SegMaskSam3OneStageNew']
OUT = os.path.dirname(os.path.abspath(__file__))


def cam_name(c, p, p1):
    return f"{p}{c:0{p1}d}"


def ds_info(name):
    d = CFG[name]
    perms = d.get('perms') or list(range(d['start_cam'], d['num_cam'] + d['start_cam']))
    cams = [cam_name(c, d['prefix'], d['prefix1']) for c in d['cam_list']]
    # reference rule from runMVSeg.pick_reference: max id in the GT at start_frame
    best, bestn = None, -1
    for c in d['cam_list']:
        p = os.path.join(ROOT, d['folder'], 'Mask', cam_name(c, d['prefix'], d['prefix1']),
                         f"{d['start_frame']:06d}.png")
        n = int(np.max(cv2.imread(p, cv2.IMREAD_GRAYSCALE)))
        if n > bestn:
            best, bestn = c, n
    ref = cam_name(best, d['prefix'], d['prefix1'])
    view_index = {cam_name(c, d['prefix'], d['prefix1']): perms.index(c) for c in d['cam_list']}
    return dict(cams=cams, ref=ref, ref_maxid=bestn, view_index=view_index,
                n_views=len(perms), start=d['start_frame'], perms=perms, ref_view=perms.index(best))


def per_frame(job):
    """J and F per object per frame for one (dataset, cam, method); same code path as eval_jf."""
    ds, cam, method = job
    ds_dir = os.path.join(ROOT, ds)
    gt_dir = os.path.join(ds_dir, 'Mask', cam)
    gt_files = sorted((f for f in os.listdir(gt_dir) if f.endswith('.png')), key=E.frame_key)
    gts = []
    ids = set()
    for f in gt_files:
        g = cv2.imread(os.path.join(gt_dir, f), cv2.IMREAD_UNCHANGED)
        if g.ndim == 3:
            g = g[..., 0]
        gts.append(g)
        ids.update(np.unique(g).tolist())
    ids.discard(0)
    ids = sorted(int(o) for o in ids)
    h, w = gts[0].shape[:2]
    bound_pix = int(math.ceil(E.BOUND_TH * np.linalg.norm((h, w))))
    se = E._disk(bound_pix)
    m_cam = os.path.join(ds_dir, method, cam)
    if not os.path.isdir(m_cam):
        return dict(job=job, result=None)
    pred_frames = sorted(os.listdir(m_cam), key=E.frame_key)
    by_id = {E.frame_key(p): p for p in pred_frames}
    pred_frames = [by_id.get(E.frame_key(f)) for f in gt_files]
    n_o, n_f = len(ids), len(gts)
    J = np.zeros((n_o, n_f)); F = np.zeros((n_o, n_f))
    gt_empty = np.zeros((n_o, n_f), bool); pred_empty = np.zeros((n_o, n_f), bool)
    pred_missing = np.zeros((n_o, n_f), bool)
    gt_area = np.zeros((n_o, n_f))
    for fi, gt in enumerate(gts):
        pf = pred_frames[fi]
        for oi, obj in enumerate(ids):
            gm = gt == obj
            gt_area[oi, fi] = gm.sum()
            pm = None
            if pf is not None:
                p = os.path.join(m_cam, pf, f'{obj}.png')
                if os.path.exists(p):
                    pm = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if pm is None:
                pred_missing[oi, fi] = True
                pm = np.zeros((h, w), bool)
            else:
                if pm.shape[:2] != (h, w):
                    pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_NEAREST)
                pm = pm > 127
            gt_empty[oi, fi] = not gm.any(); pred_empty[oi, fi] = not pm.any()
            box = E.crop_box(gm, pm, bound_pix + 2, (h, w))
            if box is None:
                J[oi, fi] = 1.0; F[oi, fi] = 1.0; continue
            y0, y1, x0, x1 = box
            J[oi, fi] = E.db_eval_iou(gm[y0:y1, x0:x1], pm[y0:y1, x0:x1])
            F[oi, fi] = E.db_eval_boundary(pm[y0:y1, x0:x1], gm[y0:y1, x0:x1], bound_pix, se)
    return dict(job=job, result=dict(ids=ids, J=J.tolist(), F=F.tolist(), gt_empty=gt_empty.tolist(),
                                     pred_empty=pred_empty.tolist(), pred_missing=pred_missing.tolist(),
                                     gt_area=gt_area.tolist(), h=h, w=w))


def main():
    info = {d: ds_info(d) for d in DS12 + DS3}
    json.dump(info, open(os.path.join(OUT, 'ds_info.json'), 'w'), indent=1)
    jobs = [(d, c, m) for d in DS12 + DS3 for c in info[d]['cams'] for m in METHODS
            if os.path.isdir(os.path.join(ROOT, d, m, c))]
    print(len(jobs), 'jobs', flush=True)
    with Pool(os.cpu_count()) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(per_frame, jobs), 1):
            res.append(r)
            print(f'[{i}/{len(jobs)}] {r["job"]}', flush=True)
    json.dump(res, open(os.path.join(OUT, 'per_frame.json'), 'w'))
    print('done', flush=True)


if __name__ == '__main__':
    main()
