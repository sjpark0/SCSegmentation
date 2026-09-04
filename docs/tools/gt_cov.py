import json, os
import numpy as np, cv2
ROOT = '/home/sjpark/Documents/SCSegmentation/Data/MVSeg'
CFG = json.load(open('/home/sjpark/Documents/SCSegmentation/SCSam3/demo/MVSeg.json'))
DS = ['AlexaMeadeFacePaint', 'Barn', 'Blocks', 'Breakfast', 'Carpark', 'Dog', 'Fencing', 'Frog', 'MATF',
      'Painter', 'PoznanStreet', 'Welder', 'AlexaMeadeExhibit', 'CoffeeMartini', 'FlameSteak']
def cn(c, p, p1): return f"{p}{c:0{p1}d}"

print("=== pixel-level diff OneStage vs OneStageNew at first tracked frame, view-index-0 cameras")
for ds, cam in [('Fencing', 'v0'), ('Barn', 'v0'), ('Welder', 'camera_0001'), ('Carpark', 'v0')]:
    st = CFG[ds]['start_frame']
    for fr in (st + 1, st + 10, st + 20):
        d1 = os.path.join(ROOT, ds, 'SegMaskSam3OneStage', cam, str(fr))
        d2 = os.path.join(ROOT, ds, 'SegMaskSam3OneStageNew', cam, str(fr))
        tot = 0; ndiff = 0; nobj = 0; maxd = 0
        for f in sorted(os.listdir(d1)):
            a = cv2.imread(os.path.join(d1, f), 0) > 127
            b = cv2.imread(os.path.join(d2, f), 0) > 127
            nd = int((a ^ b).sum()); ndiff += nd; tot += int(a.sum()); nobj += 1; maxd = max(maxd, nd)
        print(f"  {ds}/{cam} frame {fr}: {nobj} objs, differing px total={ndiff} (max per obj {maxd}), fg px OneStage={tot}")

print("\n=== GT coverage: objects scored per camera vs objects present in the reference seed frame")
tot_pairs = 0; tot_struct0 = 0; tot_gtempty = 0
for ds in DS:
    d = CFG[ds]; cams = [cn(c, d['prefix'], d['prefix1']) for c in d['cam_list']]; st = d['start_frame']
    best, bn = None, -1
    for cam in cams:
        n = int(np.max(cv2.imread(os.path.join(ROOT, ds, 'Mask', cam, f"{st:06d}.png"), 0)))
        if n > bn: best, bn = cam, n
    ref_gt = cv2.imread(os.path.join(ROOT, ds, 'Mask', best, f"{st:06d}.png"), 0)
    ref_ids = set(int(x) for x in np.unique(ref_gt)) - {0}
    line = f"{ds:20s} ref={best} maxid={bn} ids_in_ref_seed={len(ref_ids)} ->"
    for cam in cams:
        files = sorted(f for f in os.listdir(os.path.join(ROOT, ds, 'Mask', cam)) if f.endswith('.png'))
        gts = [cv2.imread(os.path.join(ROOT, ds, 'Mask', cam, f), 0) for f in files]
        ids = sorted(set(int(x) for g in gts for x in np.unique(g)) - {0})
        n_pairs = len(ids) * len(gts)
        gt_nonempty = np.array([[bool((g == o).any()) for g in gts] for o in ids])
        n_gtempty = int((~gt_nonempty).sum())
        missing_in_ref = [o for o in ids if o not in ref_ids]
        n_struct0 = int(sum(gt_nonempty[i].sum() for i, o in enumerate(ids) if o not in ref_ids))
        tot_pairs += n_pairs; tot_struct0 += n_struct0; tot_gtempty += n_gtempty
        line += f" {cam}: {len(ids)} objs, {len(missing_in_ref)} not in ref seed ({n_struct0} nonempty pairs->J=0 structurally), {n_gtempty}/{n_pairs} (obj,frame) pairs GT-empty"
    print(line)
print(f"TOTAL (15 ds): {tot_pairs} (obj,frame) pairs; structurally-zero nonempty pairs={tot_struct0} ({100*tot_struct0/tot_pairs:.1f}%); GT-empty pairs={tot_gtempty} ({100*tot_gtempty/tot_pairs:.1f}%)")
