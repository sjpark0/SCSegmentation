import json, os, statistics as st
import numpy as np
A = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(os.path.join(A, 'per_frame.json')))
info = json.load(open(os.path.join(A, 'ds_info.json')))
R = {tuple(r['job']): r['result'] for r in res if r['result'] is not None}
DS12 = ['AlexaMeadeFacePaint', 'Barn', 'Blocks', 'Breakfast', 'Carpark', 'Dog', 'Fencing', 'Frog', 'MATF', 'Painter', 'PoznanStreet', 'Welder']
M = ['SegMaskNew1', 'SegMaskSam3OneStage', 'SegMaskSam3OneStageNew']

def cam_stats(r):
    J = np.array(r['J']); F = np.array(r['F']); ge = np.array(r['gt_empty']); pe = np.array(r['pred_empty']); pm = np.array(r['pred_missing'])
    return dict(J0=J[:, 0].mean(), F0=F[:, 0].mean(), Jall=J.mean(), Fall=F.mean(), Jrest=J[:, 1:].mean(), Frest=F[:, 1:].mean(),
                n_pairs=J.size, gt_empty=int(ge.sum()), both_empty=int((ge & pe).sum()), gt_nonempty_pred_empty=int((~ge & pe).sum()),
                J_nonempty=J[~ge].mean() if (~ge).any() else float('nan'), F_nonempty=F[~ge].mean() if (~ge).any() else float('nan'),
                J=J, F=F, ge=ge, pe=pe)

print("=== seed frame (first scored frame) J/F vs remaining 20 frames, per scored camera (ref marked *)")
agg = {m: dict(J0=[], Jrest=[], F0=[], Frest=[], Jall=[], Fall=[], ref_J0=[], ref_Jall=[], ref_Jrest=[], nonref_Jall=[], both_empty=0, pairs=0, gt_empty=0, JF_nonempty=[], JF_all=[]) for m in M}
for ds in DS12:
    for cam in info[ds]['cams']:
        isref = cam == info[ds]['ref']
        line = f"{ds:20s} {cam:12s} {'*' if isref else ' '}"
        for m in M:
            r = R.get((ds, cam, m))
            if r is None: line += f"  {m[-6:]}: -"; continue
            s = cam_stats(r)
            line += f"  {m[-6:]}: J0={s['J0']:.3f} F0={s['F0']:.3f} | J1-20={s['Jrest']:.3f} F1-20={s['Frest']:.3f} | gtEmpty {s['gt_empty']}/{s['n_pairs']} bothEmpty {s['both_empty']} | J(nonempty gt)={s['J_nonempty']:.3f}"
            a = agg[m]; a['J0'].append(s['J0']); a['Jrest'].append(s['Jrest']); a['F0'].append(s['F0']); a['Frest'].append(s['Frest']); a['Jall'].append(s['Jall']); a['Fall'].append(s['Fall'])
            a['both_empty'] += s['both_empty']; a['pairs'] += s['n_pairs']; a['gt_empty'] += s['gt_empty']
            a['JF_nonempty'].append((s['J_nonempty'] + s['F_nonempty']) / 2); a['JF_all'].append((s['Jall'] + s['Fall']) / 2)
            if isref: a['ref_J0'].append(s['J0']); a['ref_Jall'].append(s['Jall']); a['ref_Jrest'].append(s['Jrest'])
            else: a['nonref_Jall'].append(s['Jall'])
        print(line)
print()
for m in M:
    a = agg[m]
    print(f"{m:24s} mean over 36 cams: J0={st.mean(a['J0']):.4f} J1-20={st.mean(a['Jrest']):.4f} F0={st.mean(a['F0']):.4f} F1-20={st.mean(a['Frest']):.4f} | ref cams: J0={st.mean(a['ref_J0']):.4f} Jall={st.mean(a['ref_Jall']):.4f} J1-20={st.mean(a['ref_Jrest']):.4f} | non-ref Jall={st.mean(a['nonref_Jall']):.4f} | (obj,frame) pairs={a['pairs']} GT-empty={a['gt_empty']} both-empty(credited 1.0)={a['both_empty']} | J&F cam-mean all={st.mean(a['JF_all']):.4f} nonempty-GT-only={st.mean(a['JF_nonempty']):.4f}")

print("\n=== per-frame J delta OneStageNew-OneStage on view-index-0 cameras (no cross-view memory), frames 0..20")
for ds in DS12:
    for cam in info[ds]['cams']:
        if info[ds]['view_index'][cam] != 0: continue
        a = R[(ds, cam, 'SegMaskSam3OneStage')]; b = R[(ds, cam, 'SegMaskSam3OneStageNew')]
        d = (np.array(b['J']) - np.array(a['J'])).mean(0)
        print(f"  {ds}/{cam}: " + ' '.join(f"{x:+.3f}" for x in d) + f"  | max|dJ| per (obj,frame)={np.abs(np.array(b['J']) - np.array(a['J'])).max():.3f}")

print("\n=== where does the OneStageNew gain come from: per-object J delta on Fencing v9, PoznanStreet v8/v4, Welder camera_0003")
for ds, cam in [('Fencing', 'v9'), ('PoznanStreet', 'v8'), ('PoznanStreet', 'v4'), ('Welder', 'camera_0003'), ('Blocks', 'cam9')]:
    a = R[(ds, cam, 'SegMaskSam3OneStage')]; b = R[(ds, cam, 'SegMaskSam3OneStageNew')]
    Ja = np.array(a['J']); Jb = np.array(b['J']); Fa = np.array(a['F']); Fb = np.array(b['F'])
    dJ = Jb.mean(1) - Ja.mean(1); dF = Fb.mean(1) - Fa.mean(1)
    order = np.argsort(-np.abs(dJ))
    print(f"  {ds}/{cam}: objs={len(a['ids'])} camera dJ={dJ.mean():+.4f} dF={dF.mean():+.4f}; top objects: " + ', '.join(f"id{a['ids'][i]}(dJ{dJ[i]:+.3f},dF{dF[i]:+.3f},area{np.mean(a['gt_area'][i]):.0f})" for i in order[:5]))
    dJf = (Jb - Ja).mean(0)
    print("     per-frame dJ: " + ' '.join(f"{x:+.2f}" for x in dJf))

print("\n=== SAM2 vs SAM3 on Welder / Fencing: per-camera J on non-empty GT frames only, and seed-frame J")
for ds in ['Welder', 'Fencing']:
    for cam in info[ds]['cams']:
        line = f"  {ds}/{cam} idx={info[ds]['view_index'][cam]} {'*' if cam == info[ds]['ref'] else ' '}"
        for m in M:
            s = cam_stats(R[(ds, cam, m)]); line += f" | {m[-6:]}: J0={s['J0']:.3f} J1-20={s['Jrest']:.3f} F1-20={s['Frest']:.3f}"
        print(line)

print("\n=== dataset-level J&F recomputed three ways (12 ds): all frames / frames 1..20 only / non-ref cameras only")
for m in M:
    allv = []; restv = []; nonref = []
    for ds in DS12:
        jj = []; ff = []; jr = []; fr = []; jn = []; fn = []
        for cam in info[ds]['cams']:
            s = cam_stats(R[(ds, cam, m)]); jj += list(s['J'].mean(1)); ff += list(s['F'].mean(1)); jr += list(s['J'][:, 1:].mean(1)); fr += list(s['F'][:, 1:].mean(1))
            if cam != info[ds]['ref']: jn += list(s['J'].mean(1)); fn += list(s['F'].mean(1))
        allv.append((st.mean(jj) + st.mean(ff)) / 2); restv.append((st.mean(jr) + st.mean(fr)) / 2); nonref.append((st.mean(jn) + st.mean(fn)) / 2)
    print(f"  {m:24s} all={st.mean(allv):.4f}  frames1-20={st.mean(restv):.4f}  nonref-cams={st.mean(nonref):.4f}")
