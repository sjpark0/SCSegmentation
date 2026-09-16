#!/usr/bin/env python3
"""S3-P readings Q1-Q6 (docs/plenoptic-P-prereg.md section 2) from the frozen scorer's output and the mask folders.

    # in the container (cv2):
    python eval/eval_jf.py --root PlenopticServer1/CROOM/SegResult --config plenoptic/Plenoptic.json \
        --methods SegMaskSam3XW0MFs SegMaskSam3XW1CGPS4MFsCs10kG2 SegMaskSam3XW0MFs_txt SegMaskSam3XW0MFs_pt \
        --out /host$PWD/plenoptic/jf_plenoptic.json T06_flag_ganggang
    # host:
    python3 plenoptic/score.py            ->  plenoptic/score.json (+ prints the tables)

Q1 mean J over the GT-present nonreference pairs at t=0; Q2 by ring distance; Q3 failures / empties / hallucinations;
Q4 prompt-seed loss against the GT-seeded run; Q5 MUVOD-style scene J&F at t=0 (objects -> cameras -> scene);
Q6 divergence events between control and headline from mask areas alone (no GT after t=0).
"""
import json, os, statistics
import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARE = os.path.join(REPO, "PlenopticServer1", "CROOM", "SegResult", "T06_flag_ganggang")
JF = os.path.join(REPO, "plenoptic", "jf_plenoptic.json")
CTRL, HEAD, TXT, PT = "SegMaskSam3XW0MFs", "SegMaskSam3XW1CGPS4MFsCs10kG2", "SegMaskSam3XW0MFs_txt", "SegMaskSam3XW0MFs_pt"
REF, NCAM, NOBJ, NFRAME = 6, 32, 6, 101
VANISH, HOLD, MIN_RUN, MIN_HOLD = 0.1, 0.5, 3, 2          # Q6 rule, registered


def ring_distance(k): d = abs(k - REF); return min(d, NCAM - d)


def cam_index(name): return int(name.split("_")[1])


def t0_rows(entries, method):
    """{(cam, obj): (J, F, gt_area)} at t=0 for one method; None where the scorer has no value."""
    out = {}
    for e in entries:
        if e["method"] != method or not e.get("result"): continue
        pf = e["result"]["per_frame"]; k = cam_index(e["camera"])
        for i, o in enumerate(e["objects"]):
            out[(k, o)] = (pf["J"][i][0], pf["F"][i][0], pf["gt_area"][i][0])
    return out


def stage1(rows):
    """Q1/Q2/Q3/Q5 for one method's t=0 rows."""
    nonref = {(k, o): v for (k, o), v in rows.items() if k != REF}
    present = {ko: v for ko, v in nonref.items() if v[2] and v[0] is not None}
    empty_gt = {ko: v for ko, v in nonref.items() if not v[2]}
    J = [v[0] for v in present.values()]
    byd = {}
    for (k, o), v in present.items(): byd.setdefault(ring_distance(k), []).append(v[0])
    # Q5: MUVOD-style, all 32 cameras incl. reference: mean over objects -> cameras -> scene, J&F at t=0
    percam = {}
    for (k, o), (j, f, g) in rows.items():
        if j is None or f is None: continue
        percam.setdefault(k, []).append(0.5 * (j + f))
    q5 = 100 * statistics.mean(statistics.mean(v) for v in percam.values())
    return dict(n_present=len(present), n_gt_empty=len(empty_gt), Q1_mean_J=statistics.mean(J), Q1_median_J=statistics.median(J),
                Q2_by_distance={d: dict(n=len(v), mean_J=statistics.mean(v), min_J=min(v)) for d, v in sorted(byd.items())},
                Q3_fail_lt05=sum(j < 0.5 for j in J), Q3_fail_rate=sum(j < 0.5 for j in J) / len(J),
                Q3_worst=sorted(((round(v[0], 3), f"cam_{k:02d}", o) for (k, o), v in present.items()))[:8],
                Q5_scene_JF_t0=q5, ref_J={o: rows[(REF, o)][0] for o in range(1, NOBJ + 1) if (REF, o) in rows})


def areas(method):
    """area[(cam, obj)][t] in pixels from the written PNGs (missing file -> 0)."""
    A = {}
    for k in range(NCAM):
        cam = os.path.join(SHARE, method, f"cam_{k:02d}")
        for o in range(1, NOBJ + 1):
            A[(k, o)] = np.array([int((np.array(Image.open(os.path.join(cam, str(t), f"{o}.png"))) > 0).sum())
                                  if os.path.exists(os.path.join(cam, str(t), f"{o}.png")) else 0 for t in range(NFRAME)])
    return A


def events(A_x, A_y, label):
    """Intervals where X's area stays < VANISH*A0 for >= MIN_RUN frames while Y keeps >= HOLD*A0 on >= MIN_HOLD of them."""
    ev = []
    for ko in A_x:
        a0 = A_x[ko][0]
        if a0 == 0 or A_y[ko][0] != a0: continue                      # Z1: identical seeds; skip pairs with no seed
        lost = A_x[ko] < VANISH * a0; t = 1
        while t < NFRAME:
            if not lost[t]: t += 1; continue
            s = t
            while t < NFRAME and lost[t]: t += 1
            if t - s >= MIN_RUN:
                held = int((A_y[ko][s:t] >= HOLD * a0).sum())
                if held >= MIN_HOLD:
                    ev.append(dict(direction=label, cam=f"cam_{ko[0]:02d}", obj=ko[1], frames=[s, t - 1], n=t - s, held_frames=held,
                                   ring_distance=ring_distance(ko[0])))
    return ev


def main():
    entries = json.load(open(JF)); out = {}
    R = {m: t0_rows(entries, m) for m in (CTRL, HEAD, TXT, PT)}
    for m in (CTRL, HEAD, TXT, PT):
        if R[m]: out[m] = stage1(R[m])
    if R[CTRL] and R[HEAD]:      # Z1: identical Stage-1 seeds -> identical t=0 scores
        diff = [ko for ko in R[CTRL] if ko in R[HEAD] and R[CTRL][ko][0] != R[HEAD][ko][0]]
        out["Z1_t0_score_mismatch"] = [f"cam_{k:02d}/{o}" for k, o in diff]
    for tag, m in (("text", TXT), ("point", PT)):
        if R[m] and R[CTRL]:
            out[f"Q4_{tag}_minus_gt"] = out[m]["Q1_mean_J"] - out[CTRL]["Q1_mean_J"]
    if os.path.isdir(os.path.join(SHARE, CTRL)) and os.path.isdir(os.path.join(SHARE, HEAD)):
        Ac, Ah = areas(CTRL), areas(HEAD)
        ev = events(Ac, Ah, "control_lost_headline_held") + events(Ah, Ac, "headline_lost_control_held")
        out["Q6_events"] = ev
        out["Q6_counts"] = {d: sum(e["direction"] == d for e in ev) for d in ("control_lost_headline_held", "headline_lost_control_held")}
        out["Q6_pairs_with_seed"] = sum(1 for ko in Ac if Ac[ko][0] > 0)
        # descriptive: how many pairs end (t=100) below 10% of their seed area in each configuration
        out["Q6_end_vanished"] = {"control": sum(1 for ko in Ac if Ac[ko][0] and Ac[ko][-1] < VANISH * Ac[ko][0]),
                                  "headline": sum(1 for ko in Ah if Ah[ko][0] and Ah[ko][-1] < VANISH * Ah[ko][0])}
    json.dump(out, open(os.path.join(REPO, "plenoptic", "score.json"), "w"), indent=1, default=float)
    for m in (CTRL, HEAD, TXT, PT):
        if m in out:
            s = out[m]; print(f"{m}: Q1 {s['Q1_mean_J']:.3f} (median {s['Q1_median_J']:.3f}, n={s['n_present']})  Q3 fail {s['Q3_fail_lt05']} ({100*s['Q3_fail_rate']:.1f}%)  Q5 {s['Q5_scene_JF_t0']:.2f}")
            print("   Q2 by d:", " ".join(f"{d}:{v['mean_J']:.2f}" for d, v in s["Q2_by_distance"].items()))
    for k in ("Q4_text_minus_gt", "Q4_point_minus_gt", "Z1_t0_score_mismatch", "Q6_counts", "Q6_end_vanished"):
        if k in out: print(k, out[k])
    for e in out.get("Q6_events", []): print("  event", e)


if __name__ == "__main__":
    main()
