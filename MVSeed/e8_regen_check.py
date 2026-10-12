#!/usr/bin/env python3
"""S1-E8 watch value Z0 (docs/stage1-E8-prereg.md section 2.1, 7): does the regeneration at the
2D repo's HEAD reproduce the SCSegmentation2D results the user was shown?

Per arm, two levels:
  bytes   every <cam>/<obj>.png of Result/<old> vs Result/<new>: identical / different /
          only-old / only-new, per scene; for different files the pixel IoU of old vs new
  scores  the 2D repo's own evaluator (mvseg2d.evaluate, its .venv) on the new folder, written to
          --score-dir (never into the 2D repo), against its stored scores/<old>.json: scene-mean
          J, F, J&F and the per-pair J difference

    python3 MVSeed/e8_regen_check.py [--arms geoset_rep,geoset_abs,geoalign_rep]
"""
import argparse, hashlib, json, os, statistics, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(os.path.dirname(REPO), "SCSegmentation2D")
ARMS = {"geoset_rep": ("geoset_pred_ft_rep", "geoset_pred_ft_rep_e8"),
        "geoset_abs": ("geoset_pred_ft_abs", "geoset_pred_ft_abs_e8"),
        "geoalign_rep": ("geoalign_ft_x_rep", "geoalign_ft_x_rep_e8")}


def digest(p):
    return hashlib.sha256(open(p, "rb").read()).digest()


def pngs(d):
    out = {}
    for cam in sorted(os.listdir(d)):
        cd = os.path.join(d, cam)
        if os.path.isdir(cd):
            for f in os.listdir(cd):
                if f.endswith(".png"): out[(cam, f)] = os.path.join(cd, f)
    return out


def pix_iou(a, b):
    from PIL import Image
    A, B = Image.open(a).convert("L"), Image.open(b).convert("L")
    ha = A.point(lambda v: 255 if v > 127 else 0); hb = B.point(lambda v: 255 if v > 127 else 0)
    from PIL import ImageChops
    inter = ImageChops.multiply(ha, hb).histogram()[255]
    union = ImageChops.lighter(ha, hb).histogram()[255]
    return 1.0 if union == 0 else inter / union


def byte_check(old, new):
    scenes = sorted(os.listdir(os.path.join(SRC, "Data")))
    per, tot = {}, dict(identical=0, different=0, only_old=0, only_new=0)
    ious = []
    for sc in scenes:
        do, dn = (os.path.join(SRC, "Data", sc, "Result", x) for x in (old, new))
        if not os.path.isdir(dn):
            per[sc] = "new folder missing"; continue
        po, pn = pngs(do), pngs(dn)
        r = dict(identical=0, different=0, only_old=len(set(po) - set(pn)), only_new=len(set(pn) - set(po)))
        diffs = []
        for k in sorted(set(po) & set(pn)):
            if digest(po[k]) == digest(pn[k]): r["identical"] += 1
            else:
                r["different"] += 1; v = pix_iou(po[k], pn[k]); diffs.append(v); ious.append(v)
        if diffs: r["diff_iou_mean"], r["diff_iou_min"] = statistics.mean(diffs), min(diffs)
        run = json.load(open(os.path.join(dn, "RUN.json")))
        r["run_git"], r["status"] = run.get("git"), run.get("status")
        per[sc] = r
        for k in tot: tot[k] += r[k]
    tot["diff_iou_mean"] = statistics.mean(ious) if ious else None
    tot["diff_iou_min"] = min(ious) if ious else None
    return tot, per


def summarize(score_file):
    d = json.load(open(score_file))
    sj, sf, sjf, pairs, status = [], [], [], {}, {}
    for sc, v in sorted(d["scenes"].items()):
        status[sc] = v["status"]
        rows = v.get("rows") or []
        if not rows: continue
        J = statistics.mean(r["J"] for r in rows); F = statistics.mean(r["F"] for r in rows)
        sj.append(J); sf.append(F); sjf.append((J + F) / 2)
        for r in rows: pairs[(sc, r["cam"], r["obj"])] = r["J"]
    return dict(J=statistics.mean(sj), F=statistics.mean(sf), JF=statistics.mean(sjf), n_scenes=len(sj),
                n_pairs=len(pairs), status=status), pairs


def score_check(old, new, score_dir):
    os.makedirs(score_dir, exist_ok=True)
    out = os.path.join(score_dir, f"{new}.json")
    cmd = [os.path.join(SRC, ".venv", "bin", "python"), "-m", "mvseg2d.evaluate", "--data", "Data", "--method", new, "--out", out]
    subprocess.run(cmd, cwd=SRC, check=True, stdout=subprocess.DEVNULL)
    so, po = summarize(os.path.join(SRC, "scores", f"{old}.json"))
    sn, pn = summarize(out)
    common = sorted(set(po) & set(pn))
    dj = [pn[k] - po[k] for k in common]
    return dict(old=so, new=sn, pairs_common=len(common), pairs_equal=sum(1 for x in dj if abs(x) < 1e-12),
                dJ_mean=statistics.mean(dj) if dj else None, dJ_max_abs=max((abs(x) for x in dj), default=None),
                only_old=len(set(po) - set(pn)), only_new=len(set(pn) - set(po)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--score-dir", default=os.path.join(REPO, "MVSeed", "runs", "e8_regen_scores"))
    a = ap.parse_args()
    path = os.path.join(REPO, "MVSeed", "runs", "e8_regen_check.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for arm in a.arms.split(","):
        old, new = ARMS[arm]
        tot, per = byte_check(old, new)
        sc = score_check(old, new, a.score_dir)
        res[arm] = dict(old=old, new=new, bytes=tot, bytes_per_scene=per, scores=sc)
        print(f"{arm}: {old} -> {new}")
        print(f"   bytes  identical {tot['identical']}  different {tot['different']}  only-old {tot['only_old']}  only-new {tot['only_new']}"
              + (f"  (different: pixel IoU mean {tot['diff_iou_mean']:.4f}, min {tot['diff_iou_min']:.4f})" if tot["diff_iou_mean"] is not None else ""))
        o, n = sc["old"], sc["new"]
        print(f"   scores old J {o['J']:.4f} F {o['F']:.4f} J&F {o['JF']:.4f}  |  new J {n['J']:.4f} F {n['F']:.4f} J&F {n['JF']:.4f}"
              f"  pairs equal {sc['pairs_equal']}/{sc['pairs_common']}  max|ΔJ| {sc['dJ_max_abs']}")
    json.dump(res, open(path, "w"), indent=1)
    print(f"wrote {os.path.relpath(path, REPO)}")


if __name__ == "__main__":
    sys.exit(main())
