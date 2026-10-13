#!/usr/bin/env python3
"""Stage-1 port check (docs/stage1-port.md): do the stage-1 outputs regenerated from stage1/ (the
code ported from SCSegmentation2D) equal the 2D repo's E8 outputs byte for byte?

Per folder (the three E8 outputs and their round-1 folders): every <cam>/<obj>.png and
round1_conf.json of SCSegmentation2D/Data/<scene>/Result/<X> vs stage1/Data/<scene>/Result/<X>:
identical / different / only-old / only-new per scene, pixel IoU of different PNGs.  RUN.json is
compared on its non-volatile fields only (args, arm, fold, checkpoint sha).  Host python (PIL).

    python3 MVSeed/stage1_port_check.py      ->  MVSeed/runs/stage1_port_check.json
"""
import json, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e8_regen_check import digest, pngs, pix_iou

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = os.path.join(os.path.dirname(REPO), "SCSegmentation2D", "Data")
NEW = os.path.join(REPO, "stage1", "Data")
FOLDERS = ["geoset_pred_ft_rep_e8", "geoset_pred_ft_r1_e8", "geoset_pred_ft_abs_e8", "geoset_pred_ft_r1_e8b",
           "geoalign_ft_x_rep_e8", "geoalign_ft_x_r1_e8"]
VOLATILE = {"date", "seconds", "gpu_mb", "git", "started", "finished", "host", "elapsed", "time", "gpu_peak_mb"}


def stable(run):
    return {k: v for k, v in run.items() if k not in VOLATILE} if isinstance(run, dict) else run


def check(folder):
    per, tot, ious = {}, dict(identical=0, different=0, only_old=0, only_new=0, conf_same=0, conf_diff=0, run_same=0, run_diff=0), []
    for sc in sorted(os.listdir(NEW)):
        do, dn = os.path.join(OLD, sc, "Result", folder), os.path.join(NEW, sc, "Result", folder)
        if not os.path.isdir(dn) or not os.path.isdir(do):
            per[sc] = dict(missing="new" if not os.path.isdir(dn) else "old"); tot["only_old" if os.path.isdir(do) else "only_new"] += 1; continue
        po, pn = pngs(do), pngs(dn)
        r = dict(identical=0, different=0, only_old=len(set(po) - set(pn)), only_new=len(set(pn) - set(po)))
        for k in sorted(set(po) & set(pn)):
            if digest(po[k]) == digest(pn[k]): r["identical"] += 1
            else: r["different"] += 1; ious.append(pix_iou(po[k], pn[k]))
        co, cn = os.path.join(do, "round1_conf.json"), os.path.join(dn, "round1_conf.json")
        if os.path.exists(co) or os.path.exists(cn):
            same = os.path.exists(co) and os.path.exists(cn) and digest(co) == digest(cn)
            tot["conf_same" if same else "conf_diff"] += 1; r["round1_conf_same"] = same
        ro, rn = (json.load(open(os.path.join(x, "RUN.json"))) for x in (do, dn))
        so, sn = stable(ro), stable(rn)
        r["run_stable_same"] = so == sn
        if so != sn: r["run_diff_keys"] = sorted(k for k in set(so) | set(sn) if so.get(k) != sn.get(k))
        tot["run_same" if so == sn else "run_diff"] += 1
        r["new_git"] = rn.get("git")
        per[sc] = r
        for k in ("identical", "different", "only_old", "only_new"): tot[k] += r[k]
    tot["diff_iou_min"] = min(ious) if ious else None
    tot["ok"] = tot["different"] == 0 and tot["only_old"] == 0 and tot["only_new"] == 0 and tot["conf_diff"] == 0
    return tot, per


def main():
    out = {}
    for f in FOLDERS:
        tot, per = check(f)
        out[f] = dict(total=tot, per_scene=per)
        print(f"{f:24s} identical {tot['identical']:5d}  different {tot['different']}  only-old {tot['only_old']}  only-new {tot['only_new']}"
              f"  round1_conf same/diff {tot['conf_same']}/{tot['conf_diff']}  RUN.json stable same/diff {tot['run_same']}/{tot['run_diff']}  -> {'OK' if tot['ok'] else 'FAIL'}")
    diffkeys = sorted({k for f in out.values() for r in f["per_scene"].values() for k in r.get("run_diff_keys", [])})
    if diffkeys: print("RUN.json fields that differ (beyond the volatile set):", diffkeys)
    out["all_ok"] = all(v["total"]["ok"] for k, v in out.items() if k in FOLDERS)
    json.dump(out, open(os.path.join(REPO, "MVSeed", "runs", "stage1_port_check.json"), "w"), indent=1)
    print("all identical:", out["all_ok"])


if __name__ == "__main__":
    main()
