#!/usr/bin/env python3
"""Build an S1-E8 seed set MVSeed_<tag> (docs/stage1-E8-prereg.md section 2).

Source: the GeoSet-2 stage 1 of the sibling repo SCSegmentation2D (branch geomem), output
<SRC>/Data/<scene>/Result/<out>/<cam>/<obj>.png -- one 0/255 PNG per object, original
resolution, every camera; no PNG where the prediction is empty.

For every camera of the scene (Video/ folder) x every basic object (seed_ids_c_ini):
  PNG present  -> format check (mode L, scene resolution, values in {0, 255}), bytes copied
  PNG absent   -> empty PNG (area 0 = "not in this view" by the new stage 1; never G0,
                  which would mix the treatment with the control seed)
Layout <cam>/<start_frame>/<obj>.png plus SEED_MANIFEST.json (written_views = all cameras)
and SEED_BUILD.json (per-scene counts, source RUN.json digest).  Host python (PIL only):

    python3 MVSeed/build_geoset_seeds.py --tag geoset_rep --out geoset_pred_ft_rep_e8
    python3 MVSeed/build_geoset_seeds.py --tag geoset_abs --out geoset_pred_ft_abs_e8
    python3 MVSeed/build_geoset_seeds.py --tag geoalign_rep --out geoalign_ft_x_rep_e8
"""
import argparse, datetime, hashlib, json, os, shutil, subprocess, sys
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "Data", "MVSeg")
CFG = json.load(open(os.path.join(REPO, "SCSam3", "demo", "MVSeg.json"), encoding="utf-8"))
CENSUS = json.load(open(os.path.join(REPO, "docs", "raw", "muvod_object_sets.json")))
SRC_DEFAULT = os.path.join(os.path.dirname(REPO), "SCSegmentation2D")
TAGS = ("geoset_rep", "geoset_abs", "geoalign_rep")    # the three registered arms


def sha256_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def src_git(src):
    try:
        head = subprocess.check_output(["git", "-C", src, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", src, "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        return head, bool(dirty)
    except Exception as exc:                      # recorded, not fatal: the RUN.json digest still pins the source
        return f"unknown ({exc})", None


def check_png(path, size):
    im = Image.open(path)
    if im.mode != "L":
        raise ValueError(f"{path}: mode {im.mode}, expected L")
    if im.size != size:
        raise ValueError(f"{path}: size {im.size}, expected {size}")
    hist = im.histogram()
    bad = [v for v in range(256) if hist[v] and v not in (0, 255)]
    if bad:
        raise ValueError(f"{path}: values {bad[:5]} outside {{0, 255}}")
    return hist[255]


def build(scene, src, out, tag, log):
    c = CFG[scene]; folder, start = c["folder"], c["start_frame"]
    cams = sorted(os.listdir(os.path.join(DATA, folder, "Video")))
    basic = sorted(set(int(v) for v in CENSUS[scene]["seed_ids_c_ini"]))
    res = os.path.join(src, "Data", scene, "Result", out)
    run = json.load(open(os.path.join(res, "RUN.json")))
    if run.get("status") != "done" or run.get("error"):
        raise RuntimeError(f"{scene}: source RUN.json status {run.get('status')} error {run.get('error')}")
    meta = json.load(open(os.path.join(src, "Data", scene, "meta.json")))
    if meta["ref_frame"] != start or meta["ref_cam"] != CENSUS[scene]["c_ini"]:
        raise RuntimeError(f"{scene}: source keyframe/c_ini {meta['ref_frame']}/{meta['ref_cam']} "
                           f"!= start_frame/c_ini {start}/{CENSUS[scene]['c_ini']}")
    src_cams = sorted(d for d in os.listdir(res) if os.path.isdir(os.path.join(res, d)))
    if src_cams != cams:
        raise RuntimeError(f"{scene}: source cameras {src_cams} != Video/ cameras {cams}")
    extra = [(cm, p) for cm in src_cams for p in os.listdir(os.path.join(res, cm))
             if not p.endswith(".png") or int(p[:-4]) not in basic]
    if extra:
        raise RuntimeError(f"{scene}: source PNGs outside the basic set: {extra[:5]}")
    size = Image.open(os.path.join(DATA, folder, "Video", cams[0], sorted(os.listdir(os.path.join(DATA, folder, "Video", cams[0])))[0])).size

    dst = os.path.join(DATA, folder, f"MVSeed_{tag}")
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    n_copy = n_empty = n_copy_zero = 0
    for cam in cams:
        d = os.path.join(dst, cam, str(start)); os.makedirs(d)
        for o in basic:
            s = os.path.join(res, cam, f"{o}.png"); t = os.path.join(d, f"{o}.png")
            if os.path.exists(s):
                n_copy_zero += check_png(s, size) == 0
                shutil.copyfile(s, t); n_copy += 1
            else:
                Image.new("L", size, 0).save(t); n_empty += 1
    head, dirty = src_git(src)
    man = dict(schema="MVSeed build_geoset_seeds", schema_version=1, set=tag, prereg="docs/stage1-E8-prereg.md",
               dataset=scene, start_frame=start, c_ini=CENSUS[scene]["c_ini"], written_views=cams, objects=basic,
               source=dict(repo=src, git_head=head, git_dirty=dirty, result=out, run_json_sha256=sha256_file(os.path.join(res, "RUN.json")),
                           run_git=run.get("git"), run_date=run.get("date"), fold=run["config"].get("fold"),
                           ckpt_sha256=run["config"].get("ckpt_sha256"), arm=run["config"].get("arm"), gate=run["config"].get("gate")),
               n_png=n_copy + n_empty, n_copied=n_copy, n_empty=n_empty, n_copied_zero_area=n_copy_zero,
               generated_at=datetime.datetime.now().isoformat(timespec="seconds"))
    json.dump(man, open(os.path.join(dst, "SEED_MANIFEST.json"), "w"), indent=1)
    log(f"  {scene:20s} {len(cams):2d} cams x {len(basic):2d} objects = {n_copy + n_empty:4d} PNG "
        f"({n_copy} copied, {n_empty} empty, {n_copy_zero} copied-but-zero)")
    return dict(scene=scene, cams=len(cams), objects=len(basic), copied=n_copy, empty=n_empty, copied_zero=n_copy_zero,
                run_git=run.get("git"), fold=run["config"].get("fold"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=SRC_DEFAULT, help="SCSegmentation2D checkout")
    ap.add_argument("--tag", required=True, choices=TAGS, help="seed folder MVSeed_<tag>")
    ap.add_argument("--out", required=True, help="Result/<out> folder name in the source repo")
    ap.add_argument("--scenes", default="")
    a = ap.parse_args()
    scenes = a.scenes.split(",") if a.scenes else sorted(CENSUS)
    log = lambda m: print(m, flush=True)
    rows = [build(s, a.src, a.out, a.tag, log) for s in scenes]
    head, dirty = src_git(a.src)
    summary = dict(prereg="docs/stage1-E8-prereg.md", tag=a.tag, src=a.src, src_git_head=head, src_git_dirty=dirty, result=a.out,
                   n_png=sum(r["copied"] + r["empty"] for r in rows), copied=sum(r["copied"] for r in rows),
                   empty=sum(r["empty"] for r in rows), scenes=rows)
    log(json.dumps({k: v for k, v in summary.items() if k != "scenes"}))
    json.dump(summary, open(os.path.join(REPO, "MVSeed", "runs", f"e8_seed_set_{a.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
