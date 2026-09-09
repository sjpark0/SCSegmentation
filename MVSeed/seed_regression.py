#!/usr/bin/env python3
"""Pair raw stage-1 seed scores with the mainline's per-frame scores; freeze F0 / N0.

Why.  docs/stage1-headroom.md section 2 says the seed decides the result (frames 1-20 J
= 0.964 x frame-0 J + 0.027, R^2 0.933) -- but its x is the frame 0 the mainline writes
to disk, which is the seed AFTER the second stage re-predicted it from a mask prompt
(MVSeed/README.md, "본류가 디스크에 쓰는 프레임 0은 시드가 아닙니다").  The raw seed
that score_seeds.py measures is one step earlier.  Before every gate in
docs/stage1-plan.md section 5 is read in raw-seed J, S1-E0 has to show that the raw
seed predicts frames 1-20 as well as the re-predicted one does, and how much the
re-prediction costs.  S1-F0 then freezes the failure set F0 (raw seed J < 0.5, non-
reference, GT present) and its complement N0 so that no later experiment can move the
denominator.  This script is both readings; it reads only, and writes only under
--out-dir.

Pairing key is (scene, camera, object).  Seeds come from score_seeds.py's JSON in
either of its layouts -- the 2026-09-09 one ({run: {scene: {c_ini, rows}}}, rows with
gt / pred, which is what MVSeed/runs/baseline_seedJ.json holds) or the current one
({"runs": {run: {"scenes": {scene: ...}}}, optional "baseline", rows with gt_px /
pred_px) -- read on the scene's start_frame; the per-frame side is eval_jf.py's
version-2 raw file (result.per_frame.J = one list per object over the 21 frames).  eval_jf lists only the objects that appear somewhere in that camera's
GT, so a seed row can be unmatched only when its GT at the seed frame is empty; such
rows are outside the population anyway (plan section 2: 664 pairs = non-reference,
GT present, MUVOD basic), and an unmatched row WITH ground truth aborts the run.

Readings, printed and (--report) written as JSON:

  (1) frame-0 J minus raw seed J on the population: mean, median, |delta| > 0.05,
      pairs that fail only after re-prediction (raw >= 0.5, frame 0 < 0.5) and the
      reverse; the same block for the reference camera (GT prompt re-predicted).
  (2) least-squares fit of the frames 1-20 mean J on the raw seed J, and on the
      frame-0 J.  The second must reproduce headroom section 2 (0.964 / 0.027 /
      0.933); the script says whether it does.
  (3) F0 = raw seed J < 0.5 on the population, N0 = the rest.  One record per pair
      with gt_px, raw/frame-0/frames-1-20 J, kind (empty: raw prediction 0 px /
      misplaced / ok), size_bin (<2000 / 2000-10000 / >=10000 px), rig (hemisphere:
      AlexaMeadeExhibit, AlexaMeadeFacePaint, Dog, Welder; pinhole otherwise) and
      view_distance (|view index - c_ini index| in the loaded camera ring, the same
      construction as eval/e4_judge.view_index_map), plus a decomposition of each set.
      Written to <out-dir>/F0.json and N0.json with the SHA-256 of every input and
      the generation time; existing files are refused without --overwrite because
      the freeze is meant to be written once.

Until the raw seed run exists the seeds file may be MVSeed/runs/baseline_seedJ.json,
whose J IS the frame-0 J; then both fits coincide, every delta is 0, and the outputs
carry seeds_equal_frame0 = true so that they cannot be mistaken for the freeze.

    python3 MVSeed/seed_regression.py --seeds MVSeed/runs/baseline_seedJ.json \\
        --run SegMaskSam3XW1CGPS4M --jf Data/MVSeg/jf_e4.json \\
        --method SegMaskSam3XW1CGPS4M --out-dir /tmp/e0_check

Standard library only; runs on the host.
"""
import argparse
import dataclasses
import hashlib
import json
import os
import statistics
import sys
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(REPO, "SCSam3", "demo", "MVSeg.json")
DEFAULT_JF = os.path.join(REPO, "Data", "MVSeg", "jf_e4.json")
DEFAULT_OUT = os.path.join(REPO, "MVSeed", "runs")
METHOD = "SegMaskSam3XW1CGPS4M"

FAIL_TAU = 0.5              # F0 rule: raw seed J below this (plan section 5, S1-F0)
BIG_DELTA = 0.05            # reading (1): |frame-0 J - raw J| above this is "moved"
BOUNDARY = (0.4, 0.6)       # plan section 2: pairs near the failure threshold
TIE = 1e-9                  # |delta| below this is "identical"
SIZE_EDGES = (2000, 10000)  # plan section 2 size bins, in GT pixels at the seed frame
SIZE_BINS = ("<2000", "2000-10000", ">=10000")
HEMISPHERE = ("AlexaMeadeExhibit", "AlexaMeadeFacePaint", "Dog", "Welder")
# stage1-headroom section 2, quoted to three decimals: slope, intercept, R^2
HEADROOM_FIT = (0.964, 0.027, 0.933)
SCHEMA = "MVSeed seed_regression freeze"
SCHEMA_VERSION = 1
FREEZE_FILES = {"F0": "F0.json", "N0": "N0.json"}


# --------------------------------------------------------------------------- config
def cam_name(c, prefix, prefix1):
    return f"{prefix}{c:0{prefix1}d}"


def scene_config(cfg, scene):
    """The MVSeg.json entry for a scene, by key or by folder name (as e4_judge does)."""
    d = cfg.get(scene)
    if d is None:
        d = next((v for v in cfg.values() if v.get("folder") == scene), None)
    if d is None:
        raise KeyError(f"{scene!r} is not in the config")
    return d


def view_index_map(d):
    """camera name -> index in the loaded camera ring, plus the c_ini index.

    Same construction as eval/e4_judge.view_index_map, taking the config entry instead
    of reading the file, so tests can hand in a synthetic ring."""
    perms = d.get("perms") or list(range(d["start_cam"], d["num_cam"] + d["start_cam"]))
    names = {cam_name(x, d["prefix"], d["prefix1"]): perms.index(x) for x in perms}
    return names, perms.index(d["c_ini"])


# --------------------------------------------------------------------------- seeds file
def seed_runs(doc):
    """{run: {scene: {"c_ini", "rows", ...}}} from either score_seeds.py layout.

    Current layout: doc["runs"][run]["scenes"], plus doc["baseline"] as one more run
    (named by its "run" key) when present.  Legacy layout: doc[run] is the scene map."""
    if isinstance(doc.get("runs"), dict) and all("scenes" in v for v in doc["runs"].values()):
        runs = {run: v["scenes"] for run, v in doc["runs"].items()}
        base = doc.get("baseline")
        if isinstance(base, dict) and "scenes" in base:
            runs.setdefault(base.get("run", "baseline"), base["scenes"])
        return runs
    return {run: v for run, v in doc.items() if run != "meta"}


def row_px(r):
    """(gt pixels, raw seed pixels) of a score_seeds row, whichever layout named them."""
    gt = r["gt_px"] if "gt_px" in r else r["gt"]
    pred = r["pred_px"] if "pred_px" in r else r["pred"]
    return int(gt), int(pred)


# --------------------------------------------------------------------------- labels
def size_bin(gt_px):
    if gt_px < SIZE_EDGES[0]:
        return SIZE_BINS[0]
    if gt_px < SIZE_EDGES[1]:
        return SIZE_BINS[1]
    return SIZE_BINS[2]


def rig(scene):
    return "hemisphere" if scene in HEMISPHERE else "pinhole"


def kind(raw_pred_px, raw_J, tau=FAIL_TAU):
    """empty: the raw seed has no pixels; misplaced: it has some but J < tau; else ok."""
    if raw_pred_px == 0:
        return "empty"
    return "misplaced" if raw_J < tau else "ok"


@dataclasses.dataclass
class Pair:
    scene: str
    cam: str
    obj: int
    ref: bool                # camera is the scene's c_ini
    view_distance: int
    gt_px: int               # GT pixels at the seed frame (score_seeds `gt`)
    raw_pred_px: int         # raw seed pixels (score_seeds `pred`)
    raw_J: float             # score_seeds J: the stage-1 seed before re-prediction
    frame0_J: float          # eval_jf per_frame J[0]: the seed after re-prediction
    later_J: float           # mean of eval_jf per_frame J[1:], the frames the seed decides

    @property
    def in_population(self):
        return not self.ref and self.gt_px > 0

    def record(self):
        return dict(scene=self.scene, cam=self.cam, obj=self.obj, gt_px=self.gt_px,
                    raw_pred_px=self.raw_pred_px, raw_J=self.raw_J, frame0_J=self.frame0_J,
                    frames1_20_J=self.later_J, kind=kind(self.raw_pred_px, self.raw_J),
                    size_bin=size_bin(self.gt_px), rig=rig(self.scene),
                    view_distance=self.view_distance)


# --------------------------------------------------------------------------- pairing
def pair(seeds_run, jf_entries, method, cfg):
    """(pairs, unmatched) for one score_seeds run against one eval_jf method.

    `unmatched` lists (scene, cam, obj, gt_px) rows the jf file has no object for; the
    caller decides what to do with the ones that carry ground truth."""
    index = {(e["dataset"], e["camera"]): e for e in jf_entries if e["method"] == method}
    pairs, unmatched = [], []
    for scene in sorted(seeds_run):
        v = seeds_run[scene]
        names, ci = view_index_map(scene_config(cfg, scene))
        for r in v["rows"]:
            gt_px, pred_px = row_px(r)
            e = index.get((scene, r["cam"]))
            if e is None:
                raise KeyError(f"{scene}/{r['cam']}: no {method} entry in the jf file")
            if r["obj"] not in e["objects"]:
                unmatched.append((scene, r["cam"], int(r["obj"]), gt_px))
                continue
            i = e["objects"].index(r["obj"])
            J = e["result"]["per_frame"]["J"][i]
            if len(J) < 2:
                raise ValueError(f"{scene}/{r['cam']} obj {r['obj']}: {len(J)} frame(s), "
                                 "nothing after the seed frame to regress on")
            pairs.append(Pair(scene=scene, cam=r["cam"], obj=int(r["obj"]),
                              ref=(r["cam"] == v["c_ini"]),
                              view_distance=abs(names[r["cam"]] - ci),
                              gt_px=gt_px, raw_pred_px=pred_px,
                              raw_J=float(r["J"]), frame0_J=float(J[0]),
                              later_J=statistics.fmean(J[1:])))
    return pairs, unmatched


# --------------------------------------------------------------------------- readings
def diff_summary(pairs, tau=FAIL_TAU):
    """Reading (1): frame-0 J minus raw seed J over `pairs`."""
    d = [p.frame0_J - p.raw_J for p in pairs]
    return {
        "n": len(pairs),
        "mean_raw_J": statistics.fmean(p.raw_J for p in pairs) if pairs else None,
        "mean_frame0_J": statistics.fmean(p.frame0_J for p in pairs) if pairs else None,
        "delta_mean": statistics.fmean(d) if d else None,
        "delta_median": statistics.median(d) if d else None,
        "n_abs_delta_gt_0.05": sum(1 for x in d if abs(x) > BIG_DELTA),
        "n_identical": sum(1 for x in d if abs(x) < TIE),
        "n_new_fail_at_frame0": sum(1 for p in pairs if p.raw_J >= tau > p.frame0_J),
        "n_recovered_at_frame0": sum(1 for p in pairs if p.raw_J < tau <= p.frame0_J),
        "n_raw_fail": sum(1 for p in pairs if p.raw_J < tau),
        "n_frame0_fail": sum(1 for p in pairs if p.frame0_J < tau),
        "n_raw_in_boundary": sum(1 for p in pairs if BOUNDARY[0] <= p.raw_J < BOUNDARY[1]),
    }


def linfit(xs, ys):
    """Ordinary least squares y = slope * x + intercept, with R^2 (nan when degenerate)."""
    n = len(xs)
    if n < 2:
        return {"n": n, "slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return {"n": n, "slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"n": n, "slope": slope, "intercept": intercept, "r2": r2}


def reproduces_headroom(fit, ref=HEADROOM_FIT):
    """True when the fit rounds to the three-decimal values quoted in the headroom doc."""
    vals = (fit["slope"], fit["intercept"], fit["r2"])
    return all(v == v and round(v, 3) == r for v, r in zip(vals, ref))


def split(pairs, tau=FAIL_TAU):
    """(F0, N0) of the population: raw seed J below tau, and the rest."""
    pop = [p for p in pairs if p.in_population]
    return [p for p in pop if p.raw_J < tau], [p for p in pop if p.raw_J >= tau]


def decompose(pairs):
    """Counts by kind, size bin, rig and view distance, plus pinhole AND >= 2000 px
    (the pairs the geometry arms of plan section 5 can reach)."""
    def count(key):
        out = {}
        for p in pairs:
            k = key(p)
            out[k] = out.get(k, 0) + 1
        return out
    return {
        "n": len(pairs),
        "kind": count(lambda p: kind(p.raw_pred_px, p.raw_J)),
        "size_bin": {b: sum(1 for p in pairs if size_bin(p.gt_px) == b) for b in SIZE_BINS},
        "rig": count(lambda p: rig(p.scene)),
        "pinhole_ge2000": sum(1 for p in pairs
                              if rig(p.scene) == "pinhole" and p.gt_px >= SIZE_EDGES[0]),
        "view_distance": {str(k): v for k, v in sorted(count(lambda p: p.view_distance).items())},
        "scene": count(lambda p: p.scene),
    }


# --------------------------------------------------------------------------- output
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path):
    path = os.path.abspath(path)
    return os.path.relpath(path, REPO) if path.startswith(REPO + os.sep) else path


def input_block(path, **extra):
    return dict(path=rel(path), sha256=sha256_file(path), **extra)


def freeze_record(name, members, inputs, population, seeds_equal_frame0, generated_at):
    rule = f"raw seed J < {FAIL_TAU}" if name == "F0" else f"raw seed J >= {FAIL_TAU}"
    members = sorted(members, key=lambda p: (p.scene, p.cam, p.obj))
    return {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION, "set": name, "rule": rule,
        "generated_at": generated_at, "generated_by": "MVSeed/seed_regression.py",
        "inputs": inputs, "population": population,
        "seeds_equal_frame0": seeds_equal_frame0,
        "n": len(members), "summary": decompose(members),
        "pairs": [p.record() for p in members],
    }


def fmt_fit(fit):
    return (f"기울기 {fit['slope']:.4f}  절편 {fit['intercept']:.4f}  R² {fit['r2']:.4f}"
            f"  (n {fit['n']})")


def print_diff(label, s):
    if not s["n"]:
        print(f"    {label}: (없음)")
        return
    print(f"    {label} n={s['n']}: 원시 J 평균 {s['mean_raw_J']:.4f}  프레임0 J 평균 "
          f"{s['mean_frame0_J']:.4f}  Δ(프레임0−원시) 평균 {s['delta_mean']:+.4f}  중앙값 "
          f"{s['delta_median']:+.4f}  |Δ|>{BIG_DELTA} {s['n_abs_delta_gt_0.05']}  동일 "
          f"{s['n_identical']}")
    print(f"      실패(J<{FAIL_TAU}) 원시 {s['n_raw_fail']} → 프레임0 {s['n_frame0_fail']}: "
          f"프레임0에서 새로 실패 {s['n_new_fail_at_frame0']}, 반대 {s['n_recovered_at_frame0']}"
          f"  경계 구간 [{BOUNDARY[0]},{BOUNDARY[1]}) 원시 {s['n_raw_in_boundary']}")


def print_decomp(label, d):
    sb = "  ".join(f"{b} {d['size_bin'][b]}" for b in SIZE_BINS)
    kd = "  ".join(f"{k} {v}" for k, v in sorted(d["kind"].items()))
    rg = "  ".join(f"{k} {v}" for k, v in sorted(d["rig"].items()))
    vd = "  ".join(f"d{k}:{v}" for k, v in d["view_distance"].items())
    print(f"    {label} n={d['n']}: {kd} | {sb} | {rg} | 핀홀∧≥{SIZE_EDGES[0]} "
          f"{d['pinhole_ge2000']} | 시점 거리 {vd}")


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", required=True, help="score_seeds.py output JSON")
    ap.add_argument("--run", default=None,
                    help="run name inside --seeds (default: the only one there)")
    ap.add_argument("--jf", default=DEFAULT_JF, help="eval_jf.py version-2 raw file")
    ap.add_argument("--method", default=METHOD, help="method column of --jf to pair with")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--out-dir", default=DEFAULT_OUT, help="where F0.json / N0.json go")
    ap.add_argument("--report", default=None, help="also write readings (1)-(3) as JSON here")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing F0.json / N0.json (the freeze is written once)")
    args = ap.parse_args(argv)

    seeds_all = seed_runs(json.load(open(args.seeds, encoding="utf-8")))
    run = args.run
    if run is None:
        if len(seeds_all) != 1:
            sys.exit(f"--run is required: {args.seeds} holds {sorted(seeds_all)}")
        run = next(iter(seeds_all))
    if run not in seeds_all:
        sys.exit(f"{run!r} is not in {args.seeds}: {sorted(seeds_all)}")
    jf = json.load(open(args.jf, encoding="utf-8"))
    if not any(e["method"] == args.method for e in jf):
        sys.exit(f"{args.method!r} is not in {args.jf}")
    cfg = json.load(open(args.config, encoding="utf-8"))
    for name in FREEZE_FILES.values():
        p = os.path.join(args.out_dir, name)
        if os.path.exists(p) and not args.overwrite:
            sys.exit(f"{p} exists; the freeze is written once -- pass --overwrite to redo it")

    pairs, unmatched = pair(seeds_all[run], jf, args.method, cfg)
    bad = [u for u in unmatched if u[3] > 0]
    if bad:
        for scene, cam, obj, gt in bad[:10]:
            print(f"  {scene}/{cam} obj {obj} (gt {gt} px) has no {args.method} object in {args.jf}")
        sys.exit(f"{len(bad)} seed row(s) with ground truth could not be paired; "
                 "the population would be wrong")
    pop = [p for p in pairs if p.in_population]
    refp = [p for p in pairs if p.ref and p.gt_px > 0]
    n_gt_empty = sum(1 for p in pairs if p.gt_px == 0) + len(unmatched)
    if not pop:
        sys.exit("empty population (no non-reference pair with ground truth)")

    print(f"입력   seeds {rel(args.seeds)} [{run}]")
    print(f"       jf    {rel(args.jf)} [{args.method}]")
    print(f"짝짓기 {len(pairs)} 쌍 (jf에 객체 없음 {len(unmatched)}, 모두 정답 없음)  "
          f"모집단(비기준·정답 있음) {len(pop)}  기준 {len(refp)}  정답 없음 {n_gt_empty}")

    # (1)
    d_pop, d_ref = diff_summary(pop), diff_summary(refp)
    equal = d_pop["n_identical"] == d_pop["n"]
    print(f"(1) 원시 시드 J vs 프레임 0 J (재예측 후)")
    print_diff("비기준", d_pop)
    print_diff("기준  ", d_ref)
    if equal:
        print(f"    * 비기준 전 쌍에서 원시 = 프레임 0: --seeds 는 프레임 0 파일입니다 "
              f"(seeds_equal_frame0)")

    # (2)
    ys = [p.later_J for p in pop]
    fit_raw = linfit([p.raw_J for p in pop], ys)
    fit_f0 = linfit([p.frame0_J for p in pop], ys)
    ok = reproduces_headroom(fit_f0)
    print(f"(2) 회귀  y = 프레임 1~20 평균 J  (비기준 {len(pop)})")
    print(f"    y ~ 원시 시드 J : {fmt_fit(fit_raw)}")
    print(f"    y ~ 프레임 0 J  : {fmt_fit(fit_f0)}  headroom §2 "
          f"{'/'.join(f'{v:.3f}' for v in HEADROOM_FIT)} 재현 {'OK' if ok else '불일치'}")

    # (3)
    f0, n0 = split(pairs)
    print(f"(3) F0/N0 동결 (원시 시드 J < {FAIL_TAU}, 모집단 {len(pop)})")
    print_decomp("F0", decompose(f0))
    print_decomp("N0", decompose(n0))

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    inputs = {"seeds": input_block(args.seeds, run=run),
              "jf": input_block(args.jf, method=args.method),
              "config": input_block(args.config)}
    population = {
        "definition": "non-reference camera, GT present at the seed frame, "
                      "MUVOD basic objects (score_seeds.py rows)",
        "n": len(pop), "n_reference": len(refp), "n_gt_empty": n_gt_empty,
        "n_unmatched": len(unmatched),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    for name, members in (("F0", f0), ("N0", n0)):
        out = os.path.join(args.out_dir, FREEZE_FILES[name])
        with open(out, "w", encoding="utf-8") as f:
            json.dump(freeze_record(name, members, inputs, population, equal, generated_at),
                      f, indent=1)
        print(f"wrote {out}")
    if args.report:
        report = {
            "schema": "MVSeed seed_regression readings", "schema_version": SCHEMA_VERSION,
            "generated_at": generated_at, "inputs": inputs, "population": population,
            "seeds_equal_frame0": equal,
            "diff": {"nonref": d_pop, "ref": d_ref},
            "fit": {"frames1_20_on_raw": fit_raw, "frames1_20_on_frame0": fit_f0,
                    "headroom_ref": list(HEADROOM_FIT), "reproduces_headroom": ok},
            "F0": decompose(f0), "N0": decompose(n0),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
