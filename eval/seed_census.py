#!/usr/bin/env python3
"""Seed census: the zero-cost P5 diagnostic (REPORT.md P5, ROADMAP Phase 1).

What a seed is.  runMVSeg.py builds one cross-view session from frame
`start_frame` of every camera, prompts the reference camera with its GT, runs
PropagateAcrossViews, and then TrackForward() prompts every tracked camera at
frame `start_frame` with the mask the cross-view pass produced (a zero mask
when it produced none) and tracks forward.  The first written frame of every
scored camera is therefore that mask - the seed the temporal tracker was
conditioned on for all 21 frames.  For the reference camera it is the GT
prompt itself, re-predicted.  So `<method>/<cam>/<start_frame>/<obj>.png` IS
the seed, and this script reads nothing else from the method folders.

Per (dataset, camera, object, method) it computes, at the seed frame:

    gt_area, pred_area   pixels
    ref_gt_area          pixels of the object in the reference camera's seed
                         GT, i.e. the size of the prompt the runner gave it
    iou                  IoU(pred, GT); union == 0 scores 1.0 like eval_jf
    pred_missing         no seed file (or unreadable)

and classifies the seed:

    no_folder    `<dataset>/<method>/` does not exist.  Nothing to classify:
                 the (dataset, method) is skipped with a warning, its rows
                 carry this class, and it is neither degenerate nor part of
                 the averages.
    unreachable  id absent from the reference camera's seed GT.  The runner
                 prompts `(ref_gt == id)` for id in 1..max_id, so this is a
                 zero prompt, filtered at registration, and the tracker never
                 predicts a pixel.  J = F = 0 on every frame whose GT contains
                 the object; frames without it score 1.0 (DAVIS convention,
                 eval_jf: union == 0 -> 1.0), so the stored J = F equals the
                 fraction of GT-empty frames (Breakfast v9 obj 17: 7 of 21
                 frames -> 0.3333).  Reference camera per the max-id rule,
                 taken from eval_jf.dataset_meta so the two scripts cannot
                 disagree.
    gt_empty     reachable but gt_area == 0 at the seed frame (the object
                 appears later in this camera).  Reported separately, not
                 called degenerate: an empty seed is the right seed here.
    missing      reachable, GT present, no seed file
    empty        seed file exists, 0 px
    tiny         0 < pred_area < max(64, 0.05 * gt_area).  This is the
                 GT-oracle variant of the P5 area test: REPORT.md P5's runtime
                 test compares against 0.05 x the median predicted area across
                 views, because at run time there is no GT to compare with.
    misaligned   not tiny, gt_area > 0, IoU < --misaligned-iou (0.10)
    ok           everything else

`degenerate` = missing | empty | tiny | misaligned.  Precedence is the order
above: an unreachable object stays unreachable even when its seed file is
also missing, and gt_empty takes precedence over missing/empty because an
absent or empty prediction on an empty GT is correct.  The SAM 2 runner
writes a 0 px file where the SAM 3 runner writes nothing, so `empty` under
SegMaskNew1 and `missing` under the SAM 3 folders are the same failure mode
(0-px file vs no file) - but not the same set of objects.  The script
computes the overlap between the first method's empty|missing rows and every
other method's (`empty_missing_overlap` in the JSON, quoted in the md) instead
of asserting an identity.

Each row is joined with the stored per-object J_all / F_all from the jf_*.json
files, and per (dataset, method) the script reports

    jf_stored   mean over (cam, obj) of (J + F) / 2 - identical to the
                report_jf.py `as-is / all` table in docs/experiments.md
    jf_repair   the same with every degenerate object given
                max(its stored (J + F) / 2, the mean of the median J and the
                median F of that dataset-method's `ok` objects) - the repair
                target; the max keeps a repair from lowering an object whose
                stored score already beats the median
    jf_upper    the same with every degenerate object given J = F = 1
    jf_repair_promptable
                jf_repair applied only to degenerate objects whose reference
                prompt is at least 64 px (ref_gt_area >= 64).  A prompt of a
                few pixels is lost at the tracker's input resolution in every
                view, so no donor view exists for P5 to copy from.

plus the average over datasets of each, and a sensitivity block that
re-counts degenerate seeds and recomputes the averages at --misaligned-iou
in {0.10, 0.20, 0.30}.  Rows with class no_folder, unreachable or gt_empty
are never altered in jf_repair / jf_upper.

The closing paragraph of the md ("판독") is generated entirely from the
computed rows and summary - every number, threshold and (camera, object)
list in it - so it cannot go stale under another --methods or
--misaligned-iou; the only literals are the REPORT.md anchor quotes, which
are printed next to the computed values for comparison.

Runs inside the scsam3 container (numpy + cv2), as yourself so the outputs
are not root-owned.  CPU only, about a minute:

    docker run --rm --user $(id -u):$(id -g) -v /:/host -w /host/$PWD scsam3 \\
        python eval/seed_census.py

Idempotent: reads PNGs and jf_*.json, writes only --out-json and --out-md,
no timestamps, deterministic ordering, so a re-run reproduces both files
byte for byte.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_jf import cam_name, dataset_meta, frame_key  # noqa: E402

REPO = os.path.dirname(HERE)
DEFAULT_ROOT = os.path.join(REPO, "Data", "MVSeg")
DEFAULT_CONFIG = os.path.join(REPO, "SCSam3", "demo", "MVSeg.json")
DEFAULT_METHODS = ["SegMaskNew1", "SegMaskSam3OneStage", "SegMaskSam3MVOpt"]
DEFAULT_OUT_JSON = os.path.join(DEFAULT_ROOT, "seed_census.json")
DEFAULT_OUT_MD = os.path.join(REPO, "docs", "raw", "seed_census.md")

# Where the stored per-object scores live (docs/experiments.md, "파일 목록").
# Anything not listed is a SAM 2 folder scored in jf_raw.json.
JF_FILES = {
    "SegMaskSam3OneStage": "jf_sam3_onestage.json",
    "SegMaskSam3MVOpt": "jf_sam3_mvopt_all.json",
    "SegMaskSam3OneStageNew": "jf_sam3_onestagenew.json",
}
JF_DEFAULT_FILE = "jf_raw.json"

TINY_MIN_PX = 64
TINY_FRAC = 0.05
DEFAULT_MISALIGNED_IOU = 0.10
SENSITIVITY_IOUS = [0.10, 0.20, 0.30]
BORDERLINE_IOU = 0.30          # `ok` seeds below this are listed for the eye
CLASSES = ["no_folder", "unreachable", "gt_empty", "missing", "empty", "tiny",
           "misaligned", "ok"]
DEGENERATE = ("missing", "empty", "tiny", "misaligned")
ABSENT_SEED = ("empty", "missing")     # the two spellings of "no mask written"
SCHEMA_VERSION = 2

# Anchors from REPORT.md A3 / P5, printed at the end so the run can be checked
# against the analysis they came from.
ANCHOR_DATASET, ANCHOR_CAM = "Welder", "camera_0004"
ANCHOR_OBJS = [8, 12, 14]
ANCHOR_TINY_OBJS = [8, 14]      # REPORT.md P5: the two tiny seeds
ANCHOR_LOWIOU_OBJ = 12          # REPORT.md P5: the "misaligned" seed
# REPORT.md P5's own estimate for the anchor dataset, quoted next to the
# computed value; it is a citation, not a number this script derives.
REPORT_P5_QUOTE = "Welder 0.8437 → ~0.879, 헤드라인 +0.0023~0.0027"


# ------------------------------------------------------------------ reading
def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = img[..., 0]
    return img


def classify(r, misaligned_iou):
    """Class of one row dict; `cls` is rewritten by the sensitivity pass.

    `tiny` is the GT-oracle variant of the P5 area test (the runtime test in
    REPORT.md P5 uses 0.05 x the median predicted area across views).
    """
    if r["no_folder"]:
        return "no_folder"
    if not r["reachable"]:
        return "unreachable"
    if r["gt_area"] == 0:
        return "gt_empty"
    if r["pred_missing"]:
        return "missing"
    if r["pred_area"] == 0:
        return "empty"
    if r["pred_area"] < max(TINY_MIN_PX, TINY_FRAC * r["gt_area"]):
        return "tiny"
    if r["iou"] < misaligned_iou:
        return "misaligned"
    return "ok"


def census_camera(job):
    """One (dataset, camera): the seed rows of every object for every method."""
    root, dataset, cam, start_frame, ref_areas, methods = job
    ds_dir = os.path.join(root, dataset)
    gt_dir = os.path.join(ds_dir, "Mask", cam)

    # The object set is what eval_jf scores: every id that appears in this
    # camera's GT at any frame.
    obj_ids, seed_gt = set(), None
    for f in sorted((f for f in os.listdir(gt_dir) if f.endswith(".png")),
                    key=frame_key):
        g = read_gray(os.path.join(gt_dir, f))
        if g is None:
            sys.exit(f"unreadable ground truth: {os.path.join(gt_dir, f)}")
        obj_ids.update(np.unique(g).tolist())
        if frame_key(f) == start_frame:
            seed_gt = g
    obj_ids.discard(0)
    obj_ids = sorted(int(o) for o in obj_ids)
    if seed_gt is None:
        sys.exit(f"{dataset}/{cam}: no GT frame {start_frame:06d}.png")
    h, w = seed_gt.shape[:2]

    rows = []
    for method in methods:
        # A method folder absent for this dataset is reported as class
        # no_folder (main() warns once per (dataset, method)), not as a
        # dataset full of missing seeds.
        no_folder = not os.path.isdir(os.path.join(ds_dir, method))
        seed_dir = os.path.join(ds_dir, method, cam, f"{start_frame:d}")
        for obj in obj_ids:
            gm = seed_gt == obj
            gt_area = int(np.count_nonzero(gm))
            pm, missing = None, True
            p = os.path.join(seed_dir, f"{obj}.png")
            if not no_folder and os.path.exists(p):
                pm = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if pm is not None:
                missing = False
                if pm.shape[:2] != (h, w):
                    pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_NEAREST)
                pm = pm > 127
            else:
                pm = np.zeros((h, w), dtype=bool)
            pred_area = int(np.count_nonzero(pm))
            union = int(np.count_nonzero(gm | pm))
            iou = 1.0 if union == 0 else int(np.count_nonzero(gm & pm)) / union
            ref_gt_area = ref_areas.get(obj, 0)
            rows.append({
                "dataset": dataset, "camera": cam, "obj": obj, "method": method,
                "gt_area": gt_area, "pred_area": pred_area,
                "ref_gt_area": ref_gt_area,
                "pred_missing": missing, "iou": round(iou, 6),
                "reachable": ref_gt_area > 0,
                "no_folder": no_folder,
            })
    return dataset, cam, obj_ids, rows


# ----------------------------------------------------------------- scoring
def load_scores(root, methods):
    """(dataset, camera, method, obj) -> (J_all, F_all) from the jf files."""
    scores, files = {}, {}
    for m in methods:
        name = JF_FILES.get(m, JF_DEFAULT_FILE)
        files[m] = name
        path = os.path.join(root, name)
        if not os.path.exists(path):
            print(f"warning: {path} not found, {m} rows get J = F = null",
                  file=sys.stderr)
            continue
        for e in json.load(open(path)):
            if e["method"] != m or e.get("result") is None:
                continue
            r = e["result"]
            for i, obj in enumerate(e["objects"]):
                scores[(e["dataset"], e["camera"], m, int(obj))] = (
                    r["J_all"][i], r["F_all"][i])
    return scores, files


def mean(v):
    return sum(v) / len(v) if v else None


def median(v):
    if not v:
        return None
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def summarize(rows, datasets, methods):
    """Per (method, dataset) class counts and the J&F numbers, plus averages.

    Uses each row's current `cls`, so the sensitivity pass can call it again
    after reclassifying.  A (method, dataset) whose folder is absent (every
    row no_folder) keeps its counts but gets no J&F numbers and is left out
    of the averages.
    """
    summary = {m: {} for m in methods}
    for m in methods:
        for d in datasets:
            rs = [r for r in rows if r["method"] == m and r["dataset"] == d]
            no_folder = bool(rs) and all(r["no_folder"] for r in rs)
            scored = [] if no_folder else [r for r in rs if r["J"] is not None]
            counts = {c: sum(1 for r in rs if r["cls"] == c) for c in CLASSES}
            counts["degenerate"] = sum(counts[c] for c in DEGENERATE)
            counts["degenerate_prompt_tiny"] = sum(
                1 for r in rs if r["cls"] in DEGENERATE
                and r["ref_gt_area"] < TINY_MIN_PX)
            ok = [r for r in scored if r["cls"] == "ok"]
            med_j, med_f = median([r["J"] for r in ok]), median([r["F"] for r in ok])
            med = (med_j + med_f) / 2 if med_j is not None else None
            stored, repair, repair_p, upper = [], [], [], []
            for r in scored:
                jf = (r["J"] + r["F"]) / 2
                stored.append(jf)
                if r["cls"] in DEGENERATE:
                    # Repair target: never below what the object already
                    # scores, otherwise the median of the ok objects.
                    fixed = max(jf, med) if med is not None else jf
                    repair.append(fixed)
                    upper.append(1.0)
                    repair_p.append(fixed if r["ref_gt_area"] >= TINY_MIN_PX else jf)
                else:
                    repair.append(jf)
                    upper.append(jf)
                    repair_p.append(jf)
            per_cam = {}
            for cam in sorted({r["camera"] for r in rs}):
                cr = [r for r in scored if r["camera"] == cam]
                per_cam[cam] = {
                    "n_objects": len(cr),
                    "J": mean([r["J"] for r in cr]),
                    "F": mean([r["F"] for r in cr]),
                    "jf": mean([(r["J"] + r["F"]) / 2 for r in cr]),
                    "degenerate": sum(1 for r in rs if r["camera"] == cam
                                      and r["cls"] in DEGENERATE),
                }
            summary[m][d] = {
                "n_objects": len(rs),
                "n_scored": len(scored),
                "no_folder": no_folder,
                "counts": counts,
                "median_ok_J": med_j, "median_ok_F": med_f,
                "jf_stored": mean(stored),
                "jf_repair": mean(repair),
                "jf_repair_promptable": mean(repair_p),
                "jf_upper": mean(upper),
                "per_camera": per_cam,
            }
    average = {}
    for m in methods:
        vals = [s for s in summary[m].values() if s["jf_stored"] is not None]
        average[m] = {
            "n_datasets": len(vals),
            "n_datasets_no_folder": sum(1 for s in summary[m].values() if s["no_folder"]),
            "n_objects": sum(s["n_objects"] for s in vals),
            "counts": {c: sum(s["counts"][c] for s in vals)
                       for c in CLASSES + ["degenerate", "degenerate_prompt_tiny"]},
            "jf_stored": mean([s["jf_stored"] for s in vals]),
            "jf_repair": mean([s["jf_repair"] for s in vals]),
            "jf_repair_promptable": mean([s["jf_repair_promptable"] for s in vals]),
            "jf_upper": mean([s["jf_upper"] for s in vals]),
        }
    return summary, average


def has_numbers(summary, m, d):
    """True when (method, dataset) has stored J&F (folder present, scored)."""
    s = summary[m].get(d)
    return s is not None and s["jf_stored"] is not None


def absent_seed_overlap(rows, methods):
    """How far the first method's empty|missing seeds coincide with each
    other method's.

    The SAM 2 runner writes a 0-px file (empty) where the SAM 3 runner writes
    no file (missing): one failure mode, two spellings.  This measures whether
    the two runners fail on the same (dataset, camera, object) triples rather
    than assuming they do.
    """
    def triples(m):
        return {(r["dataset"], r["camera"], r["obj"]) for r in rows
                if r["method"] == m and r["cls"] in ABSENT_SEED}

    def spelled(m):
        return {c: sum(1 for r in rows if r["method"] == m and r["cls"] == c)
                for c in ABSENT_SEED}

    base = methods[0]
    a = triples(base)
    out = []
    for m in methods[1:]:
        b = triples(m)
        out.append({
            "base": base, "other": m,
            "classes": list(ABSENT_SEED),
            "n_base": len(a), "n_other": len(b), "n_overlap": len(a & b),
            "base_spelling": spelled(base), "other_spelling": spelled(m),
            "base_only": [list(t) for t in sorted(a - b)],
            "other_only": [list(t) for t in sorted(b - a)],
        })
    return out


def sensitivity(rows, datasets, methods, ious):
    """Degenerate counts and averages when the misaligned threshold moves."""
    keep = [r["cls"] for r in rows]
    out = {}
    for t in ious:
        for r in rows:
            r["cls"] = classify(r, t)
        summ, avg = summarize(rows, datasets, methods)
        out[f"{t:.2f}"] = {
            m: {
                "degenerate": avg[m]["counts"]["degenerate"],
                "misaligned": avg[m]["counts"]["misaligned"],
                "jf_stored": avg[m]["jf_stored"],
                "jf_repair": avg[m]["jf_repair"],
                "jf_upper": avg[m]["jf_upper"],
                "welder": {k: summ[m][ANCHOR_DATASET][k]
                           for k in ("jf_stored", "jf_repair", "jf_upper")}
                if has_numbers(summ, m, ANCHOR_DATASET) else None,
            } for m in methods}
    for r, c in zip(rows, keep):
        r["cls"] = c
    return out


# ---------------------------------------------------------------- reporting
def f4(x):
    return "—" if x is None else f"{x:.4f}"


def sgn(x):
    return "—" if x is None else f"{x:+.4f}"


def triple_list(triples):
    return ", ".join("/".join(str(x) for x in t) for t in triples) or "없음"


def overlap_text(out):
    """C3: the empty-vs-missing overlap sentence(s), from the computed sets."""
    ov = out["empty_missing_overlap"]
    if not ov:
        return ""
    base = ov[0]["base"]
    parts = []
    for o in ov:
        parts.append(
            f"`{base}` {o['n_base']}개 vs `{o['other']}` {o['n_other']}개 중 "
            f"{o['n_overlap']}개가 겹칩니다(`{base}`에만: {triple_list(o['base_only'])}; "
            f"`{o['other']}`에만: {triple_list(o['other_only'])})")
    return ("같은 실패 양상(0-px 파일 vs 파일 없음)이지만 같은 객체 집합은 아닙니다 — "
            "empty+missing 기준으로 " + "; ".join(parts) + ".")


def write_md(path, out):
    methods, datasets = out["methods"], sorted(out["datasets"])
    summary, average, thr = out["summary"], out["average"], out["thresholds"]
    rows = out["rows"]
    L = []
    L.append("# Seed census — cross-view seed 품질 진단 (P5, 비용 0)")
    L.append("")
    L.append("`eval/seed_census.py`가 생성. 마스크 PNG와 `jf_*.json`만 읽고 GPU를 쓰지 않습니다. "
             "REPORT.md P5의 \"먼저 비용 0 진단\" 항목이며 ROADMAP Phase 1의 퇴화 시드 집계입니다.")
    L.append("")
    L.append("**seed란.** `runMVSeg.py`의 `TrackForward()`는 cross-view pass가 만든 마스크를 "
             "각 카메라의 `start_frame`에 mask prompt로 넣고(없으면 zero mask) 앞으로만 추적합니다. "
             "따라서 각 scored 카메라의 **첫 기록 프레임** `<method>/<cam>/<start_frame>/<obj>.png`이 "
             "temporal tracker가 21프레임 내내 조건화된 seed 그 자체입니다 "
             "(reference 카메라에서는 GT prompt를 재예측한 것). 객체 집합은 `eval_jf.py`가 채점하는 것과 같이 "
             "그 카메라 GT의 어느 프레임에든 등장하는 id 전부입니다.")
    L.append("")
    L.append("**분류 규칙** (우선순위 순):")
    L.append("")
    L.append("| 클래스 | 조건 |")
    L.append("|---|---|")
    L.append("| no_folder | 그 데이터셋에 `<method>/` 폴더 자체가 없음. 분류할 것이 없으므로 그 (데이터셋, 방법)은 경고와 함께 건너뛰고 평균에서 제외. 퇴화가 아님 |")
    L.append("| unreachable | 객체 id가 reference 카메라(max-id 규칙, `eval_jf.dataset_meta`)의 seed GT에 없음 → zero prompt, tracker가 한 픽셀도 예측하지 않음. GT에 객체가 있는 프레임은 J=F=0, 없는 프레임은 1.0(DAVIS 관례, `eval_jf`: union=0 → 1.0)이므로 저장된 J=F는 GT가 빈 프레임의 비율(예: Breakfast v9 obj 17은 21프레임 중 7프레임 → 0.3333). 구조적이며 어떤 방법도 못 고침 (P8의 몫) |")
    L.append("| gt_empty | 도달 가능하지만 이 카메라의 seed 프레임 GT에 0 px (나중에 등장). 퇴화가 아님, 별도 집계 |")
    L.append("| missing | seed 파일 없음 |")
    L.append("| empty | seed 파일은 있으나 0 px |")
    L.append(f"| tiny | 0 < pred_area < max({thr['tiny_min_px']}, {thr['tiny_frac']} × gt_area). "
             f"P5 면적 검사의 **GT-oracle 변형** — REPORT.md P5의 런타임 검사는 GT 없이 "
             f"{thr['tiny_frac']} × (뷰 전체 예측 면적의 중앙값)과 비교합니다 |")
    L.append(f"| misaligned | tiny가 아니고 gt_area > 0인데 IoU < {thr['misaligned_iou']:.2f} |")
    L.append("| ok | 나머지 |")
    L.append("")
    L.append("**퇴화(degenerate)** = missing + empty + tiny + misaligned. "
             "SAM 2 러너는 0 px 파일을 쓰고 SAM 3 러너는 파일을 안 쓰므로 `SegMaskNew1`의 empty와 "
             "SAM 3 폴더의 missing은 " + overlap_text(out) + " "
             "**저장된 J&F** = `(cam, obj)` 쌍의 (J+F)/2 평균 = `report_jf.py` as-is/all 표(`docs/experiments.md`). "
             "**복구 목표 J&F** = 퇴화 객체에 max(저장된 (J+F)/2, 그 데이터셋·방법의 ok 객체 중앙값 J와 중앙값 F의 평균)을 준 값 "
             "— 저장값이 이미 중앙값보다 높은 객체는 낮추지 않습니다. "
             "**상한 J&F** = 퇴화 객체를 J=F=1로 둔 값. no_folder·unreachable·gt_empty는 두 계산에서 건드리지 않습니다.")
    L.append("")
    jf_files = ", ".join(f"`{m}` ← `{f}`" for m, f in out["jf_files"].items())
    L.append(f"점수 출처: {jf_files}. 데이터셋별 reference 카메라: " + ", ".join(
        f"{d} `{out['datasets'][d]['ref_cam']}`" for d in datasets) + ".")
    L.append("")
    if out["no_folder"]:
        L.append("폴더가 없어 건너뛴 (데이터셋, 방법): " + ", ".join(
            f"{d} `{m}`" for d, m in out["no_folder"]) + ".")
        L.append("")

    # --- per-method tables
    for m in methods:
        L.append(f"## {m}")
        L.append("")
        L.append("| 데이터셋 | 객체 수 | unreachable | missing | empty | tiny | misaligned | 저장된 J&F | 복구 목표 J&F | 상한 J&F |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        skipped = []
        for d in datasets:
            s = summary[m].get(d)
            if s is None:
                continue
            if s["no_folder"]:
                skipped.append(d)
                continue
            c = s["counts"]
            L.append(f"| {d} | {s['n_objects']} | {c['unreachable']} | {c['missing']} | "
                     f"{c['empty']} | {c['tiny']} | {c['misaligned']} | "
                     f"{f4(s['jf_stored'])} | {f4(s['jf_repair'])} | {f4(s['jf_upper'])} |")
        a = average[m]
        c = a["counts"]
        L.append(f"| **평균 ({a['n_datasets']}개)** | {a['n_objects']} | {c['unreachable']} | "
                 f"{c['missing']} | {c['empty']} | {c['tiny']} | {c['misaligned']} | "
                 f"**{f4(a['jf_stored'])}** | **{f4(a['jf_repair'])}** | **{f4(a['jf_upper'])}** |")
        L.append("")
        L.append(f"gt_empty {c['gt_empty']}개, ok {c['ok']}개 (합계 {a['n_objects']}개 중). "
                 f"평균 행의 J&F는 데이터셋별 값의 평균이고 객체 수 열은 합계입니다."
                 + (f" 폴더 없음(no_folder, 제외): {', '.join(skipped)}." if skipped else ""))
        L.append("")

    # --- cross-method average table
    L.append(f"## {len(datasets)}개 데이터셋 평균 — 방법 비교")
    L.append("")
    L.append("| 방법 | 퇴화 객체 | 그중 ref 프롬프트 <64 px | 저장된 J&F | 복구 목표 J&F | Δ복구 | 복구 목표(프롬프트 ≥64 px만) | Δ | 상한 J&F | Δ상한 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in methods:
        a = average[m]
        if a["jf_stored"] is None:
            continue
        name = m if a["n_datasets"] == len(datasets) else f"{m} ({a['n_datasets']}개)"
        L.append(f"| {name} | {a['counts']['degenerate']} | {a['counts']['degenerate_prompt_tiny']} | "
                 f"{f4(a['jf_stored'])} | {f4(a['jf_repair'])} | "
                 f"{sgn(a['jf_repair'] - a['jf_stored'])} | "
                 f"{f4(a['jf_repair_promptable'])} | "
                 f"{sgn(a['jf_repair_promptable'] - a['jf_stored'])} | "
                 f"{f4(a['jf_upper'])} | {sgn(a['jf_upper'] - a['jf_stored'])} |")
    L.append("")
    L.append("\"ref 프롬프트 <64 px\"는 reference 카메라의 seed GT에서 그 객체가 64 px 미만인 경우입니다 — "
             "프롬프트 자체가 tracker 입력 해상도에서 사라지므로 어떤 뷰에도 donor가 없고 P5로 복구할 수 없습니다.")
    L.append("")

    # --- sensitivity
    L.append("## misaligned 임계값 민감도")
    L.append("")
    L.append("| IoU 임계값 | 방법 | 퇴화 객체 | misaligned | 평균 저장 | 평균 복구 목표 | 평균 상한 | Welder 저장 → 복구 → 상한 |")
    L.append("|---:|---|---:|---:|---:|---:|---:|---|")
    for t, per_m in out["sensitivity"].items():
        for m in methods:
            s = per_m[m]
            w = s["welder"]
            wtxt = (f"{w['jf_stored']:.4f} → {w['jf_repair']:.4f} → {w['jf_upper']:.4f}"
                    if w else "—")
            L.append(f"| {t} | {m} | {s['degenerate']} | {s['misaligned']} | "
                     f"{f4(s['jf_stored'])} | {f4(s['jf_repair'])} | {f4(s['jf_upper'])} | {wtxt} |")
    L.append("")

    # --- anchors
    L.append(f"## REPORT.md 앵커 (A3 / P5) — {ANCHOR_DATASET} {ANCHOR_CAM}")
    L.append("")
    L.append("| 방법 | obj | 클래스 | gt_area | pred_area | IoU | J | F |")
    L.append("|---|---:|---|---:|---:|---:|---:|---:|")
    by_key = {(r["dataset"], r["camera"], r["obj"], r["method"]): r for r in rows}
    for m in methods:
        for obj in ANCHOR_OBJS:
            r = by_key.get((ANCHOR_DATASET, ANCHOR_CAM, obj, m))
            if r is None:
                continue
            L.append(f"| {m} | {obj} | {r['cls']} | {r['gt_area']} | {r['pred_area']} | "
                     f"{r['iou']:.3f} | {f4(r['J'])} | {f4(r['F'])} |")
    L.append("")
    base = methods[0]
    if has_numbers(summary, base, ANCHOR_DATASET):
        cams = sorted(summary[base][ANCHOR_DATASET]["per_camera"])
        n_tot = summary[base][ANCHOR_DATASET]["n_scored"]
        L.append(f"{ANCHOR_DATASET} 카메라별 저장값 J / F / J&F (퇴화 seed 수), 그리고 "
                 f"`{base}` 대비 데이터셋 J&F 차이에 대한 카메라별 기여 "
                 f"(= ΔJ&F<sub>cam</sub> × n<sub>cam</sub> / {n_tot}):")
        L.append("")
        L.append("| 방법 | " + " | ".join(cams) + " | 데이터셋 | " +
                 " | ".join(f"기여 {c}" for c in cams) + " | 합 |")
        L.append("|---|" + "---:|" * (2 * len(cams) + 2))
        for m in methods:
            if not has_numbers(summary, m, ANCHOR_DATASET):
                continue
            s = summary[m][ANCHOR_DATASET]
            cells, contrib = [], []
            for c in cams:
                pc = s["per_camera"][c]
                cells.append(f"{pc['J']:.4f} / {pc['F']:.4f} / {pc['jf']:.4f} ({pc['degenerate']})")
                b = summary[base][ANCHOR_DATASET]["per_camera"][c]
                contrib.append((pc["jf"] - b["jf"]) * pc["n_objects"] / n_tot)
            L.append(f"| {m} | " + " | ".join(cells) + f" | {f4(s['jf_stored'])} | " +
                     " | ".join(sgn(x) for x in contrib) + f" | {sgn(sum(contrib))} |")
        L.append("")

    # --- every degenerate row
    L.append("## 퇴화 seed 전체 목록")
    L.append("")
    for m in methods:
        deg = [r for r in rows if r["method"] == m and r["cls"] in DEGENERATE]
        L.append(f"### {m} — {len(deg)}개")
        L.append("")
        if not deg:
            L.append("없음.")
            L.append("")
            continue
        L.append("| 데이터셋 | 카메라 | obj | 클래스 | gt_area | pred_area | ref 프롬프트 px | IoU | J | F | ref |")
        L.append("|---|---|---:|---|---:|---:|---:|---:|---:|---:|:---:|")
        for r in deg:
            ref = "●" if r["is_ref_cam"] else ""
            L.append(f"| {r['dataset']} | {r['camera']} | {r['obj']} | {r['cls']} | "
                     f"{r['gt_area']} | {r['pred_area']} | {r['ref_gt_area']} | {r['iou']:.3f} | "
                     f"{f4(r['J'])} | {f4(r['F'])} | {ref} |")
        L.append("")
    L.append("`ref` ●는 reference 카메라(seed = GT prompt 재예측). "
             "\"ref 프롬프트 px\"는 reference 카메라 seed GT에서의 면적.")
    L.append("")

    # --- borderline: ok but low IoU
    L.append(f"## 경계 사례 — ok이지만 IoU < {BORDERLINE_IOU:.2f}")
    L.append("")
    L.append("분류상 ok지만 seed IoU가 낮아 REPORT.md가 obj 12를 \"어긋난\" seed라 부른 것과 같은 부류입니다. "
             f"IoU < {thr['misaligned_iou']:.2f} 규칙은 이들을 잡지 않습니다.")
    L.append("")
    L.append("| 데이터셋 | 카메라 | obj | 방법 | gt_area | pred_area | IoU | J | F |")
    L.append("|---|---|---:|---|---:|---:|---:|---:|---:|")
    for r in rows:
        if r["cls"] == "ok" and r["iou"] < BORDERLINE_IOU:
            L.append(f"| {r['dataset']} | {r['camera']} | {r['obj']} | {r['method']} | "
                     f"{r['gt_area']} | {r['pred_area']} | {r['iou']:.3f} | "
                     f"{f4(r['J'])} | {f4(r['F'])} |")
    L.append("")

    # --- gt_empty list, compact
    L.append("## gt_empty (참고 — 퇴화 아님)")
    L.append("")
    for m in methods:
        ge = [r for r in rows if r["method"] == m and r["cls"] == "gt_empty"]
        bad = [r for r in ge if r["pred_area"] > 0]
        items = ", ".join(f"{r['dataset']}/{r['camera']}/{r['obj']}" for r in ge)
        L.append(f"- **{m}**: {len(ge)}개 (seed가 0 px가 아닌 것 {len(bad)}개)"
                 f"{' — ' + items if ge else ''}")
    L.append("")

    # --- narrative
    L.append("## 판독 — P5의 기대 이득")
    L.append("")
    L.extend(narrative(out))
    L.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(L))


def fmt_objs(rs):
    """'camera_0004 obj 8/14 tiny; cam16 obj 19 missing' for a list of rows."""
    groups = {}
    for r in rs:
        groups.setdefault((r["camera"], r["cls"]), []).append(r["obj"])
    return "; ".join(f"{cam} obj {'/'.join(str(o) for o in sorted(objs))} {cls}"
                     for (cam, cls), objs in sorted(groups.items())) or "없음"


def prompt_tiny_text(out):
    """The degenerate objects whose reference prompt is under TINY_MIN_PX,
    grouped by (dataset, object) with the cameras and class per method."""
    methods, rows = out["methods"], out["rows"]
    per_obj = {}
    for r in rows:
        if r["cls"] in DEGENERATE and r["ref_gt_area"] < TINY_MIN_PX:
            per_obj.setdefault((r["dataset"], r["obj"]), {}) \
                   .setdefault(r["method"], []).append((r["camera"], r["cls"]))

    def sig_text(sig):
        classes = {c for _, c in sig}
        if len(classes) == 1:
            return "/".join(cam for cam, _ in sig) + " " + classes.pop()
        return ", ".join(f"{cam} {c}" for cam, c in sig)

    items = []
    for (d, obj), per_m in sorted(per_obj.items()):
        px = next(r["ref_gt_area"] for r in rows
                  if r["dataset"] == d and r["obj"] == obj)
        groups = {}
        for m in methods:
            if m in per_m:
                groups.setdefault(tuple(sorted(per_m[m])), []).append(m)
        desc = ", ".join(f"{sig_text(sig)}({'·'.join(f'`{m}`' for m in ms)})"
                         for sig, ms in groups.items())
        items.append(f"{d} obj {obj}(ref {out['datasets'][d]['ref_cam']} GT {px} px; {desc})")
    return items


def narrative(out):
    """Korean reading of the numbers - what the census says about P5.

    Everything below is computed from `out`; the only literal is
    REPORT_P5_QUOTE, printed beside the value it is compared with.
    """
    methods, summary, average = out["methods"], out["summary"], out["average"]
    rows = out["rows"]
    lines = []
    for m in methods:
        a = average[m]
        if a["jf_stored"] is None:
            continue
        per_ds = [(d, s) for d, s in summary[m].items()
                  if s["counts"]["degenerate"] > 0 and s["jf_stored"] is not None]
        per_ds.sort(key=lambda x: -(x[1]["jf_repair"] - x[1]["jf_stored"]))
        where = "; ".join(
            f"{d} {s['counts']['degenerate']}개"
            + (f"(프롬프트 <64 px {s['counts']['degenerate_prompt_tiny']})"
               if s["counts"]["degenerate_prompt_tiny"] else "")
            + f" {s['jf_stored']:.4f} → {s['jf_repair']:.4f} → {s['jf_upper']:.4f}"
            for d, s in per_ds) or "없음"
        lines.append(f"- **{m}**: 퇴화 seed {a['counts']['degenerate']}개 / 도달 가능 객체 "
                     f"{a['n_objects'] - a['counts']['unreachable']}개. "
                     f"복구 목표 {a['jf_stored']:.4f} → {a['jf_repair']:.4f} "
                     f"({sgn(a['jf_repair'] - a['jf_stored'])}), "
                     f"프롬프트 ≥64 px만 {a['jf_repair_promptable']:.4f} "
                     f"({sgn(a['jf_repair_promptable'] - a['jf_stored'])}), "
                     f"상한 {a['jf_upper']:.4f} ({sgn(a['jf_upper'] - a['jf_stored'])}). "
                     f"데이터셋별(저장 → 복구 → 상한): {where}.")
    lines.append("")

    # Welder anchor reading
    base = methods[0]
    by_key = {(r["dataset"], r["camera"], r["obj"], r["method"]): r for r in rows}
    sam3 = [m for m in methods if m != base and has_numbers(summary, m, ANCHOR_DATASET)]
    if has_numbers(summary, base, ANCHOR_DATASET) and sam3:
        m = sam3[-1]
        sb, sm = summary[base][ANCHOR_DATASET], summary[m][ANCHOR_DATASET]
        pb, pm = sb["per_camera"][ANCHOR_CAM], sm["per_camera"][ANCHOR_CAM]
        contrib = (pm["jf"] - pb["jf"]) * pm["n_objects"] / sm["n_scored"]
        r8 = by_key.get((ANCHOR_DATASET, ANCHOR_CAM, ANCHOR_TINY_OBJS[0], m))
        r12 = by_key.get((ANCHOR_DATASET, ANCHOR_CAM, ANCHOR_LOWIOU_OBJ, m))
        r14 = by_key.get((ANCHOR_DATASET, ANCHOR_CAM, ANCHOR_TINY_OBJS[1], m))
        if r8 and r12 and r14 and all(r["J"] is not None for r in (r8, r12, r14)):
            lines.append(
                f"**Welder 앵커 재현.** `{m}`의 {ANCHOR_CAM}에서 obj 8/14는 {r8['pred_area']} px / "
                f"{r14['pred_area']} px seed(GT {r8['gt_area']:,} / {r14['gt_area']:,} px)로 "
                f"{r8['cls']} / {r14['cls']}, "
                f"J {r8['J']:.2f} / {r14['J']:.2f}. obj 12는 {r12['pred_area']:,} px, IoU {r12['iou']:.3f}, "
                f"J {r12['J']:.2f} — REPORT.md는 이를 \"어긋난\" seed라 했고, 이 census의 규칙(IoU < "
                f"{out['thresholds']['misaligned_iou']:.2f})으로는 **{r12['cls']}**입니다"
                + (f" (IoU {r12['iou']:.3f}보다 높은 임계값이면 misaligned로 잡힙니다). "
                   if r12['cls'] == 'ok' else ". ")
                + 
                f"{ANCHOR_CAM}의 J&F는 `{base}` {pb['jf']:.4f} vs `{m}` {pm['jf']:.4f}"
                f"(REPORT.md의 0.861 vs 0.707은 J&F가 아니라 J: {pb['J']:.4f} vs {pm['J']:.4f}), "
                f"데이터셋 격차 {sgn(sm['jf_stored'] - sb['jf_stored'])} 중 이 카메라의 기여가 "
                f"{sgn(contrib)} — \"Welder 손실 전부가 camera_0004\"는 "
                + ("성립합니다" if contrib <= sm['jf_stored'] - sb['jf_stored'] < 0 else "성립하지 않습니다")
                + "".join(f" ({c}에서는 오히려 `{m}`가 앞섭니다)"
                          for c, pc in sorted(sm["per_camera"].items())
                          if c != ANCHOR_CAM and pc["jf"] > sb["per_camera"][c]["jf"])
                + ".")
            lines.append("")

    # The verdict.  Every number, threshold and (camera, object) list is
    # computed from `out` so the paragraph cannot drift from the tables when
    # the method set or thresholds change; REPORT_P5_QUOTE is the one literal
    # and is printed beside the computed value it is compared with.
    scored_methods = [m for m in methods if average[m]["jf_stored"] is not None]
    if not scored_methods:
        return lines

    def contributions(m):
        """(dataset, headline gain from repairing its promptable seeds), desc."""
        n = average[m]["n_datasets"]
        c = [(d, (s["jf_repair_promptable"] - s["jf_stored"]) / n)
             for d, s in summary[m].items() if has_numbers(summary, m, d)]
        return sorted(c, key=lambda x: -x[1])

    parts = []
    for m in scored_methods:
        a = average[m]
        n = a["n_datasets"]
        reach = a["n_objects"] - a["counts"]["unreachable"]
        zero = sum(1 for d in summary[m] if has_numbers(summary, m, d)
                   and summary[m][d]["counts"]["degenerate"] == 0)
        gain = a["jf_repair"] - a["jf_stored"]
        gain_p = a["jf_repair_promptable"] - a["jf_stored"]
        top = [(d, c) for d, c in contributions(m) if c > 0][:3]
        top_txt = ", ".join(f"{d} {sgn(c)}" for d, c in top) or "없음"
        parts.append(
            f"`{m}`: 퇴화 {a['counts']['degenerate']}개는 도달 가능 객체 {reach}개의 "
            f"{100 * a['counts']['degenerate'] / reach:.1f}%, 데이터셋 {n}개 중 {zero}개는 0개. "
            f"복구 목표 이득 {sgn(gain)} 중 {sgn(gain - gain_p)}는 ref 프롬프트 <{TINY_MIN_PX} px "
            f"객체의 몫이라 P5가 닿을 수 있는 이득은 {sgn(gain_p)}, 기여 상위: {top_txt}.")

    # Objects whose reference prompt is too small to have a donor view.
    tiny_items = prompt_tiny_text(out)
    if tiny_items:
        parts.append(
            f"ref 프롬프트 <{TINY_MIN_PX} px 객체: " + ", ".join(tiny_items) + ". "
            "이들은 tracker 입력 해상도에서 프롬프트 자체가 사라져 donor 뷰가 있을 수 없으므로 "
            "P5의 재전파가 아니라 프롬프트 해상도(또는 라벨 자체)의 문제입니다.")
    else:
        parts.append(f"ref 프롬프트 <{TINY_MIN_PX} px인 퇴화 객체는 없습니다.")

    # Where the reachable gain of the last method comes from, item by item.
    m_top = scored_methods[-1]
    top2 = [(d, c) for d, c in contributions(m_top) if c > 0][:2]
    items = []
    for rank, (d, c) in enumerate(top2):
        s = summary[m_top][d]
        deg = [r for r in rows if r["method"] == m_top and r["dataset"] == d
               and r["cls"] in DEGENERATE and r["ref_gt_area"] >= TINY_MIN_PX]
        t = (f"{'첫째' if rank == 0 else '둘째'}: {d} {fmt_objs(deg)} — "
             f"데이터셋 {sgn(s['jf_repair_promptable'] - s['jf_stored'])} = 헤드라인 {sgn(c)}")
        if m_top != base and has_numbers(summary, base, d):
            gap = s["jf_stored"] - summary[base][d]["jf_stored"]
            after = s["jf_repair_promptable"] - summary[base][d]["jf_stored"]
            t += (f"; `{base}` 대비 {d} 격차 {sgn(gap)} → 복구 후 {sgn(after)}"
                  + (" (소거)" if after >= 0 else " (축소)"))
        if d == ANCHOR_DATASET:
            t += f"; REPORT.md P5의 \"{REPORT_P5_QUOTE}\"와 비교"
        items.append(t)
    if items:
        parts.append(f"`{m_top}`에서 P5가 닿는 이득의 출처. " + ". ".join(items) + ".")

    # The first method against the last: same failure mode, overlap, ranking.
    if m_top != base:
        cb = average[base]["counts"]
        spelled = ", ".join(f"{c} {cb[c]}" for c in DEGENERATE if cb[c]) or "없음"
        t = f"`{base}`도 같은 종류의 seed 실패를 {cb['degenerate']}개({spelled}) 갖고 있고"
        ov = next((o for o in out["empty_missing_overlap"] if o["other"] == m_top), None)
        if ov:
            t += (f", 그중 empty+missing {ov['n_base']}개 가운데 {ov['n_overlap']}개는 "
                  f"`{m_top}`의 missing+empty {ov['n_other']}개와 같은 (카메라, 객체)입니다")
        order_s = sorted(scored_methods, key=lambda m: -average[m]["jf_stored"])
        order_r = sorted(scored_methods, key=lambda m: -average[m]["jf_repair_promptable"])
        rank_s = " > ".join(f"`{m}`" for m in order_s)
        if order_s == order_r:
            t += (f". P5를 `{base}` 러너에도 적용하면 두 방법이 함께 오르고 저장값 순위({rank_s})는 "
                  "복구 목표(프롬프트 ≥64 px)에서도 그대로이므로, P5는 순위를 바꾸는 항목이 아니라 "
                  "두 방법의 바닥을 올리는 항목입니다.")
        else:
            rank_r = " > ".join(f"`{m}`" for m in order_r)
            t += (f". 복구 목표(프롬프트 ≥64 px)에서는 저장값 순위 {rank_s}가 {rank_r}로 바뀝니다.")
        parts.append(t)

    # Sensitivity to the misaligned threshold.
    cur = out["thresholds"]["misaligned_iou"]
    cur_key = f"{cur:.2f}"
    others = [t for t in out["sensitivity"] if t != cur_key]
    if others:
        hi = max(others, key=float)
        deltas = ", ".join(f"`{m}` {out['sensitivity'][cur_key][m]['degenerate']} → "
                           f"{out['sensitivity'][hi][m]['degenerate']}" for m in scored_methods)
        r12 = by_key.get((ANCHOR_DATASET, ANCHOR_CAM, ANCHOR_LOWIOU_OBJ, m_top))
        ex = (f"({ANCHOR_DATASET} {ANCHOR_CAM} obj {ANCHOR_LOWIOU_OBJ}, IoU {r12['iou']:.3f} 부류)"
              if r12 and r12["cls"] == "ok" and r12["iou"] < float(hi) else "")
        parts.append(
            f"misaligned 임계값을 {cur_key}에서 {'~'.join(others)}(으)로 올리면 퇴화 수가 늘고"
            f"({deltas}) 복구 목표도 위 민감도 표만큼 오르지만{ex}, 그 seed들은 정의상 tiny가 "
            "아니라 면적 검사로는 잡히지 않아 P5의 감지기가 IoU 계열 신호(donor 마스크와의 IoU)를 "
            "함께 봐야 합니다.")

    # Unreachable: identical across methods unless a folder is absent.
    unr = {m: average[m]["counts"]["unreachable"] for m in scored_methods}
    total = average[base]["n_objects"]
    if len(set(unr.values())) == 1:
        parts.append(f"unreachable은 {total}개 scored 객체 중 {unr[base]}개로 모든 방법에 동일하며 "
                     "P5가 아니라 P8(reference 규칙)의 몫입니다.")
    else:
        parts.append("unreachable은 " + ", ".join(f"`{m}` {v}개" for m, v in unr.items())
                     + "로 P5가 아니라 P8(reference 규칙)의 몫입니다.")
    lines.append("**P5의 기대 이득.** " + " ".join(parts))
    return lines


def print_anchors(out):
    methods = out["methods"]
    by_key = {(r["dataset"], r["camera"], r["obj"], r["method"]): r for r in out["rows"]}
    print(f"\nREPORT.md anchors ({ANCHOR_DATASET} {ANCHOR_CAM}):")
    print(f"{'method':<22s}{'obj':>4s}  {'cls':<11s}{'gt_px':>8s}{'pred_px':>8s}"
          f"{'IoU':>7s}{'J':>8s}{'F':>8s}")
    for m in methods:
        for obj in ANCHOR_OBJS:
            r = by_key.get((ANCHOR_DATASET, ANCHOR_CAM, obj, m))
            if r is None:
                continue
            print(f"{m:<22s}{obj:>4d}  {r['cls']:<11s}{r['gt_area']:>8d}{r['pred_area']:>8d}"
                  f"{r['iou']:>7.3f}{f4(r['J']):>8s}{f4(r['F']):>8s}")
    if ANCHOR_DATASET in out["summary"][methods[0]]:
        print(f"\n{ANCHOR_DATASET} per-camera J/F/J&F (stored), degenerate count in ():")
        for m in methods:
            if not has_numbers(out["summary"], m, ANCHOR_DATASET):
                print(f"  {m:<22s}(no folder for {ANCHOR_DATASET})")
                continue
            s = out["summary"][m][ANCHOR_DATASET]
            cams = "  ".join(f"{c}={pc['J']:.4f}/{pc['F']:.4f}/{pc['jf']:.4f}({pc['degenerate']})"
                             for c, pc in sorted(s["per_camera"].items()))
            print(f"  {m:<22s}{cams}  dataset={s['jf_stored']:.4f}")


def print_summary(out):
    for m in out["methods"]:
        print(f"\n{m}")
        print(f"{'dataset':<22s}{'n':>4s}{'unr':>5s}{'mis':>5s}{'emp':>5s}{'tiny':>5s}"
              f"{'msal':>5s}{'gte':>5s}{'ok':>5s}{'stored':>9s}{'repair':>9s}"
              f"{'rep>=64':>9s}{'upper':>9s}")
        for d in sorted(out["summary"][m]):
            s = out["summary"][m][d]
            c = s["counts"]
            print(f"{d:<22s}{s['n_objects']:>4d}{c['unreachable']:>5d}{c['missing']:>5d}"
                  f"{c['empty']:>5d}{c['tiny']:>5d}{c['misaligned']:>5d}{c['gt_empty']:>5d}"
                  f"{c['ok']:>5d}{f4(s['jf_stored']):>9s}{f4(s['jf_repair']):>9s}"
                  f"{f4(s['jf_repair_promptable']):>9s}{f4(s['jf_upper']):>9s}")
        a = out["average"][m]
        c = a["counts"]
        print(f"{'AVERAGE(' + str(a['n_datasets']) + ')':<22s}{a['n_objects']:>4d}"
              f"{c['unreachable']:>5d}{c['missing']:>5d}{c['empty']:>5d}{c['tiny']:>5d}"
              f"{c['misaligned']:>5d}{c['gt_empty']:>5d}{c['ok']:>5d}"
              f"{f4(a['jf_stored']):>9s}{f4(a['jf_repair']):>9s}"
              f"{f4(a['jf_repair_promptable']):>9s}{f4(a['jf_upper']):>9s}")
    print("\nmisaligned-IoU sensitivity (degenerate / avg repair / avg upper):")
    for t, per_m in out["sensitivity"].items():
        print("  " + t + "  " + "  ".join(
            f"{m}={s['degenerate']}/{f4(s['jf_repair'])}/{f4(s['jf_upper'])}"
            for m, s in per_m.items()))


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                    help="result folder names to census")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="MVSeg data root")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="MVSeg.json the runner used (cameras, start frame)")
    ap.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    ap.add_argument("--out-md", default=DEFAULT_OUT_MD)
    ap.add_argument("--misaligned-iou", type=float, default=DEFAULT_MISALIGNED_IOU,
                    help="seed IoU below which a non-tiny seed is misaligned")
    ap.add_argument("--jobs", type=int, default=0,
                    help="worker processes (default: cpu count)")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    config = json.load(open(args.config))

    # The 15 main datasets: config entries whose folder has Mask/<cam>/ for
    # every camera in cam_list.  The COLMAP/SA3D variants keep a flat Mask/.
    datasets, jobs = {}, []
    for name, d in config.items():
        ds_dir = os.path.join(root, d.get("folder", name))
        cams = [cam_name(c, d["prefix"], d["prefix1"]) for c in d["cam_list"]]
        if not all(os.path.isdir(os.path.join(ds_dir, "Mask", c)) for c in cams):
            continue
        folder = os.path.basename(ds_dir)
        meta = dataset_meta(folder, ds_dir, config)
        scored = sorted(x for x in os.listdir(os.path.join(ds_dir, "Mask"))
                        if os.path.isdir(os.path.join(ds_dir, "Mask", x)))
        if sorted(cams) != scored:
            print(f"warning: {folder}: Mask/ cameras {scored} != cam_list {cams}",
                  file=sys.stderr)
        ref = meta["ref"]["maxid"]
        ref_gt = read_gray(os.path.join(ds_dir, "Mask", ref["cam"],
                                        f"{meta['start_frame']:06d}.png"))
        ids, counts = np.unique(ref_gt, return_counts=True)
        ref_areas = {int(i): int(n) for i, n in zip(ids, counts) if i != 0}
        assert sorted(ref_areas) == list(ref["seed_ids"]), folder
        datasets[folder] = {
            "start_frame": meta["start_frame"],
            "cameras": scored,
            "ref_cam": ref["cam"],
            "ref_max_id": ref["max_id"],
            "ref_seed_ids": ref["seed_ids"],
            "ref_seed_area": ref_areas,
            "view_index": meta["view_index"],
        }
        for cam in scored:
            jobs.append((root, folder, cam, meta["start_frame"], ref_areas,
                         args.methods))
    if not datasets:
        sys.exit(f"no config dataset under {root} with Mask/<cam>/ folders")
    print(f"{len(datasets)} datasets, {len(jobs)} cameras, methods {args.methods}",
          flush=True)

    scores, jf_files = load_scores(root, args.methods)

    rows, objects = [], {}
    with Pool(min(args.jobs or os.cpu_count(), len(jobs))) as pool:
        for i, (d, cam, obj_ids, cam_rows) in enumerate(
                pool.imap_unordered(census_camera, jobs), 1):
            objects[(d, cam)] = obj_ids
            rows.extend(cam_rows)
            print(f"[{i}/{len(jobs)}] {d}/{cam} ({len(obj_ids)} objects)", flush=True)
    for (d, cam), ids in sorted(objects.items()):
        datasets[d].setdefault("objects", {})[cam] = ids

    unscored = 0
    for r in rows:
        jf = scores.get((r["dataset"], r["camera"], r["method"], r["obj"]))
        r["J"], r["F"] = (jf if jf else (None, None))
        unscored += jf is None and not r["no_folder"]
        r["is_ref_cam"] = r["camera"] == datasets[r["dataset"]]["ref_cam"]
        r["cls"] = classify(r, args.misaligned_iou)
    if unscored:
        print(f"warning: {unscored} rows have no stored J/F", file=sys.stderr)
    rows.sort(key=lambda r: (r["dataset"], r["camera"], r["obj"],
                             args.methods.index(r["method"])))

    # A method folder absent for a dataset: warn, class no_folder, skip.
    no_folder = sorted({(r["dataset"], r["method"]) for r in rows if r["no_folder"]})
    for d, m in no_folder:
        n = sum(1 for r in rows if r["dataset"] == d and r["method"] == m)
        print(f"warning: {d}/{m}: folder absent, {n} objects classed no_folder "
              f"(not degenerate, excluded from the averages)", file=sys.stderr)

    ds_names = sorted(datasets)
    summary, average = summarize(rows, ds_names, args.methods)
    ious = sorted(set(SENSITIVITY_IOUS) | {args.misaligned_iou})
    sens = sensitivity(rows, ds_names, args.methods, ious)
    out = {
        "schema": SCHEMA_VERSION,
        "root": root,
        "config": os.path.abspath(args.config),
        "methods": args.methods,
        "jf_files": jf_files,
        "thresholds": {"tiny_min_px": TINY_MIN_PX, "tiny_frac": TINY_FRAC,
                       "misaligned_iou": args.misaligned_iou},
        "classes": CLASSES,
        "degenerate_classes": list(DEGENERATE),
        "absent_seed_classes": list(ABSENT_SEED),
        "repair_rule": ("degenerate object -> max(stored (J+F)/2, mean of median J and "
                        "median F of the dataset-method's ok objects)"),
        "no_folder": [list(x) for x in no_folder],
        "datasets": datasets,
        "rows": rows,
        "summary": summary,
        "average": average,
        "sensitivity": sens,
        "empty_missing_overlap": absent_seed_overlap(rows, args.methods),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(out, fh, indent=1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
    write_md(args.out_md, out)

    print_summary(out)
    print_anchors(out)
    print(f"\nwrote {args.out_json}\nwrote {args.out_md}")


if __name__ == "__main__":
    main()
