#!/usr/bin/env python3
"""Build the S1-E7 seed sets (docs/stage1-E7-prereg.md section 2).

  --set gt      MVSeed_gt      : frame-0 ground truth of the three annotated cameras (c_ini
                                 included) for the basic objects; an object absent from that
                                 camera's frame-0 GT gets an empty PNG (area 0 = "not here").
  --set oracle  MVSeed_e6_oracle: G0 folder copied whole, then every GT-present non-reference
                                 row of the frozen F0.json + N0.json replaced by the E6 stream
                                 (seed_vote.STREAMS, same mask_path) with the highest GT IoU;
                                 ties and all-zero keep G0.  c_ini is never touched.

Writes SEED_MANIFEST.json (written_views, dataset, start_frame) so the runner's coverage
check applies, and SEED_BUILD.json with what was written.  Needs cv2 -> run in the
scsam3 container:

    docker run --rm --user $(id -u):$(id -g) -v /:/host -w /host$PWD scsam3 \
        python MVSeed/build_seed_sets.py --set gt --set oracle
"""
import argparse, datetime, json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seed_vote as SV                      # REPO/DATA/CACHE/STREAMS/mask_path

REPO, DATA = SV.REPO, SV.DATA
CFG = json.load(open(SV.CONFIG, encoding="utf-8"))
CENSUS = json.load(open(SV.CENSUS))


def annotated_cams(scene):
    return sorted(d for d in os.listdir(os.path.join(DATA, CFG[scene]["folder"], "Mask")) if not d.endswith(".json"))


def gt_mask(cv2, scene, cam, t):
    c = CFG[scene]
    m = cv2.imread(os.path.join(DATA, c["folder"], "Mask", cam, f"{c['start_frame']+t:06d}.png"), cv2.IMREAD_GRAYSCALE)
    assert m is not None, (scene, cam, t)
    return m


def write_png(cv2, path, mask):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, (mask.astype("uint8") * 255))


def manifest(scene, tag, written, extra):
    c = CFG[scene]
    return dict(schema="MVSeed build_seed_sets", schema_version=1, set=tag, prereg="docs/stage1-E7-prereg.md",
                dataset=scene, start_frame=c["start_frame"], c_ini=CENSUS[scene]["c_ini"],
                written_views=sorted(written), generated_at=datetime.datetime.now().isoformat(timespec="seconds"), **extra)


def build_gt(cv2, scene, log):
    c = CFG[scene]; start = c["start_frame"]
    dst = os.path.join(DATA, c["folder"], "MVSeed_gt")
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    basic = sorted(int(v) for v in set(CENSUS[scene]["seed_ids_c_ini"]))
    n_written = n_empty = 0
    for cam in annotated_cams(scene):
        g = gt_mask(cv2, scene, cam, 0)
        for o in basic:
            m = g == o
            write_png(cv2, os.path.join(dst, cam, str(start), f"{o}.png"), m)
            n_written += 1; n_empty += int(not m.any())
    man = manifest(scene, "gt", annotated_cams(scene), dict(objects=basic, n_png=n_written, n_empty=n_empty))
    json.dump(man, open(os.path.join(dst, "SEED_MANIFEST.json"), "w"), indent=1)
    log(f"  {scene:20s} gt: {len(annotated_cams(scene))} cams x {len(basic)} objects = {n_written} PNG ({n_empty} empty)")
    return n_written, n_empty


def build_oracle(cv2, np, scene, rows, log):
    c = CFG[scene]; start, folder = c["start_frame"], c["folder"]
    src = os.path.join(DATA, folder, SV.BASE); dst = os.path.join(DATA, folder, "MVSeed_e6_oracle")
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    for f in ("SEED_MANIFEST.json", "SEED_VOTE.json"):
        p = os.path.join(dst, f)
        if os.path.exists(p): os.remove(p)
    c_ini = CENSUS[scene]["c_ini"]
    choices, by_stream, cams = [], {}, set()
    for r in rows:
        if r["scene"] != scene: continue
        cam, obj = r["cam"], r["obj"]
        assert cam != c_ini, ("F0/N0 must be non-reference", r)
        g = gt_mask(cv2, scene, cam, 0) == obj
        best, best_iou = "G0", -1.0
        ious = {}
        for name, kind, fname in SV.STREAMS:
            m = cv2.imread(SV.mask_path(kind, fname, scene, folder, cam, start, obj), cv2.IMREAD_GRAYSCALE)
            mb = (m > 127) if m is not None else np.zeros_like(g)
            v = SV.iou(mb, g); ious[name] = round(v, 4)
            if v > best_iou + 1e-9:          # strict: earlier stream (G0 first) wins ties
                best, best_iou = name, v
        cams.add(cam)
        choices.append(dict(cam=cam, obj=obj, gt_px=int(g.sum()), best=best, iou=ious))
        if best != "G0":
            kind, fname = {n: (k, f) for n, k, f in SV.STREAMS}[best]
            m = cv2.imread(SV.mask_path(kind, fname, scene, folder, cam, start, obj), cv2.IMREAD_GRAYSCALE)
            write_png(cv2, os.path.join(dst, cam, str(start), f"{obj}.png"), m > 127)
            by_stream[best] = by_stream.get(best, 0) + 1
    # written_views must cover every scored camera (seeds_from.check_coverage), c_ini included:
    # the folder is a whole G0 copy, so list every camera directory it holds
    written = sorted(d for d in os.listdir(dst) if os.path.isdir(os.path.join(dst, d)))
    man = manifest(scene, "oracle", written, dict(base=SV.BASE, streams=SV.NAMES, n_rows=len(choices),
                                                   n_replaced=sum(by_stream.values()), by_stream=by_stream))
    json.dump(man, open(os.path.join(dst, "SEED_MANIFEST.json"), "w"), indent=1)
    json.dump(dict(choices=choices), open(os.path.join(dst, "SEED_BUILD.json"), "w"), indent=1)
    log(f"  {scene:20s} oracle: {len(choices)} rows, replaced {sum(by_stream.values())} {by_stream}")
    return choices


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", choices=("gt", "oracle"), required=True)
    ap.add_argument("--scenes", default="")
    a = ap.parse_args()
    import cv2, numpy as np
    scenes = a.scenes.split(",") if a.scenes else sorted(CENSUS)
    log = lambda m: print(m, flush=True)
    summary = {}
    if "gt" in a.set:
        tot = [build_gt(cv2, scene, log) for scene in scenes]
        summary["gt"] = dict(n_png=sum(t[0] for t in tot), n_empty=sum(t[1] for t in tot))
    if "oracle" in a.set:
        rows = json.load(open(os.path.join(REPO, "MVSeed", "runs", "F0.json")))["pairs"] + \
               json.load(open(os.path.join(REPO, "MVSeed", "runs", "N0.json")))["pairs"]
        allc = []
        for scene in scenes:
            allc += build_oracle(cv2, np, scene, rows, log)
        by = {}
        for ch in allc: by[ch["best"]] = by.get(ch["best"], 0) + 1
        summary["oracle"] = dict(n_rows=len(allc), best_by_stream=by)
    log(json.dumps(summary))
    json.dump(summary, open(os.path.join(REPO, "MVSeed", "runs", "e7_seed_sets.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
