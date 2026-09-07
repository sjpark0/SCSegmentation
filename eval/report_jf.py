#!/usr/bin/env python3
"""Aggregate an eval_jf.py raw file into per-dataset and overall J&F tables.

    python3 report_jf.py                                  the SAM 2 results
    python3 report_jf.py --raw jf_raw.json jf_raw_sam3.json   two runs side by side
    python3 report_jf.py --methods SegMask1 SegMaskNew1   a subset, in that order

Two aggregations are printed for each metric:

  as-is        every object in the ground truth counts; a method that exported
               no mask for an object is scored on an empty prediction, which is
               what the DAVIS protocol does and what an incomplete result costs.
  exported     object/camera pairs for which the method produced no file at all
               are dropped, so the number reflects only the objects the method
               actually segmented.

Frames: "all" uses every frame, "inner" drops the first and last frame of each
camera the way the DAVIS evaluation code does.
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def find_absent(raw):
    """(dataset|camera|method) -> object ids with no exported mask in any frame.

    Read off the directory listings, so it stays correct for any method set
    without a separate bookkeeping file.
    """
    absent = {}
    for e in raw:
        if e.get("result") is None:
            continue
        d, cam, m = e["dataset"], e["camera"], e["method"]
        mdir = os.path.join(ROOT, d, m, cam)
        seen = set()
        if os.path.isdir(mdir):
            for fr in os.listdir(mdir):
                for f in os.listdir(os.path.join(mdir, fr)):
                    stem = os.path.splitext(f)[0]
                    if stem.isdigit():
                        seen.add(int(stem))
        gone = [o for o in e["objects"] if o not in seen]
        if gone:
            absent[f"{d}|{cam}|{m}"] = set(gone)
    return absent


def collect(raw, absent, methods, variant, drop_absent):
    """(dataset, method) -> (list of J, list of F) over (camera, object) pairs."""
    acc, datasets = {}, []
    for e in raw:
        if e.get("result") is None or e["method"] not in methods:
            continue
        d, cam, m = e["dataset"], e["camera"], e["method"]
        skip = absent.get(f"{d}|{cam}|{m}", set()) if drop_absent else set()
        js, fs = acc.setdefault((d, m), ([], []))
        for i, obj in enumerate(e["objects"]):
            if obj in skip:
                continue
            js.append(e["result"][f"J_{variant}"][i])
            fs.append(e["result"][f"F_{variant}"][i])
        if d not in datasets:
            datasets.append(d)
    return acc, sorted(datasets)


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def table(title, acc, datasets, methods, metric):
    w = max(len(d) for d in datasets) + 2
    pick = {"J": lambda j, f: mean(j),
            "F": lambda j, f: mean(f),
            "J&F": lambda j, f: (mean(j) + mean(f)) / 2}[metric]
    print(title)
    print(" " * w + "".join(f"{m:>13s}" for m in methods))
    print("-" * (w + 13 * len(methods)))
    for d in datasets:
        row = "".join(f"{pick(*acc[(d, m)]):>13.4f}" if (d, m) in acc
                      else f"{'-':>13s}" for m in methods)
        print(f"{d:<{w}s}{row}")
    print("-" * (w + 13 * len(methods)))
    avg = "".join(
        f"{mean([pick(*acc[(d, m)]) for d in datasets if (d, m) in acc]):>13.4f}"
        for m in methods)
    print(f"{'AVERAGE':<{w}s}{avg}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", nargs="+", default=["jf_raw.json"],
                    help="one or more eval_jf.py outputs; entries are pooled")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="result folders to show, in this order (default: all in the file)")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="restrict to these datasets")
    ap.add_argument("--common", action="store_true",
                    help="restrict to the datasets every selected method covers, "
                         "so the averages compare like with like")
    ap.add_argument("--no-missing", action="store_true",
                    help="skip the list of objects with no exported mask")
    args = ap.parse_args()

    raw = []
    for name in args.raw:
        raw.extend(json.load(open(os.path.join(ROOT, name))))
    methods = args.methods or sorted({e["method"] for e in raw})
    if args.datasets:
        raw = [e for e in raw if e["dataset"] in args.datasets]
    if args.common:
        have = {m: {e["dataset"] for e in raw
                    if e["method"] == m and e.get("result") is not None}
                for m in methods}
        keep = set.intersection(*have.values()) if have else set()
        dropped = sorted(set.union(*have.values()) - keep) if have else []
        raw = [e for e in raw if e["dataset"] in keep]
        print(f"common subset: {len(keep)} datasets"
              + (f"   (dropped: {', '.join(dropped)})" if dropped else ""))
        print()
    absent = find_absent(raw)

    for drop, label in (
            (False, "as-is  (missing exports scored as empty predictions)"),
            (True, "exported only  (objects with no output file dropped)")):
        acc, datasets = collect(raw, absent, methods, "all", drop)
        if not datasets:
            continue
        print("=" * 100)
        print(f"{label}   -   all frames, {len(datasets)} datasets")
        print("=" * 100)
        for metric in ("J&F", "J", "F"):
            table(metric, acc, datasets, methods, metric)

    print("=" * 100)
    print("AVERAGE row only, for every combination")
    print("=" * 100)
    print(f"{'':<38s}" + "".join(f"{m:>13s}" for m in methods))
    for variant, vlabel in (("all", "all frames"),
                            ("inner", "DAVIS (drop first/last)")):
        for drop, dlabel in ((False, "as-is"), (True, "exported only")):
            acc, datasets = collect(raw, absent, methods, variant, drop)
            row = "".join(
                f"{mean([(mean(acc[(d, m)][0]) + mean(acc[(d, m)][1])) / 2 for d in datasets if (d, m) in acc]):>13.4f}"
                for m in methods)
            print(f"J&F  {vlabel:<24s} {dlabel:<12s}{row}")

    if not args.no_missing and absent:
        print()
        print("objects with no exported mask at all (camera, object) - only these "
              "differ between the two aggregations:")
        for k, v in sorted(absent.items()):
            d, cam, m = k.split("|")
            if m in methods:
                print(f"  {d}/{cam}/{m}: {sorted(v)}")


if __name__ == "__main__":
    main()
