"""Markdown report over the spec's axes (2.4) from one or more scores/<method>.json files.

    .venv/bin/python -m mvseg2d.report scores/*.json --out scores/report.md

Headline aggregation (spec 2.3): pair -> view mean -> scene mean -> overall mean.  Scenes whose
status is "error" or "missing" contribute no rows; they are listed under the headline table.
"""
import argparse
import json
import os
from collections import defaultdict

from .io import SIZE_BINS

VIEW_DIST_BINS = ("1", "2-3", ">=4")


def view_dist_bin(d):
    if d <= 1:
        return VIEW_DIST_BINS[0]
    if d <= 3:
        return VIEW_DIST_BINS[1]
    return VIEW_DIST_BINS[2]


def flatten(score):
    rows = []
    for s in score["scenes"].values():
        if s["status"] != "ok":
            continue
        for r in s["rows"]:
            rows.append(dict(r, method=score["method"]))
    return rows


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _hier_mean(rows, key):
    """Mean of `key` over pairs -> views -> scenes (each level an unweighted mean)."""
    by_scene = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_scene[r["scene"]][r["cam"]].append(r[key])
    scene_means = [_mean([_mean(v) for v in views.values()]) for views in by_scene.values()]
    return _mean(scene_means), len(by_scene)


def headline(rows):
    out = {}
    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    for m, rs in sorted(by_method.items()):
        J, n = _hier_mean(rs, "J")
        F, _ = _hier_mean(rs, "F")
        out[m] = {"J": J, "F": F, "JF": (J + F) / 2, "n_scenes": n, "n_pairs": len(rs)}
    return out


def aggregate(rows, by):
    """Plain pair means of J/F grouped by the tuple of fields `by` (always including method)."""
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in by)].append(r)
    return {k: {"n": len(v), "J": _mean([r["J"] for r in v]), "F": _mean([r["F"] for r in v]),
                "JF": _mean([(r["J"] + r["F"]) / 2 for r in v])} for k, v in g.items()}


def _f(x):
    return "-" if x != x else f"{x:.3f}"


def _table(header, lines):
    return "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n" + \
        "".join("| " + " | ".join(str(c) for c in line) + " |\n" for line in lines) + "\n"


def _axis_table(rows, methods, field, values, title):
    agg = aggregate(rows, ("method", field))
    lines = []
    for v in values:
        line = [v]
        for m in methods:
            a = agg.get((m, v))
            line.append(f"{_f(a['JF'])} (n={a['n']})" if a else "-")
        lines.append(line)
    return f"## {title}\n\n" + _table([field] + methods, lines)


def tables(scores):
    rows = [r for s in scores for r in flatten(s)]
    methods = sorted({s["method"] for s in scores})
    hl = headline(rows)
    md = ["# MVSeg2D report\n"]

    # headline + runtime
    lines = []
    for m in methods:
        secs, gpus, halluc = [], [], 0
        for s in scores:
            if s["method"] != m:
                continue
            for sc in s["scenes"].values():
                halluc += sc.get("hallucinations", 0)
                run = sc.get("run") or {}
                if sc["status"] == "ok" and run.get("seconds") is not None:
                    secs.append(run["seconds"])
                if sc["status"] == "ok" and run.get("gpu_mb") is not None:
                    gpus.append(run["gpu_mb"])
        h = hl.get(m, {"J": float("nan"), "F": float("nan"), "JF": float("nan"), "n_scenes": 0, "n_pairs": 0})
        lines.append([m, _f(h["J"]), _f(h["F"]), _f(h["JF"]), h["n_scenes"], h["n_pairs"], halluc,
                      f"{sum(secs):.0f}" if secs else "-", f"{max(gpus):.0f}" if gpus else "-"])
    md.append("## Headline (scene-mean J&F over the scored cameras)\n\n" +
              _table(["method", "J", "F", "J&F", "scenes", "pairs", "halluc.", "total s", "peak GPU MB"], lines))

    def _last_line(err):
        lines = [ln for ln in (err or "").strip().splitlines() if ln.strip()]
        return lines[-1].strip() if lines else ""
    failed = [(s["method"], sc["scene"], sc["status"], _last_line(sc.get("error")))
              for s in scores for sc in s["scenes"].values() if sc["status"] != "ok"]
    if failed:
        md.append("**실패/누락 장면** (평균에서 제외):\n\n" +
                  "".join(f"- {m} / {sc}: {st} {err}\n" for m, sc, st, err in failed) + "\n")

    # per scene
    agg = aggregate(rows, ("method", "scene"))
    scenes = sorted({r["scene"] for r in rows})
    lines = [[sc] + [_f(agg[(m, sc)]["JF"]) if (m, sc) in agg else "-" for m in methods] for sc in scenes]
    md.append("## Per scene (pair-mean J&F)\n\n" + _table(["scene"] + methods, lines))

    md.append(_axis_table(rows, methods, "size_bin", SIZE_BINS, "Object size (target GT px), pair-mean J&F"))
    md.append(_axis_table(rows, methods, "rig", ("pinhole", "hemisphere"), "Rig, pair-mean J&F"))
    for r in rows:
        r["view_dist_bin"] = view_dist_bin(r["view_dist"])
    md.append(_axis_table(rows, methods, "view_dist_bin", VIEW_DIST_BINS,
                          "View distance from the reference camera (camera-index distance; not spatial on hemisphere rigs), pair-mean J&F"))
    return "".join(md)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scores", nargs="+")
    ap.add_argument("--out", default="scores/report.md")
    a = ap.parse_args(argv)
    scores = []
    for p in a.scores:
        with open(p, encoding="utf-8") as f:
            scores.append(json.load(f))
    md = tables(scores)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(md.split("## Per scene")[0])


if __name__ == "__main__":
    main()
