"""T6: eval/report_jf.py nb-bin cluster CI columns (host, standard library only).

The paired nb table gains "cluster CI lo"/"cluster CI hi" (dataset-cluster bootstrap of
the bin's mean delta J&F, same boot_ci/seed/scheme as the paired camera-level row) and
its statement line names the bootstrap scheme; every existing number is unchanged, and
`--paper` differs from docs/raw/paper_tables.md only by those columns in section 4.
"""
import math
import os
import re
import subprocess
import sys

import pytest

from conftest import REPO, load_report_jf

RAW = os.path.join(REPO, "Data", "MVSeg", "jf_v2.json")
PAPER = os.path.join(REPO, "docs", "raw", "paper_tables.md")
A, B = "SegMaskSam3OneStage", "SegMaskSam3MVOpt"
NEW_COLS = ["cluster CI lo", "cluster CI hi"]
RAW_XW = os.path.join(REPO, "Data", "MVSeg", "jf_xw.json")
XW0, XW4 = "SegMaskSam3XW0", "SegMaskSam3XW4"


@pytest.fixture(scope="module")
def rj():
    if not os.path.isfile(RAW):
        pytest.skip(f"{RAW} not present")
    mod = load_report_jf()
    store = mod.Store(mod.load([RAW]))
    for m in (A, B):
        if m not in store.methods:
            pytest.skip(f"{m} not in {RAW}")
    return mod, store


def test_nb_table_cluster_ci(rj):
    mod, store = rj
    cfg = mod.Cfg()
    methods = [A, B]
    plain = mod.nb_table(store, methods, store.datasets, cfg)
    paired = mod.nb_table(store, methods, store.datasets, cfg, pair=(A, B))
    assert plain.columns == ["cameras", A, B]
    assert paired.columns == plain.columns + ["delta J&F", "delta J", "delta F", "wins", "ties",
                                              "losses"] + NEW_COLS
    labels = [lab for lab, _ in paired.rows]
    assert labels == [lab for lab, _ in plain.rows] == ["nb=0", "nb=1", "nb=2", "nb=3", "nb>=4"]
    for (_, pc), (_, cells) in zip(plain.rows, paired.rows):
        assert cells[:3] == pc                                  # columns without pair unchanged
        d = dict(zip(paired.columns, cells))
        lo, hi, mean = d["cluster CI lo"], d["cluster CI hi"], d["delta J&F"]
        assert d["cameras"] > 0 and not math.isnan(mean)
        assert lo <= mean <= hi
        assert d["wins"] + d["ties"] + d["losses"] == d["cameras"]
    # same construction as the paired camera-level row: recompute by hand, bit for bit
    _, cs = mod.score_all(store, methods, store.datasets, cfg)
    bins = {}
    for d in store.datasets:
        for cam in mod.select_cams(store, d, cfg):
            if (d, cam) in store.gt:
                bins.setdefault(min(store.view_index(d, cam), cfg.window), []).append((d, cam))
    for lab, cells in paired.rows:
        b = int(lab.replace("nb>=", "").replace("nb=", ""))
        groups = [[cs[(d, c, B)][0] - cs[(d, c, A)][0]
                   for d2, c in bins[b] if d2 == d and (d, c, A) in cs and (d, c, B) in cs]
                  for d in store.datasets]
        assert (cells[-2], cells[-1]) == mod.boot_ci([g for g in groups if g])
    # statement: the paired table names the bootstrap scheme, nothing else changes
    assert "bootstrap=" not in plain.statement
    assert paired.statement.replace(f" | bootstrap={mod.BOOT_SCHEME}", "") == plain.statement
    # the nonref split (the pre-registered E1 reading) works the same way
    nonref = mod.nb_table(store, methods, store.datasets, cfg.replace(split="nonref"), pair=(A, B))
    assert nonref.columns == paired.columns and len(nonref.rows) == 5


def normalise(text):
    """Drop the two nb-table columns and the bootstrap statement field so files written
    before and after the change compare equal when every number is unchanged."""
    out, strip = [], False
    for line in text.split("\n"):
        if line.startswith("| | cameras |") and line.endswith("| cluster CI lo | cluster CI hi |"):
            strip = True
        elif not line.startswith("|"):
            strip = False
        if strip and line.startswith("|"):
            cells = line.split("|")
            assert cells[0] == "" and cells[-1] == ""
            line = "|".join(cells[:-3]) + "|"
        if line.startswith("*aggregation="):
            line = re.sub(r" \| bootstrap=[^|]*(?= \| )", "", line)
        out.append(line)
    return "\n".join(out)


# 2026-09-09: the paper's column set changed on purpose.  The SAM 2 column moved from
# the unreproducible SegMaskNew1 to SegMaskNew3, and the SAM 3 columns became the
# adopted XW lineage (docs/sam2-baseline.md, ROADMAP "채택된 구성").  So the two things
# this used to check in one test are now separate: the stored file must still be
# reproducible from the committed scorer, and the scoring core must not have drifted
# from what produced the published numbers.
PAPER_RAW = os.path.join(REPO, "Data", "MVSeg", "jf_paper.json")
# report_jf.py prints the raw file name it was given into the header, and the stored
# documents were generated with the short name, so the tests must pass the short name.
PAPER_RAW_NAME = "jf_paper.json"
RAW_NAME = "jf_v2.json"
OLD_PAPER_METHODS = ["SegMaskNew1", "SegMaskSam3OneStage", "SegMaskSam3MVOpt"]


def run_paper(tmp_path, raw, methods=()):
    out = tmp_path / "paper.md"
    cmd = [sys.executable, os.path.join(REPO, "eval", "report_jf.py"),
           "--raw", raw, "--paper", str(out)]
    if methods:
        cmd += ["--methods", *methods]
    subprocess.run(cmd, check=True, cwd=REPO,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out.read_text()


def test_paper_tables_are_reproducible(rj, tmp_path):
    """docs/raw/paper_tables.md is exactly what the committed scorer writes today."""
    if not os.path.isfile(PAPER):
        pytest.skip(f"{PAPER} not present")
    if not os.path.isfile(PAPER_RAW):
        pytest.skip(f"{PAPER_RAW} not present (score the adopted lineage first)")
    fresh = run_paper(tmp_path, PAPER_RAW_NAME)
    assert "| cluster CI lo | cluster CI hi |" in fresh
    assert fresh == open(PAPER, encoding="utf-8").read(), \
        "docs/raw/paper_tables.md is out of date: rerun --paper --raw jf_paper.json"
    mod = load_report_jf()
    assert mod.PAPER_METHODS == ["SegMaskNew3", "SegMaskSam3XW0", "SegMaskSam3XW1GPS4"]
    for col in mod.PAPER_METHODS:
        assert f"| {col} |" in fresh or f" {col} |" in fresh


def test_scoring_core_has_not_drifted(rj, tmp_path):
    """The old column set still reproduces the numbers frozen at commit 5353d07.

    This is the drift guard: the paper's columns changed, but J and F themselves must
    not have.  Only the nb-table CI columns and the bootstrap statement field may
    differ, which is what normalise() removes."""
    snapshot = os.path.join(REPO, "tests", "data", "paper_tables_5353d07.md")
    if not os.path.isfile(snapshot):
        pytest.skip(f"{snapshot} not present")
    fresh = run_paper(tmp_path, RAW_NAME, OLD_PAPER_METHODS)
    assert normalise(fresh) == normalise(open(snapshot, encoding="utf-8").read())
    # the normaliser only removes what it claims: 4 nb tables x (header + rule + 5 rows)
    n_stripped = sum(1 for a, b in zip(fresh.split("\n"), normalise(fresh).split("\n")) if a != b)
    assert n_stripped >= 4 * 7


# ------------------------------------------------------------------ P4: --nb-mode
def test_nb_mode_both_partition():
    """W=1: `lower` bins nb=0 (9 index-0 cameras) / nb>=1 (36); `both` bins nb=0 (empty),
    nb=1 (15 edge cameras), nb>=2 (30 interior), plus the cumulative nb>=1 row (SPEC_P4
    amendment A4) that is the primary bin of modes B/C/E; `--split nonref` gives the 28
    (lower, A/D) and 30 (both, B/C/E) camera sets of the pre-registered E1 reading."""
    if not os.path.isfile(RAW_XW):
        pytest.skip(f"{RAW_XW} not present")
    mod = load_report_jf()
    store = mod.Store(mod.load([RAW_XW]))
    for m in (XW0, XW4):
        if m not in store.methods:
            pytest.skip(f"{m} not in {RAW_XW}")
    methods, pair = [XW0, XW4], (XW0, XW4)
    cfg_l, cfg_b = mod.Cfg(window=1), mod.Cfg(window=1, nb_mode="both")
    assert mod.nb_max(cfg_l) == 1 and mod.nb_max(cfg_b) == 2
    lower = mod.nb_table(store, methods, store.datasets, cfg_l, pair=pair)
    both = mod.nb_table(store, methods, store.datasets, cfg_b, pair=pair)
    assert [lab for lab, _ in lower.rows] == ["nb=0", "nb>=1"]
    assert [c[0] for _, c in lower.rows] == [9, 36]
    assert [lab for lab, _ in both.rows] == ["nb=0", "nb=1", "nb>=2", "nb>=1"]
    assert [c[0] for _, c in both.rows] == [0, 15, 30, 45]
    assert both.columns == lower.columns
    empty = dict(zip(both.columns, both.rows[0][1]))                  # nb=0 stays as a nan row
    assert all(math.isnan(empty[k]) for k in (XW0, XW4, "delta J&F", "cluster CI lo", "cluster CI hi"))
    assert (empty["wins"], empty["ties"], empty["losses"]) == (0, 0, 0)
    for lab, cells in both.rows[1:]:
        d = dict(zip(both.columns, cells))
        assert not math.isnan(d["delta J&F"]) and d["cluster CI lo"] <= d["delta J&F"] <= d["cluster CI hi"]
        assert d["wins"] + d["ties"] + d["losses"] == d["cameras"]
    # the two definitions partition the same 45 cameras; nb=0 (lower) = view index 0
    cams = [(d, c) for d in store.datasets for c in mod.select_cams(store, d, cfg_l) if (d, c) in store.gt]
    assert len(cams) == 45
    lb = {k: mod.nb_of(store, *k, cfg_l) for k in cams}
    bb = {k: mod.nb_of(store, *k, cfg_b) for k in cams}
    assert [sum(1 for x in lb.values() if x == b) for b in (0, 1)] == [9, 36]
    assert [sum(1 for x in bb.values() if x == b) for b in (0, 1, 2)] == [0, 15, 30]
    assert all((lb[k] == 0) == (store.view_index(*k) == 0) for k in cams)
    assert all(bb[k] >= lb[k] for k in cams) and all(bb[k] == 2 for k in cams if 0 < store.view_index(*k) < store.n_views(*k) - 1)
    # the cumulative row pools every camera with a neighbour: the same cameras (and the
    # same cluster bootstrap) as the paired camera-level row of the whole set
    stats = mod.paired_tables(store, XW0, XW4, store.datasets, cfg_b)[0]
    cam_row = dict(zip(stats.columns, next(cells for lab, cells in stats.rows if lab.startswith("camera"))))
    cum = dict(zip(both.columns, both.rows[-1][1]))
    assert cum["cameras"] == cam_row["n"] == 45
    assert cum["delta J&F"] == pytest.approx(cam_row["mean"], abs=1e-12)
    assert cum["cluster CI lo"] == pytest.approx(cam_row["cluster CI lo"], abs=1e-12)
    assert cum["cluster CI hi"] == pytest.approx(cam_row["cluster CI hi"], abs=1e-12)
    # the E1 primary bins: nonref -> 28 (lower nb>=1, modes A/D) and 30 (both nb>=1, B/C/E)
    lower_nr = mod.nb_table(store, methods, store.datasets, cfg_l.replace(split="nonref"), pair=pair)
    both_nr = mod.nb_table(store, methods, store.datasets, cfg_b.replace(split="nonref"), pair=pair)
    assert [c[0] for _, c in lower_nr.rows] == [2, 28]
    assert both_nr.rows[-1][0] == "nb>=1" and both_nr.rows[-1][1][0] == 30
    assert both_nr.rows[0][1][0] == 0 and both_nr.rows[1][1][0] + both_nr.rows[2][1][0] == 30
    # statement: the default keeps every existing line, `both` adds one field
    assert "nb_mode=" not in mod.Cfg().statement(15) and "nb_mode=" not in cfg_l.statement(15)
    assert " | nb_mode=both | " in cfg_b.statement(15)
    nd = len(store.datasets)
    assert both.statement == lower.statement.replace(f" | datasets={nd} | ", f" | datasets={nd} | nb_mode=both | ")
