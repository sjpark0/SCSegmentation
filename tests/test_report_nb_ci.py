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


def test_paper_tables_columns_only(rj, tmp_path):
    if not os.path.isfile(PAPER):
        pytest.skip(f"{PAPER} not present")
    out = tmp_path / "paper.md"
    subprocess.run([sys.executable, os.path.join(REPO, "eval", "report_jf.py"), "--paper", str(out)],
                   check=True, cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    fresh, stored = out.read_text(), open(PAPER, encoding="utf-8").read()
    assert "| cluster CI lo | cluster CI hi |" in fresh
    assert fresh == stored, "docs/raw/paper_tables.md is out of date: rerun --paper"
    # the numbers must still be the ones published before the nb-bin CI columns
    # existed (frozen from commit 5353d07); only the added columns may differ
    snapshot = os.path.join(REPO, "tests", "data", "paper_tables_5353d07.md")
    assert normalise(fresh) == normalise(open(snapshot, encoding="utf-8").read())
    # the normaliser only removes what it claims: 4 nb tables x (header + rule + 5 rows)
    n_stripped = sum(1 for a, b in zip(fresh.split("\n"), normalise(fresh).split("\n")) if a != b)
    assert n_stripped >= 4 * 7
