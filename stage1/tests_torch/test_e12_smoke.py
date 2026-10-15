"""S1-E12: 20-step training smokes of the two new arms against geoset_pred_ft (rng-stream identity, step-1 loss identity
of the zero-dose arm, cut statistics, config).  A separate module: no in-process model may hold the GPU while the training
subprocesses run (the module-scoped `setup` fixture of test_e12_cut_torch.py would)."""
import csv
import json
import math
import os
import subprocess
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "Data")
gpu_blocks = pytest.mark.skipif(not torch.cuda.is_available() or not os.path.isdir(os.path.join(DATA, "Blocks", "Cache", "sam3_feats")),
                                reason="needs GPU and the Blocks caches")


# ---------------------------------------------------------------- GPU: 20-step training smokes
def _run(tmp, arm, steps=20):
    cmd = [sys.executable, "-m", "geomem.train", "--arm", arm, "--fold", "0", "--max-steps", str(steps),
           "--scenes", "Blocks", "--out", tmp, "--data", DATA]
    subprocess.run(cmd, check=True, cwd=REPO)
    d = os.path.join(tmp, arm, "fold0")
    return list(csv.DictReader(open(os.path.join(d, "log.csv")))), json.load(open(os.path.join(d, "config.json")))


@gpu_blocks
def test_train_smokes_stream_identity(tmp_path):
    logs = {a: _run(str(tmp_path), a) for a in ("geoset_pred_ft", "geoset_cut0_ft", "geoset_cutall_ft")}
    rows = {a: r for a, (r, _) in logs.items()}
    cfg = {a: c for a, (_, c) in logs.items()}
    for a in rows:
        assert len(rows[a]) == 20 and all(math.isfinite(float(r["loss"])) for r in rows[a])
        assert cfg[a]["trainable_params"] == 5_922_321 and cfg[a]["extra_mem_pred"] is True and cfg[a]["extra_mem_prob"] == 0.5
    for col in ("objs", "mem_coin", "mem_cam"):                                    # the shared rng / memory_rng streams
        assert [r[col] for r in rows["geoset_pred_ft"]] == [r[col] for r in rows["geoset_cut0_ft"]] == \
               [r[col] for r in rows["geoset_cutall_ft"]], col
    assert rows["geoset_pred_ft"][0]["loss"] == rows["geoset_cut0_ft"][0]["loss"]
    assert rows["geoset_cutall_ft"][0]["loss"] != rows["geoset_pred_ft"][0]["loss"], "the cut never reached the model"
    assert "cut_prob" not in cfg["geoset_pred_ft"] and all(r["cut"] == "" for r in rows["geoset_pred_ft"])
    assert cfg["geoset_cut0_ft"]["cut_prob"] == 0.0 and all(r["cut"] == "0" and r["n_cut"] == "0" for r in rows["geoset_cut0_ft"])
    assert cfg["geoset_cutall_ft"]["cut_prob"] == 1.0 and all(r["cut"] == "1" for r in rows["geoset_cutall_ft"])
    cut = rows["geoset_cutall_ft"]
    assert sum(int(r["n_cut"]) >= 1 for r in cut) >= 19
    fr = [float(r["cut_frac"]) for r in cut if r["cut_frac"]]
    assert fr and all(0.0 < x <= 0.71 for x in fr)


