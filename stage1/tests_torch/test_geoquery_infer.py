"""GPU: untrained GeoQuery inference on Blocks equals untrained geobias_only_ft and records hub statistics."""
import glob
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "Data")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not os.path.isdir(os.path.join(DATA, "Blocks")),
                                reason="needs GPU and Data/Blocks")


def _run(arm, out):
    subprocess.run([sys.executable, "methods/geomem/run.py", "--scene", "Blocks", "--arm", arm, "--untrained",
                    "--out", out, "--overwrite", "--data", DATA], check=True, cwd=REPO)
    return os.path.join(DATA, "Blocks", "Result", out)


def test_untrained_geoquery_matches_untrained_geobias_only_and_writes_stats():
    a, b = _run("geobias_only_ft", "dbg_gbo"), _run("geoquery_ft", "dbg_gq")
    try:
        pngs = sorted(os.path.relpath(p, a) for p in glob.glob(os.path.join(a, "*", "*.png")))
        assert len(pngs) > 10 and pngs == sorted(os.path.relpath(p, b) for p in glob.glob(os.path.join(b, "*", "*.png")))
        for rel in pngs:
            ma, mb = cv2.imread(os.path.join(a, rel), 0), cv2.imread(os.path.join(b, rel), 0)
            assert (ma == mb).mean() > 0.9999, rel
        run = json.load(open(os.path.join(b, "RUN.json")))
        cfg = run["config"]
        assert run["status"] == "done" and run["error"] is None and cfg["use_hub"] is True
        assert 0.0 <= cfg["entropy_mean"] <= 1.0 and 0.0 <= cfg["ref_frac_mean"] <= 1.0 and cfg["top3_frac_mean"] <= 1.0
        assert cfg["n_hub_views"] == 10 and cfg["cloud_size_mean"] > 0 and cfg["n_empty_clouds"] >= 0
        assert not os.path.isdir(b + "_r1")
    finally:
        shutil.rmtree(a, ignore_errors=True)
        shutil.rmtree(b, ignore_errors=True)
