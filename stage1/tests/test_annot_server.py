"""annot/server.py: 'not visible in this camera' must also work before any object exists (first camera of a new instant)."""
import json
import os
import sys

import numpy as np
import pytest
from PIL import Image

S1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, S1)
from annot import server  # noqa: E402
from annot.labels import read_label  # noqa: E402


@pytest.fixture
def inst(tmp_path):
    root = tmp_path / "CROOM"
    d = root / "T99_test" / "000010"
    (d / "images").mkdir(parents=True)
    (d / "annot" / "labels").mkdir(parents=True)
    for c in ("cam_00", "cam_01"):
        Image.fromarray(np.zeros((12, 16, 3), np.uint8)).save(d / "images" / f"{c}.jpg")
    server.ROOT = str(root)
    server.app.config["TESTING"] = True
    return d


def _events(d):
    p = d / "annot" / "log.jsonl"
    return [json.loads(l) for l in open(p)] if p.is_file() else []


def test_mark_empty_without_object(inst):
    c = server.app.test_client()
    r = c.post("/api/mark_empty", json=dict(take="T99_test", frame=10, cam="cam_00"))
    assert r.status_code == 200 and r.get_json()["ok"]
    lab = read_label(inst / "annot" / "labels" / "cam_00.png")
    assert lab.shape == (12, 16) and not lab.any()
    ev = _events(inst)
    assert ev[-1]["ev"] == "not_visible" and ev[-1]["obj"] == 0 and ev[-1]["cam"] == "cam_00"
    # the camera now counts as visited (status entry present, no objects)
    st = c.get("/api/instant?take=T99_test&frame=10").get_json()["status"]
    assert st == {"cam_00": []}
    # undo removes the all-zero label again
    r = c.post("/api/mark_empty", json=dict(take="T99_test", frame=10, cam="cam_00", undo=True))
    assert r.get_json()["removed_empty_label"] is True
    assert not (inst / "annot" / "labels" / "cam_00.png").exists()
    assert _events(inst)[-1]["ev"] == "not_visible_undo" and _events(inst)[-1]["obj"] == 0


def test_mark_empty_with_object_keeps_existing_label(inst):
    c = server.app.test_client()
    p = inst / "annot" / "labels" / "cam_01.png"
    lab = np.zeros((12, 16), np.uint8); lab[2:5, 3:9] = 1
    Image.fromarray(lab).save(p)
    r = c.post("/api/mark_empty", json=dict(take="T99_test", frame=10, cam="cam_01", obj=2))
    assert r.get_json()["ok"] and (read_label(p) == lab).all()
    assert _events(inst)[-1]["obj"] == 2
