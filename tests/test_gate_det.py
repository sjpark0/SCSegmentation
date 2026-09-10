"""S1-E2 gate script: pure summary/tagging logic (host) and the union scoring (container, torch)."""
import importlib.util
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("gate_det", os.path.join(REPO, "MVSeed", "gate_det.py"))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def _row(scene, cam, obj, gt_px, best, rank=1, top1=None, box=None, pts=None, n_union=2, rig="pinhole", top1_prob=0.8):
    return {"scene": scene, "cam": cam, "obj": obj, "gt_px": gt_px, "rig": rig, "best_iou": best, "best_rank": rank if n_union else None,
            "top1_iou": best if top1 is None else top1, "top1_prob": top1_prob if n_union else None, "n_union": n_union, "n_union_dedup": n_union,
            "n_cand": {"box": n_union // 2, "pts": n_union - n_union // 2},
            "best_iou_by_run": {"box": best if box is None else box, "pts": best if pts is None else pts}}


def _rows(n_ge2k=19, n_pin=16, exist_ge2k=12, exist_pin=9, n_lt2k=22, n_n0=623, n_empty=132, halluc=10, rank=1):
    rows, f0, n0 = [], [], []
    for i in range(n_ge2k):
        rig = "pinhole" if i < n_pin else "hemisphere"
        ex = (i < exist_pin) if rig == "pinhole" else (i - n_pin < exist_ge2k - exist_pin)
        rows.append(_row("S", "c1", i, 5000, 0.9 if ex else 0.1, rank=rank, rig=rig)); f0.append({"scene": "S", "cam": "c1", "obj": i, "kind": "misplaced", "raw_J": 0.1})
    for i in range(n_lt2k):
        rows.append(_row("S", "c2", i, 500, 0.6 if i % 2 else 0.0)); f0.append({"scene": "S", "cam": "c2", "obj": i, "kind": "empty" if i == 0 else "misplaced", "raw_J": 0.0})
    for i in range(n_n0):
        rows.append(_row("T", "c1", i, 3000 if i % 3 else 500, 0.9)); n0.append({"scene": "T", "cam": "c1", "obj": i})
    for i in range(n_empty):
        rows.append(_row("U", "c1", i, 0, 0.0, n_union=1 if i < halluc else 0))
    return rows, f0, n0


def test_tag_and_summary_pass_case():
    rows, f0, n0 = _rows()
    f0m, n0m = gate.membership(f0, n0)
    rows = [gate.tag_row(dict(r), f0m, n0m) for r in rows]
    s = gate.summarize_gate(rows)
    assert s["denominators"]["match"] is True
    assert s["f0_ge2k"]["exist"] == 12 and s["f0_ge2k_pinhole"]["exist"] == 9 and s["f0_lt2k"]["exist"] == 11
    assert s["f0_ge2k"]["exist_by_rig"] == {"pinhole": [9, 16], "hemisphere": [3, 3]}
    assert s["gt_empty"]["halluc"] == 10 and s["gt_empty"]["halluc_rate"] == round(10 / 132, 4)
    assert s["gates"]["gate1_exist_ge2k"]["pass"] and s["gates"]["gate1p_pinhole"]["pass"] and s["gates"]["gate2_rank"]["pass"] and s["gates"]["gate3_halluc"]["pass"]
    assert s["gates"]["verdict"] == "proceed"
    assert set(s["f0_by_kind"]) == {"misplaced", "empty"} and s["n0_by_size"][">=10000"]["n"] == 0


def test_gate_thresholds_and_contingencies():
    # 10/19 fails gate 1 (0.526 < 0.55); 11/19 passes
    rows, f0, n0 = _rows(exist_ge2k=10, exist_pin=8)
    f0m, n0m = gate.membership(f0, n0)
    s = gate.summarize_gate([gate.tag_row(dict(r), f0m, n0m) for r in rows])
    assert not s["gates"]["gate1_exist_ge2k"]["pass"] and s["gates"]["verdict"].startswith("discard G-det: gate 1")
    rows, f0, n0 = _rows(exist_ge2k=11, exist_pin=8)
    s = gate.summarize_gate([gate.tag_row(dict(r), *gate.membership(f0, n0)) for r in rows])
    assert s["gates"]["gate1_exist_ge2k"]["pass"] and s["gates"]["verdict"] == "proceed"
    # with F0's composition (16 pinhole + 3 hemisphere) gate 1 implies the pinhole sub-gate: 7 pinhole + 3 = 10 < 11
    rows, f0, n0 = _rows(exist_ge2k=11, exist_pin=7)
    s = gate.summarize_gate([gate.tag_row(dict(r), *gate.membership(f0, n0)) for r in rows])
    assert s["f0_ge2k"]["exist"] == 10 and s["gates"]["verdict"].startswith("discard G-det: gate 1")
    # the hemisphere-limited branch needs a different composition (10 pinhole + 9 hemisphere): 7 + 5 = 12 passes gate 1
    rows, f0, n0 = _rows(n_pin=10, exist_ge2k=12, exist_pin=7)
    s = gate.summarize_gate([gate.tag_row(dict(r), *gate.membership(f0, n0)) for r in rows])
    assert s["denominators"]["match"] is False and s["f0_ge2k_pinhole"]["exist"] == 7
    assert s["gates"]["gate1_exist_ge2k"]["pass"] and "hemisphere-limited" in s["gates"]["verdict"]
    # hallucination 14/132 > 10% -> proceed with the empty-G0 restriction
    rows, f0, n0 = _rows(halluc=14)
    s = gate.summarize_gate([gate.tag_row(dict(r), *gate.membership(f0, n0)) for r in rows])
    assert not s["gates"]["gate3_halluc"]["pass"] and "never fills an empty G0" in s["gates"]["verdict"]
    # rank median 4 among existing -> gate 2 fails -> discard
    rows, f0, n0 = _rows(rank=4)
    s = gate.summarize_gate([gate.tag_row(dict(r), *gate.membership(f0, n0)) for r in rows])
    assert s["gates"]["verdict"] == "discard G-det: gate 2 (rank) failed"


def test_tag_row_rejects_unknown_gt_present_row():
    f0m, n0m = gate.membership([], [])
    with pytest.raises(AssertionError):
        gate.tag_row(_row("X", "c", 1, 100, 0.5), f0m, n0m)
    r = gate.tag_row(_row("X", "c", 1, 0, 0.0, n_union=0), f0m, n0m)
    assert r["set"] == "gt_empty" and r["size_bin"] is None


torch = pytest.importorskip("torch")


def test_score_union_ranks_pooled_candidates_and_dedups():
    H, W = 8, 8
    gt = torch.zeros(H, W, dtype=torch.bool); gt[2:6, 2:6] = True
    def m(y0, y1, x0, x1, v=5.0):
        t = torch.full((4, 4), -5.0); t[y0:y1, x0:x1] = v; return t           # low-res logits, upsampled x2
    runs = {"box": [(0.9, m(0, 1, 0, 1)), (0.6, m(1, 3, 1, 3))],             # top-1 wrong, 2nd ~ target
            "pts": [(0.7, m(1, 3, 1, 3))]}                                   # duplicate of the target candidate
    rec = gate.score_union(runs, gt, (H, W))
    assert rec["n_cand"] == {"box": 2, "pts": 1} and rec["n_union"] == 3
    assert rec["best_src"] == "pts" and rec["best_rank"] == 2                # prob order: 0.9 box, 0.7 pts, 0.6 box
    assert rec["best_iou"] > 0.5 and rec["top1_iou"] < 0.5 and rec["top1_src"] == "box"
    assert rec["best_iou_by_run"]["box"] == rec["best_iou_by_run"]["pts"] == rec["best_iou"]
    assert rec["n_union_dedup"] == 2                                          # the two target copies collapse
    empty = gate.score_union(runs, torch.zeros(H, W, dtype=torch.bool), (H, W))
    assert empty["best_iou"] == 0.0 and empty["n_union"] == 3                 # gt-empty: IoU 0 by definition
    none = gate.score_union({"box": [], "pts": []}, gt, (H, W))
    assert none["n_union"] == 0 and none["best_rank"] is None and none["top1_prob"] is None
