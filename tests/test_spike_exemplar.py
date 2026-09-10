"""S1-E2-0 / S1-E2-0' spike: pure summary logic (host) and the masked ROI pooling (container, torch)."""
import importlib.util
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("spike_exemplar", os.path.join(REPO, "MVSeed", "spike_exemplar.py"))
spike = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spike)


def _arm(iou, n_cand=1, rank=1, all_iou=None, dlogit=5.0):
    return {"n_cand": n_cand, "all_best_iou": iou if all_iou is None else all_iou, "all_best_rank": 1,
            "all_best_prob": 0.9, "cand_best_iou": iou, "cand_best_rank": rank if n_cand else None,
            "cand_best_prob": 0.9, "cand_top1_iou": iou, "cand_top1_prob": 0.9, "cand_probs": [0.9],
            "max_abs_dlogit_vs_base": dlogit}


def _rec(i, same, slot, same_mask, slot_mask, fill, tag="registered"):
    arms = {"same": _arm(same), "slot": _arm(slot), "replace": _arm(slot), "base": _arm(0.0, n_cand=0),
            "same_mask": _arm(same_mask), "slot_mask": _arm(slot_mask),
            "same_mask0": _arm(same_mask), "slot_mask0": _arm(slot_mask),
            "same_maskimg": _arm(same_mask), "slot_maskimg": _arm(slot_mask)}
    return {"set": tag, "scene": f"S{i}", "cam": "c", "obj": i, "fill_ratio": fill, "arms": arms}


def test_arm_keys_keep_e2_0_names():
    assert spike.arm_keys("box") == ("same", "slot")
    assert spike.arm_keys("mask") == ("same_mask", "slot_mask")


def test_summarize_e2_0b_verdict_on_mask_arm_and_paired_split():
    # box: c1 3/4, c2 2/4 ; mask: c1 4/4, c2 4/4 -> verdict on mask passes although box fails c2
    recs = [_rec(1, 0.9, 0.9, 0.95, 0.9, 0.7),
            _rec(2, 0.9, 0.2, 0.9, 0.8, 0.3),      # low fill: mask recovers
            _rec(3, 0.5, 0.1, 0.85, 0.7, 0.2),     # low fill: mask recovers both
            _rec(4, 0.9, 0.6, 0.9, 0.6, 0.6)]
    s = spike.summarize(recs, {"t": 1}, variant="e2_0b")
    reg = s["registered"]
    assert reg["c1_same_image"]["hits"] == 3 and reg["c1_same_mask"]["hits"] == 4
    assert reg["c2_slot"]["hits"] == 2 and reg["c2_slot_mask"]["hits"] == 4
    assert reg["c2_slot_mask"]["max_abs_dlogit_vs_base_min"] == 5.0
    assert s["verdict"]["arm"] == "mask" and s["verdict"]["verdict"].startswith("pass")
    pr = reg["paired_slot_mask_vs_box"]
    assert pr["fill_lt_0.5"]["n"] == 2 and pr["fill_lt_0.5"]["hits_slot"] == 0 and pr["fill_lt_0.5"]["hits_slot_mask"] == 2
    assert pr["fill_ge_0.5"]["n"] == 2 and pr["fill_ge_0.5"]["n_up"] == 0
    assert pr["all"]["n_up"] == 2 and pr["all"]["n_down"] == 0
    # e2_0 variant on the same records: verdict on box, c2 2/4 < 0.7 -> discard
    s0 = spike.summarize(recs, {"t": 1}, variant="e2_0")
    assert s0["verdict"]["arm"] == "box" and s0["verdict"]["verdict"].startswith("discard: criterion 1")   # box c1 3/4 fails first
    assert "c2_slot_mask" not in s0["registered"]


def test_summarize_kill_rule_c1_first():
    recs = [_rec(i, 0.9, 0.9, 0.5, 0.9, 0.5) for i in range(4)]     # mask c1 0/4 -> criterion 1 kills
    s = spike.summarize(recs, {}, variant="e2_0b")
    assert s["verdict"]["verdict"].startswith("discard: criterion 1")


def test_compare_with_reference_counts_identical_box_arms():
    recs = [_rec(1, 0.9, 0.9, 0.9, 0.9, 0.5), _rec(2, 0.9, 0.2, 0.9, 0.8, 0.3)]
    ref = [_rec(1, 0.9, 0.9, 0.0, 0.0, 0.5), _rec(2, 0.9, 0.25, 0.0, 0.0, 0.3)]
    c = spike.compare_with_reference(recs, ref)
    assert c["n_compared"] == 2 and c["n_equal"] == 1 and c["max_abs_diff_cand_best_iou"] == 0.05
    s = spike.summarize(recs, {}, variant="e2_0b", reference=ref)
    assert s["box_vs_e2_0_reference"]["n_equal"] == 1


# ----------------------------------------------------------------------------- torch part
torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")


def test_feat_mask_from_gt_area_average():
    gt = torch.zeros(4, 4, dtype=torch.bool)
    gt[:2, :2] = True
    gt[2, 2] = True
    m = spike.feat_mask_from_gt(gt, 2, 2)
    assert m.shape == (1, 1, 2, 2)
    assert torch.allclose(m[0, 0], torch.tensor([[1.0, 0.0], [0.0, 0.25]]))


def _feats_const_inside(H=16, W=16, C=3, a=2.0, b=-1.0):
    m = torch.zeros(1, 1, H, W)
    m[:, :, 4:12, 4:8] = 1.0                             # object: rows 4-11, cols 4-7
    f = torch.full((1, C, H, W), b)
    f[:, :, 4:12, 4:8] = a
    return f, m


def test_masked_roi_pool_identity_when_mask_is_one():
    f, _ = _feats_const_inside()
    f = f + torch.randn_like(f)
    rois = [torch.tensor([[2.0, 3.0, 13.0, 14.0]])]
    ref = torchvision.ops.roi_align(f, rois, 7)
    got = spike.masked_roi_pool(f, torch.ones(1, 1, 16, 16), rois, 7, normalize=True, fill_min=0.05)
    assert torch.allclose(got, ref, atol=1e-6)
    got0 = spike.masked_roi_pool(f, torch.ones(1, 1, 16, 16), rois, 7, normalize=False, fill_min=0.05)
    assert torch.allclose(got0, ref, atol=1e-6)


def test_masked_roi_pool_normalised_sees_object_only():
    f, m = _feats_const_inside(a=2.0, b=-1.0)
    rois = [torch.tensor([[0.0, 0.0, 16.0, 16.0]])]      # the whole frame: object fills 1/8 of the box
    got = spike.masked_roi_pool(f, m, rois, 7, normalize=True, fill_min=0.05)
    assert torch.allclose(got, torch.full_like(got, 2.0), atol=1e-4)   # every bin = object value (mean fill)
    got0 = spike.masked_roi_pool(f, m, rois, 7, normalize=False, fill_min=0.05)
    assert got0.max() <= 2.0 + 1e-6 and got0.mean() < 1.0                # naive zeroing shrinks the token
    assert (got0 >= -1e-6).all()                                          # background (-1) never leaks in


def test_masked_encode_boxes_matches_reference_with_full_mask():
    from types import SimpleNamespace
    torch.manual_seed(0)
    d, P = 8, 6

    class PosEnc:
        def encode_boxes(self, cx, cy, w, h):
            return torch.stack([cx, cy, w, h, cx * w, cy * h], -1)

    enc = SimpleNamespace(boxes_direct_project=torch.nn.Linear(4, d), boxes_pool_project=torch.nn.Conv2d(d, d, 7),
                          boxes_pos_enc_project=torch.nn.Linear(P, d), label_embed=torch.nn.Embedding(3, d),
                          pos_enc=PosEnc(), roi_size=7, d_model=d)
    feats = torch.randn(1, d, 12, 12)
    boxes = torch.tensor([[[0.5, 0.5, 0.6, 0.4]]])       # (n_boxes=1, bs=1, 4) cxcywh normalised
    labels = torch.ones(1, 1, dtype=torch.long)
    try:
        from sam3.model.box_ops import box_cxcywh_to_xyxy
    except ImportError:
        pytest.skip("sam3 package not importable on this host")
    # reference = the package algorithm (geometry_encoders.py 632-681) written out
    xyxy = box_cxcywh_to_xyxy(boxes) * torch.tensor([12, 12, 12, 12.0]).view(1, 1, 4)
    ref_pool = torchvision.ops.roi_align(feats, xyxy.float().transpose(0, 1).unbind(0), 7)
    ref = enc.boxes_direct_project(boxes) + enc.boxes_pool_project(ref_pool).view(1, 1, d).transpose(0, 1)
    cx, cy, w, h = boxes.unbind(-1)
    ref = ref + enc.boxes_pos_enc_project(enc.pos_enc.encode_boxes(cx.flatten(), cy.flatten(), w.flatten(), h.flatten()).view(1, 1, P))
    ref = ref + enc.label_embed(labels)
    got, _ = spike.masked_encode_boxes(enc, boxes, None, labels, feats, torch.ones(1, 1, 12, 12), True, 0.05)
    assert torch.allclose(got, ref, atol=1e-5)
    # a mask that differs from the box changes the token
    m = torch.zeros(1, 1, 12, 12)
    m[:, :, 5:7, 5:7] = 1.0
    got_m, _ = spike.masked_encode_boxes(enc, boxes, None, labels, feats, m, True, 0.05)
    assert not torch.allclose(got_m, ref, atol=1e-3)


# ----------------------------------------------------------------------------- S1-E2-0'' (points / negatives)
def _rec_c(i, box_same, box_slot, pts_same, pts_slot, neg_same, neg_slot, fill=0.5, tag="registered"):
    arms = {"same": _arm(box_same), "slot": _arm(box_slot), "base": _arm(0.0, n_cand=0)}
    for v, a, b in (("pts", pts_same, pts_slot), ("neg", neg_same, neg_slot), ("ptsonly", pts_same, pts_slot),
                    ("negnc", neg_same, neg_slot), ("negall", neg_same, neg_slot), ("ptsneg", neg_same, neg_slot)):
        arms[f"same_{v}"], arms[f"slot_{v}"] = _arm(a), _arm(b)
    return {"set": tag, "scene": f"S{i}", "cam": "c", "obj": i, "fill_ratio": fill, "arms": arms}


def test_summarize_e2_0c_verdict_per_arm_and_adoption_rule():
    # box: c1 4/4, c2 3/4 (0.75).  pts: c1 4/4, c2 4/4, no pair worse -> adopt.
    # neg: c1 3/4 (below box) although c2 4/4 -> not adopted.
    recs = [_rec_c(1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9),
            _rec_c(2, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9),
            _rec_c(3, 0.9, 0.2, 0.9, 0.8, 0.9, 0.8),
            _rec_c(4, 0.9, 0.9, 0.9, 0.9, 0.5, 0.9)]
    s = spike.summarize(recs, {}, variant="e2_0c")
    v = s["verdicts"]
    assert set(v) == {"pts", "neg"} and s["verdict"] is v["pts"]
    assert v["pts"]["verdict"].startswith("pass") and v["pts"]["adopt_over_box"] is True
    assert v["neg"]["verdict"].startswith("discard: criterion 1") and v["neg"]["adopt_over_box"] is False
    assert s["registered"]["c2_slot_pts"]["hits"] == 4 and s["registered"]["c1_same_neg"]["hits"] == 3
    # a pair-level loss (median d < 0) blocks adoption even when the rates pass
    recs2 = [_rec_c(i, 0.9, 0.9, 0.9, 0.8, 0.9, 0.9) for i in range(4)]
    s2 = spike.summarize(recs2, {}, variant="e2_0c")
    assert s2["verdicts"]["pts"]["verdict"].startswith("pass") and s2["verdicts"]["pts"]["adopt_over_box"] is False


def test_negative_boxes_modes_and_order():
    gt = torch.zeros(100, 100, dtype=torch.long)
    gt[:, :] = 9                     # floor: its box contains everything
    gt[40:60, 40:60] = 1             # target
    gt[50:70, 55:80] = 2             # partial overlap with the target box
    gt[10:15, 10:15] = 3             # far away, no overlap
    gt[45:48, 45:48] = 4             # inside the target box (small)
    box = spike.tight_box_xyxy(gt == 1)
    assert box == (40, 40, 60, 60)
    ov = spike.negative_boxes(gt, 1, box, mode="overlap")
    ids = [v for v, _ in ov]
    assert ids[0] == 9 and set(ids) == {9, 2, 4}          # largest intersection first; 3 excluded
    part = spike.negative_boxes(gt, 1, box, mode="partial")
    assert [v for v, _ in part] == [2, 4]                 # the containing floor is skipped
    allb = spike.negative_boxes(gt, 1, box, mode="all", max_neg=2)
    assert len(allb) == 2 and allb[0][0] == 4             # nearest centre first, capped
    tiny = spike.negative_boxes(gt, 1, box, mode="overlap", min_overlap=0.5)
    assert [v for v, _ in tiny] == [9, 2]                 # 4 (9 px) is below 50% of the box area


def test_interior_points_are_inside_and_spread():
    cv2 = pytest.importorskip("cv2")
    gt = torch.zeros(60, 80, dtype=torch.bool)
    gt[10:50, 20:70] = True
    pts = spike.interior_points(gt, k=8, interior_px=3)
    assert pts.shape == (8, 2)
    for x, y in pts:
        assert gt[y, x] and 13 <= y <= 46 and 23 <= x <= 66      # inside and >= 3 px from the boundary
    assert len({(int(x), int(y)) for x, y in pts}) == 8            # distinct
    xs, ys = pts[:, 0], pts[:, 1]
    assert xs.max() - xs.min() > 30 and ys.max() - ys.min() > 20   # spread over the rectangle
    small = torch.zeros(20, 20, dtype=torch.bool); small[5:8, 5:8] = True
    p2 = spike.interior_points(small, k=8, interior_px=3)         # interior too thin -> whole mask, <= 9 px
    assert 1 <= len(p2) <= 8 and all(small[y, x] for x, y in p2)


def test_exemplar_prompt_formats():
    pytest.importorskip("cv2")
    try:
        from sam3.model.geometry_encoders import Prompt  # noqa: F401
    except ImportError:
        pytest.skip("sam3 not importable")
    gt = torch.zeros(100, 100, dtype=torch.long)
    gt[:, :] = 9; gt[40:60, 40:60] = 1; gt[50:70, 55:80] = 2
    for variant, n_box, labels, n_pts in (("box", 1, [1], 0), ("pts", 1, [1], 8), ("ptsonly", 0, [], 8),
                                          ("neg", 3, [1, 0, 0], 0), ("negnc", 2, [1, 0], 0), ("negall", 3, [1, 0, 0], 0),
                                          ("ptsneg", 3, [1, 0, 0], 8)):
        pr, meta = spike.exemplar_prompt(variant, gt, 1, "cpu")
        if n_box:
            assert list(pr.box_embeddings.shape) == [n_box, 1, 4] and pr.box_labels[:, 0].tolist() == labels
            assert pr.box_mask.shape == (1, n_box) and not pr.box_mask.any()
            assert (pr.box_embeddings >= 0).all() and (pr.box_embeddings <= 1).all()
        else:
            assert pr.box_embeddings is None or pr.box_embeddings.shape[0] == 0
        if n_pts:
            assert list(pr.point_embeddings.shape) == [n_pts, 1, 2] and pr.point_labels.shape == (n_pts, 1)
            assert (pr.point_labels == 1).all() and (pr.point_embeddings >= 0).all() and (pr.point_embeddings <= 1).all()
        else:
            assert pr.point_embeddings is None or pr.point_embeddings.shape[0] == 0
        assert meta["box"] == [40, 40, 60, 60]
    # the positive box of `box` equals E2-0's box_prompt_from_mask
    pr_box, _ = spike.exemplar_prompt("box", gt, 1, "cpu")
    pr_ref, box = spike.box_prompt_from_mask(gt == 1, "cpu")
    assert torch.allclose(pr_box.box_embeddings, pr_ref.box_embeddings) and box == [40, 40, 60, 60]
