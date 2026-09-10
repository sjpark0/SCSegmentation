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
