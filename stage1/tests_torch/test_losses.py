import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geomem.losses import dice, iou_l1, presence_bce, reprojection_consistency, sigmoid_focal, supervised_loss  # noqa: E402
from geomem.warp_torch import reproject_coords  # noqa: E402


def _rect(B, h, w, y0, y1, x0, x1):
    m = torch.zeros(B, h, w, dtype=torch.bool); m[:, y0:y1, x0:x1] = True; return m


def test_focal_dice_perfect_prediction_near_zero():
    t = _rect(2, 32, 32, 8, 24, 8, 24)
    logits = torch.where(t, 10.0, -10.0).unsqueeze(1)
    assert sigmoid_focal(logits, t.unsqueeze(1).float()) < 1e-3
    assert dice(logits, t.unsqueeze(1).float()) < 1e-3
    bad = -logits
    assert dice(bad, t.unsqueeze(1).float()) > 0.9


def test_iou_l1_matches_actual_iou():
    t = _rect(1, 16, 16, 0, 8, 0, 16).unsqueeze(1).float()        # top half
    p = torch.where(_rect(1, 16, 16, 4, 12, 0, 16).unsqueeze(1), 10.0, -10.0)   # middle half -> IoU 1/3
    assert abs(iou_l1(p, t, torch.tensor([1 / 3])).item()) < 1e-6
    assert abs(iou_l1(p, t, torch.tensor([1.0])).item() - 2 / 3) < 1e-6


def test_supervised_loss_absent_object_finite():
    out = {"logits_1008": torch.randn(2, 1, 64, 64, requires_grad=True), "ious": torch.tensor([0.5, 0.5], requires_grad=True),
           "presence": torch.tensor([0.0, 0.0], requires_grad=True)}
    tgt = torch.zeros(2, 20, 30, dtype=torch.bool)                 # both objects absent in the target view
    loss, terms = supervised_loss(out, tgt)
    assert torch.isfinite(loss) and set(terms) == {"focal", "dice", "iou", "presence", "band"} and terms["band"] == 0.0
    loss.backward()
    assert torch.isfinite(out["logits_1008"].grad).all()


def test_presence_bce_direction():
    assert presence_bce(torch.tensor([5.0]), torch.tensor([True])) < presence_bce(torch.tensor([-5.0]), torch.tensor([True]))


def test_reprojection_identity_geometry():
    h, w, f = 40, 60, 50.0
    K = torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    z = torch.full((h, w), 2.0)
    v, u = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    p_u = torch.stack([(u - K[0, 2]) / f * z, (v - K[1, 2]) / f * z, z], dim=-1)      # view u == ref geometry
    grid, valid = reproject_coords(p_u, K, (h, w), depth_ref=z)
    assert valid.all()
    gt = _rect(1, h, w, 10, 25, 15, 45)
    good = torch.where(gt, 10.0, -10.0).unsqueeze(1)
    shifted = torch.where(_rect(1, h, w, 10, 25, 25, 55), 10.0, -10.0).unsqueeze(1)
    l_good = reprojection_consistency(good, gt, grid, valid)
    l_bad = reprojection_consistency(shifted, gt, grid, valid)
    assert l_good < 0.02 and l_bad > l_good + 0.3
