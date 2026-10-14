"""Losses (spec 4.3-4.4).  Same formulas as the SAM 2 tracking losses, implemented here so they run
on CPU in tests (the upstream focal loss is Triton-only).  All functions take logits."""
import torch
import torch.nn.functional as F

from geomem.sam3_io import logits_to_original
from geomem.warp_torch import sample_reference

W_FOCAL, W_DICE, W_IOU, W_PRESENCE = 20.0, 1.0, 1.0, 1.0
W_SSL = 0.5
W_BAND, R_BAND = 1.0, 4      # S1-E9 D2 (docs/stage1-E9-prereg.md section 2): BCE on the GT boundary band, +-R_BAND px at original resolution


def boundary_band(targets, r=R_BAND):
    """[B,1,h,w] {0,1} float -> [B,1,h,w] bool: dilate(GT, r) \\ erode(GT, r) with a (2r+1)^2 square (max-pool)."""
    k = 2 * int(r) + 1
    dil = F.max_pool2d(targets, k, stride=1, padding=k // 2)
    ero = 1.0 - F.max_pool2d(1.0 - targets, k, stride=1, padding=k // 2)
    return (dil > 0.5) & (ero < 0.5)


def band_bce(logits, targets, r=R_BAND):
    """Mean BCE over the boundary-band pixels of each object (0 for an object without a band, i.e. absent), averaged
    over objects.  logits/targets [B,1,h,w]."""
    band = boundary_band(targets, r).float()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    n = band.flatten(1).sum(1)
    per = (ce * band).flatten(1).sum(1) / n.clamp(min=1.0)
    return per.mean()


def sigmoid_focal(logits, targets, alpha=0.25, gamma=2.0):
    """Mean over pixels of the focal loss, averaged over objects.  logits/targets [B,1,h,w]."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * (1 - p_t) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss
    return loss.flatten(1).mean(1).mean()


def dice(logits, targets, eps=1.0):
    p = torch.sigmoid(logits).flatten(1)
    t = targets.flatten(1)
    num = 2 * (p * t).sum(1) + eps
    den = p.sum(1) + t.sum(1) + eps
    return (1 - num / den).mean()


def actual_iou(logits, targets):
    pm = (logits.flatten(1) > 0)
    gm = (targets.flatten(1) > 0.5)
    inter = (pm & gm).sum(1).float()
    union = (pm | gm).sum(1).float()
    return inter / union.clamp(min=1.0)


def iou_l1(logits, targets, pred_iou):
    return F.l1_loss(pred_iou, actual_iou(logits, targets).to(pred_iou.dtype))


def presence_bce(presence_logits, present_bool):
    return F.binary_cross_entropy_with_logits(presence_logits.float(), present_bool.float())


def supervised_loss(out, tgt_masks, band=False):
    """out from GeoMemStar.decode_view, tgt_masks [B,h,w] bool -> (total, {term: float}).  `band` (S1-E9 D2) adds
    W_BAND * band_bce; the "band" term is reported either way (0.0 when off)."""
    hw = tuple(tgt_masks.shape[-2:])
    logits = logits_to_original(out["logits_1008"], hw).float()
    targets = tgt_masks.unsqueeze(1).float()
    terms = {"focal": sigmoid_focal(logits, targets), "dice": dice(logits, targets),
             "iou": iou_l1(logits, targets, out["ious"].float()),
             "presence": presence_bce(out["presence"], tgt_masks.flatten(1).any(1))}
    total = W_FOCAL * terms["focal"] + W_DICE * terms["dice"] + W_IOU * terms["iou"] + W_PRESENCE * terms["presence"]
    if band:
        terms["band"] = band_bce(logits, targets)
        total = total + W_BAND * terms["band"]
    else:
        terms["band"] = torch.zeros(())
    return total, {k: float(v.detach()) for k, v in terms.items()}


def reprojection_consistency(logits_u, gt_ref, grid, valid, eps=1.0):
    """Soft dice between sigmoid(logits_u) and the reference GT sampled at the reprojected positions, on valid pixels.

    logits_u [B,1,h,w], gt_ref [B,h_ref,w_ref] bool, grid [1,h,w,2], valid [h,w] bool -> scalar (mean over objects)."""
    p = torch.sigmoid(logits_u.float())[:, 0]                              # [B,h,w]
    g = sample_reference(gt_ref.unsqueeze(1).float(), grid)[:, 0]           # [B,h,w]
    m = valid.to(p.dtype)[None]
    num = 2 * (p * g * m).flatten(1).sum(1) + eps
    den = ((p + g) * m).flatten(1).sum(1) + eps
    return (1 - num / den).mean()
