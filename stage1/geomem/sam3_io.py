"""Image loading and backbone-feature plumbing for the SAM 3 tracker, used by the wrapper and the caches.

The tracker is built without its own backbone; the video model's detector backbone produces the
"sam2_backbone_out" the tracker expects, after which the tracker's `forward_image` post-processing
(conv_s0 / conv_s1 on the two high-resolution levels) is applied here exactly as the base does.
Runs inside `scsam3`.
"""
import torch
import torch.nn.functional as F

from methods import sam3_single


def load_image(path):
    """JPEG -> (image [1,3,1008,1008] float32 cuda, h, w) with the package's own transform."""
    frame, h, w = sam3_single.load_frame(path)
    img = frame.cuda().float()
    if img.dim() == 3:
        img = img.unsqueeze(0)
    return img, h, w


@torch.no_grad()
def backbone_features(model, image):
    """image [1,3,1008,1008] -> dict(backbone_fpn: [3 maps], vision_pos_enc: [3 maps]) for the tracker."""
    tracker = model.tracker
    out = model.detector.backbone.forward_image(image)["sam2_backbone_out"]
    fpn = list(out["backbone_fpn"])
    fpn[0] = tracker.sam_mask_decoder.conv_s0(fpn[0])
    fpn[1] = tracker.sam_mask_decoder.conv_s1(fpn[1])
    return {"backbone_fpn": [f.detach() for f in fpn],
            "vision_pos_enc": [p.detach() for p in out["vision_pos_enc"]]}


def expand_features(tracker, backbone_out, B):
    """Batch-expand the cached maps to B objects and flatten them the way the tracker consumes them."""
    exp = {"backbone_fpn": [f.expand(B, -1, -1, -1) for f in backbone_out["backbone_fpn"]],
           "vision_pos_enc": [p.expand(B, -1, -1, -1) for p in backbone_out["vision_pos_enc"]]}
    _, vision_feats, vision_pos, feat_sizes = tracker._prepare_backbone_features(exp)
    return vision_feats, vision_pos, feat_sizes


def mask_inputs(masks_bool, size=1008):
    """[B,h,w] bool -> [B,1,size,size] float (nearest), the tracker's mask-prompt format."""
    m = masks_bool.float().unsqueeze(1)
    return F.interpolate(m, size=(size, size), mode="nearest")


def logits_to_original(high_res, hw):
    """[B,1,1008,1008] logits -> [B,1,h,w] (bilinear, as the tracker's video-resolution output)."""
    return F.interpolate(high_res.float(), size=tuple(hw), mode="bilinear", align_corners=False)
