#!/usr/bin/env python3
"""S3-P P2 (docs/plenoptic-P-prereg.md section 1): the reference camera's t=0 mask from a prompt instead
of the ground truth, with the frozen SAM 3 image model.

    text   Sam3Processor(confidence_threshold=0.5).set_text_prompt("person") -> instances; Hungarian match
           to the GT persons by IoU, pairs with IoU >= 0.5 keep the GT id; unmatched instances dropped,
           unmatched persons left empty; overlapping pixels go to the higher-scoring instance.
    point  one positive click per person at the GT mask's inner-most pixel (distance-transform maximum),
           interactive predictor with multimask_output=True, highest predicted score kept.

Writes the variant scenes under Data/Plenoptic/<scene>_{txt,pt}/ (Video = relative symlink to the shared
copy, Mask/ = the shared GT with cam_06/000000.png replaced) and a report JSON.  GPU, scsam3 container:

    python plenoptic/prompt_seed.py --mode text
    python plenoptic/prompt_seed.py --mode point
"""
import argparse, hashlib, json, os, shutil, sys
import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARE = os.path.join(REPO, "PlenopticServer1", "CROOM", "SegResult")
SCENE, REF = "T06_flag_ganggang", "cam_06"
LOCAL = os.path.join(REPO, "Data", "Plenoptic")
CONF, MATCH_IOU = 0.5, 0.5


def iou(a, b):
    u = np.logical_or(a, b).sum(); return float(np.logical_and(a, b).sum() / u) if u else 0.0


def hungarian(M):
    """max-weight assignment on a small matrix (rows = instances, cols = persons); brute force is fine here."""
    from itertools import permutations
    r, c = M.shape; best, best_p = -1.0, None
    if r == 0 or c == 0: return []
    if r <= c:
        for p in permutations(range(c), r):
            s = sum(M[i, p[i]] for i in range(r))
            if s > best: best, best_p = s, p
        return [(i, best_p[i]) for i in range(r)]
    for p in permutations(range(r), c):
        s = sum(M[p[j], j] for j in range(c))
        if s > best: best, best_p = s, p
    return [(best_p[j], j) for j in range(c)]


def inner_point(mask):
    from scipy.ndimage import distance_transform_edt
    d = distance_transform_edt(mask); y, x = np.unravel_index(int(d.argmax()), d.shape); return int(x), int(y)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["text", "point"], required=True); a = ap.parse_args()
    import torch, sam3
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    bpe = os.path.join(os.path.dirname(sam3.__file__), "..", "assets", "bpe_simple_vocab_16e6.txt.gz")

    src_scene = os.path.join(SHARE, SCENE)
    img = Image.open(os.path.join(src_scene, "Video", REF, "000000.jpg")).convert("RGB")
    gt = np.array(Image.open(os.path.join(src_scene, "Mask", REF, "000000.png")))
    ids = [i for i in range(1, 7) if (gt == i).any()]
    H, W = gt.shape
    out = np.zeros((H, W), dtype=np.uint8); report = {"mode": a.mode, "ref": REF, "persons": {}}

    if a.mode == "text":
        model = build_sam3_image_model(bpe_path=bpe)
        proc = Sam3Processor(model, confidence_threshold=CONF)
        st = proc.set_image(img); st = proc.set_text_prompt(prompt="person", state=st)
        masks = st["masks"]; scores = st["scores"]
        masks = masks.cpu().numpy() if hasattr(masks, "cpu") else np.asarray(masks)
        scores = scores.cpu().numpy() if hasattr(scores, "cpu") else np.asarray(scores)
        masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1]).astype(bool)
        if masks.shape[1:] != (H, W):                      # resize to the frame if the processor returns model resolution
            masks = np.stack([np.array(Image.fromarray(m).resize((W, H), Image.NEAREST)) for m in masks]) if len(masks) else masks.reshape(0, H, W)
        M = np.array([[iou(masks[i], gt == p) for p in ids] for i in range(len(masks))]).reshape(len(masks), len(ids))
        pairs = [(i, j) for i, j in hungarian(M) if M[i, j] >= MATCH_IOU]
        report.update(n_instances=int(len(masks)), scores=[float(s) for s in scores], matched=len(pairs),
                      unmatched_instances=int(len(masks) - len(pairs)), unmatched_persons=[ids[j] for j in range(len(ids)) if j not in {j for _, j in pairs}])
        order = sorted(pairs, key=lambda ij: float(scores[ij[0]]))   # paint low score first, high score wins overlaps
        for i, j in order:
            out[masks[i]] = ids[j]; report["persons"][ids[j]] = dict(instance=int(i), score=float(scores[i]), iou_gt=float(M[i, j]))
    else:
        model = build_sam3_image_model(bpe_path=bpe, enable_inst_interactivity=True)
        pred = model.inst_interactive_predictor if hasattr(model, "inst_interactive_predictor") else None
        if pred is None:
            from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
            pred = SAM3InteractiveImagePredictor(model)
        pred.set_image(np.array(img))
        cands = []
        for p in ids:
            x, y = inner_point(gt == p)
            m, sc, _ = pred.predict(point_coords=np.array([[x, y]]), point_labels=np.array([1]), multimask_output=True)
            m = np.asarray(m).reshape(-1, H, W).astype(bool); sc = np.asarray(sc).reshape(-1)
            k = int(sc.argmax()); cands.append((p, m[k], float(sc[k]), (x, y)))
            report["persons"][p] = dict(click=[x, y], score=float(sc[k]), iou_gt=iou(m[k], gt == p))
        for p, m, sc, _ in sorted(cands, key=lambda c: c[2]):
            out[m] = p

    # variant scene: Video -> relative symlink to the shared copy, Mask copied with the reference replaced
    tag = {"text": "txt", "point": "pt"}[a.mode]
    var = os.path.join(LOCAL, f"{SCENE}_{tag}"); os.makedirs(var, exist_ok=True)
    link = os.path.join(var, "Video")
    if os.path.lexists(link): os.remove(link)
    os.symlink(os.path.relpath(os.path.join(src_scene, "Video"), var), link)
    if os.path.isdir(os.path.join(var, "Mask")): shutil.rmtree(os.path.join(var, "Mask"))
    shutil.copytree(os.path.join(src_scene, "Mask"), os.path.join(var, "Mask"))
    dst = os.path.join(var, "Mask", REF, "000000.png"); Image.fromarray(out, mode="L").save(dst)
    report["seed_png"] = dst; report["seed_sha256"] = hashlib.sha256(open(dst, "rb").read()).hexdigest()
    report["seed_J_vs_gt"] = {p: iou(out == p, gt == p) for p in ids}
    report["mean_seed_J"] = float(np.mean(list(report["seed_J_vs_gt"].values())))
    json.dump(report, open(os.path.join(REPO, "plenoptic", f"prompt_seed_{tag}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "persons"}, ensure_ascii=False))
    for p, v in report["persons"].items(): print(" person", p, v)


if __name__ == "__main__":
    main()
