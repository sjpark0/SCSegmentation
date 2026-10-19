#!/usr/bin/env python3
"""Reference-camera prompt for the library cart take (docs/cart.md), SAM 3 image model, scsam3 container (GPU).

    survey      SAM 3 text "person" (confidence 0.5) on the reference camera every --step frames: instance count and
                boxes -> <out>/survey.json (is anyone besides the five people ever in view? when is the cart puller
                fully in view?)
    candidates  frame F of the reference camera: every instance for the text prompts in PROMPTS -> candidates.json
                + one overlay per prompt (numbered boxes), for choosing the objects by eye
    build       frame F: the objects in --spec (JSON: [{"id", "name", "prompt", "pick": [x, y]} or
                {"id", "name", "box": [x0, y0, x1, y1]} or {"id", "name", "points": [[x, y, label], ...]}], each
                optionally with "add": [{"prompt", "picks"|"near", "max_area"} | {"box", "points"}] merged in and
                "fill_holes"), painted in --spec order (later objects win overlaps) -> <out>/cam_ref.png (ids),
                overlay, report
"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "PlenopticServer2", "GSProject", "GSData", "library_261014", "cart", "png")
PROMPTS = ["person", "book cart", "cart", "table", "desk", "book", "laptop"]
COLS = [(0, 0, 0), (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
        (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255)]


def load(cam, f):
    return Image.open(os.path.join(SRC, cam, f"{f:06d}.png")).convert("RGB")


class Sam3:
    def __init__(self, conf=0.5, interactive=False):
        import sam3
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        bpe = os.path.join(os.path.dirname(sam3.__file__), "assets", "bpe_simple_vocab_16e6.txt.gz")
        self.model = build_sam3_image_model(bpe_path=bpe, enable_inst_interactivity=interactive)
        self.proc = Sam3Processor(self.model, confidence_threshold=conf)

    def text(self, img, prompt):
        st = self.proc.set_image(img)
        st = self.proc.set_text_prompt(prompt=prompt, state=st)
        m, s = st["masks"], st["scores"]
        m = (m.float().cpu().numpy() > 0.5 if hasattr(m, "cpu") else np.asarray(m))     # bf16 under interactivity
        s = (s.float().cpu().numpy() if hasattr(s, "cpu") else np.asarray(s)).reshape(-1)
        W, H = img.size
        m = m.reshape(-1, m.shape[-2], m.shape[-1]).astype(bool)
        if len(m) and m.shape[1:] != (H, W):
            m = np.stack([np.array(Image.fromarray(x).resize((W, H), Image.NEAREST)) for x in m])
        return m.reshape(-1, H, W), s

    def inst(self, img, points=None, box=None):
        st = self.proc.set_image(img)
        kw = dict(multimask_output=True)
        if points is not None:
            kw.update(point_coords=np.array([[x, y] for x, y, _ in points], np.float32),
                      point_labels=np.array([l for _, _, l in points], np.int32))
        if box is not None:
            kw.update(box=np.array(box, np.float32))
        m, sc, _ = self.model.predict_inst(st, **kw)
        W, H = img.size
        m = np.asarray(m).reshape(-1, H, W).astype(bool)
        sc = np.asarray(sc).reshape(-1)
        k = int(sc.argmax())
        return m[k], float(sc[k])


def bbox(m):
    ys, xs = np.nonzero(m)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None


def box_iou(a, b):
    if a is None:
        return 0.0
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u else 0.0


def overlay(img, masks, labels, path, alpha=0.55):
    a = np.asarray(img).astype(np.float32)
    for i, m in enumerate(masks):
        c = np.array(COLS[1 + i % (len(COLS) - 1)], np.float32)
        a[m] = (1 - alpha) * a[m] + alpha * c
    im = Image.fromarray(a.astype(np.uint8))
    d = ImageDraw.Draw(im)
    for i, (m, lab) in enumerate(zip(masks, labels)):
        b = bbox(m)
        if b:
            d.rectangle(b, outline=COLS[1 + i % (len(COLS) - 1)], width=3)
            d.text((b[0] + 4, b[1] + 2), lab, fill=(255, 255, 255))
    im.save(path, quality=88)


def survey(a):
    s3 = Sam3()
    rec = {}
    for cam in a.cams.split(","):
        for f in range(a.first, a.last + 1, a.step):
            m, s = s3.text(load(cam, f), "person")
            rec.setdefault(cam, {})[f] = [dict(score=round(float(x), 3), box=bbox(y), area=int(y.sum())) for y, x in zip(m, s)]
            print(cam, f, len(m), [b["box"] for b in rec[cam][f]], flush=True)
    json.dump(rec, open(os.path.join(a.out, "survey.json"), "w"), indent=1)


def candidates(a):
    s3 = Sam3()
    img = load(a.cam, a.frame)
    rec = {}
    for p in (a.prompts.split(",") if a.prompts else PROMPTS):
        m, s = s3.text(img, p)
        order = np.argsort(-s)
        rec[p] = [dict(k=int(k), score=round(float(s[k]), 3), box=bbox(m[k]), area=int(m[k].sum())) for k in order]
        overlay(img, [m[k] for k in order[:40]], [f"{p[:4]}{j}:{s[k]:.2f}" for j, k in enumerate(order[:40])],
                os.path.join(a.out, f"cand_{p.replace(' ', '_')}.jpg"))
        np.save(os.path.join(a.out, f"cand_{p.replace(' ', '_')}.npy"), np.packbits(m[order], axis=-1))
        print(p, len(m), [(r["score"], r["box"]) for r in rec[p][:12]], flush=True)
    json.dump(rec, open(os.path.join(a.out, "candidates.json"), "w"), indent=1)


def build(a):
    spec = json.load(open(a.spec))
    s3 = Sam3(interactive=True)
    img = load(a.cam, a.frame)
    W, H = img.size
    ids = np.zeros((H, W), np.uint8)
    rep, masks, labs = [], [], []
    cache = {}
    for o in spec:
        if "prompt" in o:
            if o["prompt"] not in cache:
                cache[o["prompt"]] = s3.text(img, o["prompt"])
            m, s = cache[o["prompt"]]
            if "match_box" in o:                           # the instance whose box overlaps this box most
                iou = [box_iou(bbox(m[k]), o["match_box"]) if m[k].any() else 0.0 for k in range(len(m))]
                if not iou or max(iou) < 0.5:
                    raise SystemExit(f"{o['name']}: no '{o['prompt']}' instance box near {o['match_box']}")
                k = int(np.argmax(iou))
            else:
                x, y = o["pick"]
                hit = [k for k in range(len(m)) if m[k][y, x]]
                if not hit:
                    raise SystemExit(f"{o['name']}: no '{o['prompt']}' instance covers {o['pick']}")
                k = max(hit, key=lambda k: s[k])
            mk, sc, how = m[k], float(s[k]), f"text '{o['prompt']}' instance {k}"
        elif "box" in o:
            mk, sc = s3.inst(img, box=o["box"], points=o.get("points"))
            how = f"box {o['box']}" + (f" + points {o['points']}" if o.get("points") else "")
        else:
            mk, sc = s3.inst(img, points=o["points"])
            how = f"points {o['points']}"
        for ad in o.get("add", []):                        # merge more instances into this object (table-top items...)
            if "prompt" in ad:
                if ad["prompt"] not in cache:
                    cache[ad["prompt"]] = s3.text(img, ad["prompt"])
                m2, s2 = cache[ad["prompt"]]
                near = binary_dilation(mk, iterations=ad["near"]) if "near" in ad else None
                for k in range(len(m2)):
                    if "picks" in ad and not any(m2[k][y, x] for x, y in ad["picks"]):
                        continue
                    if "match_boxes" in ad and not any(box_iou(bbox(m2[k]), b) >= 0.5 for b in ad["match_boxes"]):
                        continue
                    if near is not None and not (m2[k] & near).any():
                        continue
                    if m2[k].sum() > ad.get("max_area", 10 ** 9):
                        continue
                    mk = mk | m2[k]
                    how += f" + '{ad['prompt']}' {k}"
            else:
                m2, _ = s3.inst(img, box=ad["box"], points=ad.get("points"))
                mk = mk | m2
                how += f" + box {ad['box']}"
        if o.get("close"):                                 # bridge thin gaps so enclosed items count as holes
            mk = mk | binary_closing(mk, iterations=o["close"])
            how += f" + closed {o['close']}"
        if o.get("fill_holes"):
            mk = binary_fill_holes(mk)
            how += " + holes filled"
        for r in o.get("minus", []):                       # remove pixels claimed by an earlier object id
            mk &= ids != r
        ids[mk] = o["id"]
        rep.append(dict(id=o["id"], name=o["name"], how=how, score=round(sc, 3), area=int(mk.sum()), box=bbox(mk)))
        print(rep[-1], flush=True)
    for o in rep:
        o["area_final"] = int((ids == o["id"]).sum())
        masks.append(ids == o["id"])
        labs.append(f"{o['id']} {o['name']}")
    os.makedirs(a.out, exist_ok=True)
    Image.fromarray(ids, mode="L").save(os.path.join(a.out, "cam_ref.png"))
    overlay(img, masks, labs, os.path.join(a.out, "cam_ref_overlay.jpg"))
    json.dump(dict(cam=a.cam, frame=a.frame, objects=rep), open(os.path.join(a.out, "cam_ref.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("survey", "candidates", "build"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--cam", default="cam0005")
    ap.add_argument("--cams", default="cam0005")
    ap.add_argument("--frame", type=int, default=120)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, default=2699)
    ap.add_argument("--step", type=int, default=50)
    ap.add_argument("--spec")
    ap.add_argument("--prompts", help="candidates: comma-separated text prompts (default PROMPTS)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dict(survey=survey, candidates=candidates, build=build)[a.cmd](a)


if __name__ == "__main__":
    main()
