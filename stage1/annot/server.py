#!/usr/bin/env python3
"""CROOM ring ground-truth annotation tool (iccv2027-plan.md WP3 "필수 추가: CROOM 링 새 정답"; docs/croom-annotation.md).

Per-view, SAM 3 single-image assisted annotation: the annotator clicks / boxes one object in one camera, SAM 3 (image
model, SAM-1-style point / box prompts, no text, no cross-view propagation) proposes a mask, the annotator fixes it
with a brush and saves it under an object id.  Nothing is propagated between cameras: the identity of an object in
every camera is the annotator's decision.  Every prompt, brush stroke and save is logged (log.jsonl) with a timestamp.

Layout (prepared by annot/prep_croom.py):
  <root>/<take>/<frame:06d>/images/<cam>.jpg        the 32 camera images of one instant (copied from the NAS)
  <root>/<take>/<frame:06d>/annot/objects.json      {id: {"name": ..., "color": [r,g,b]}}
  <root>/<take>/<frame:06d>/annot/labels/<cam>.png  uint8 label map (0 background, id = object), written on every save
  <root>/<take>/<frame:06d>/annot/log.jsonl         events

Run (GPU, scsam3; the NAS is not needed once prep_croom.py copied the images):
  scripts/docker_run.sh scsam3 python -m annot.server --root /host$PWD/Data/_annot/CROOM --port 7861
  (add `-p 7861:7861` to docker run; docker_run.sh does not publish ports -- see annot/run_server.sh)
Then open http://localhost:7861 .
"""
import argparse
import base64
import io
import json
import os
import sys
import threading
import time

import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = os.path.dirname(HERE)
sys.path.insert(0, S1)
sys.path.insert(0, os.path.join(os.path.dirname(S1), "cart"))

app = Flask(__name__, static_folder=os.path.join(HERE, "static"), static_url_path="/static")
ROOT = None
LOCK = threading.Lock()
PREVIEW_W = 1920                       # display size served to the browser (the label maps stay full resolution)
COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
          (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200),
          (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128)]


class Sam:
    """SAM 3 image model with SAM-1-style prompts; the image embedding is cached per image path."""

    def __init__(self):
        import seed_cart as SC
        self.sam = SC.Sam3(conf=0.3, interactive=True)
        self.cache = {}

    def state(self, path):
        if self.cache.get("path") != path:
            img = Image.open(path).convert("RGB")
            self.cache = dict(path=path, img=img, st=self.sam.proc.set_image(img), size=img.size)
        return self.cache

    def predict(self, path, points, box):
        c = self.state(path)
        kw = dict(multimask_output=True)
        if points:
            kw.update(point_coords=np.array([[x, y] for x, y, _ in points], np.float32),
                      point_labels=np.array([l for _, _, l in points], np.int32))
        if box:
            kw.update(box=np.array(box, np.float32))
        m, sc, _ = self.sam.model.predict_inst(c["st"], **kw)
        W, H = c["size"]
        m = np.asarray(m).reshape(-1, H, W).astype(bool)
        sc = np.asarray(sc).reshape(-1)
        k = int(sc.argmax())
        return m[k], float(sc[k])


SAM = None


def inst_dir(take, frame):
    return os.path.join(ROOT, take, f"{int(frame):06d}")


def annot_dir(take, frame):
    d = os.path.join(inst_dir(take, frame), "annot")
    os.makedirs(os.path.join(d, "labels"), exist_ok=True)
    return d


def cams_of(take, frame):
    d = os.path.join(inst_dir(take, frame), "images")
    return sorted((f[:-4] for f in os.listdir(d) if f.endswith(".jpg")), key=lambda c: (len(c), c))


def objects_of(take, frame):
    p = os.path.join(annot_dir(take, frame), "objects.json")
    return json.load(open(p)) if os.path.isfile(p) else {}


def log(take, frame, **ev):
    with open(os.path.join(annot_dir(take, frame), "log.jsonl"), "a") as f:
        f.write(json.dumps(dict(t=time.time(), **ev)) + "\n")


def label_path(take, frame, cam):
    return os.path.join(annot_dir(take, frame), "labels", f"{cam}.png")


def load_label(take, frame, cam, hw):
    p = label_path(take, frame, cam)
    if os.path.isfile(p):
        return np.array(Image.open(p))
    return np.zeros(hw, np.uint8)


def png_b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", compress_level=1)
    return base64.b64encode(buf.getvalue()).decode()


def b64_png(s):
    return np.array(Image.open(io.BytesIO(base64.b64decode(s))))


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/instants")
def api_instants():
    out = []
    for take in sorted(os.listdir(ROOT)):
        td = os.path.join(ROOT, take)
        if not os.path.isdir(td):
            continue
        for fr in sorted(os.listdir(td)):
            if os.path.isdir(os.path.join(td, fr, "images")):
                cams = cams_of(take, fr)
                done = [c for c in cams if os.path.isfile(label_path(take, fr, c))]
                out.append(dict(take=take, frame=int(fr), n_cams=len(cams), n_done=len(done), n_objects=len(objects_of(take, fr))))
    return jsonify(out)


@app.route("/api/instant")
def api_instant():
    take, frame = request.args["take"], int(request.args["frame"])
    cams = cams_of(take, frame)
    img = Image.open(os.path.join(inst_dir(take, frame), "images", f"{cams[0]}.jpg"))
    W, H = img.size
    status = {}
    for c in cams:
        p = label_path(take, frame, c)
        if os.path.isfile(p):
            lab = np.array(Image.open(p))
            status[c] = sorted(int(x) for x in np.unique(lab) if x)
    return jsonify(dict(take=take, frame=frame, cams=cams, size=[W, H], objects=objects_of(take, frame), status=status,
                        preview_w=min(PREVIEW_W, W)))


@app.route("/api/objects", methods=["POST"])
def api_objects():
    d = request.json
    take, frame = d["take"], int(d["frame"])
    objs = objects_of(take, frame)
    if d.get("add"):
        oid = max([int(k) for k in objs] + [0]) + 1
        objs[str(oid)] = dict(name=d["add"], color=COLORS[(oid - 1) % len(COLORS)])
        log(take, frame, ev="add_object", id=oid, name=d["add"])
    if d.get("rename"):
        objs[str(d["rename"]["id"])]["name"] = d["rename"]["name"]
        log(take, frame, ev="rename_object", **d["rename"])
    with open(os.path.join(annot_dir(take, frame), "objects.json"), "w") as f:
        json.dump(objs, f, indent=1)
    return jsonify(objs)


@app.route("/img/<take>/<int:frame>/<cam>")
def img(take, frame, cam):
    """The camera image downscaled to PREVIEW_W (cached as JPEG next to the original)."""
    src = os.path.join(inst_dir(take, frame), "images", f"{cam}.jpg")
    cache = os.path.join(inst_dir(take, frame), "preview", f"{cam}.jpg")
    if not os.path.isfile(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        im = Image.open(src)
        W, H = im.size
        if W > PREVIEW_W:
            im = im.resize((PREVIEW_W, round(H * PREVIEW_W / W)), Image.LANCZOS)
        im.save(cache, quality=90)
    return send_file(cache, mimetype="image/jpeg")


@app.route("/api/label")
def api_label():
    take, frame, cam = request.args["take"], int(request.args["frame"]), request.args["cam"]
    img = Image.open(os.path.join(inst_dir(take, frame), "images", f"{cam}.jpg"))
    W, H = img.size
    return jsonify(dict(png=png_b64(load_label(take, frame, cam, (H, W))), size=[W, H]))


@app.route("/api/sam", methods=["POST"])
def api_sam():
    global SAM
    d = request.json
    take, frame, cam = d["take"], int(d["frame"]), d["cam"]
    path = os.path.join(inst_dir(take, frame), "images", f"{cam}.jpg")
    with LOCK:
        if SAM is None:
            SAM = Sam()
        t0 = time.time()
        m, score = SAM.predict(path, d.get("points") or None, d.get("box") or None)
    log(take, frame, ev="sam", cam=cam, obj=d.get("obj"), points=d.get("points"), box=d.get("box"), score=round(score, 4),
        area=int(m.sum()), seconds=round(time.time() - t0, 3))
    return jsonify(dict(png=png_b64(m.astype(np.uint8) * 255), score=score, area=int(m.sum())))


@app.route("/api/save", methods=["POST"])
def api_save():
    """Write one object's full-resolution mask into the camera's label map (the object's old pixels are cleared first;
    other objects' pixels are overwritten only where `overwrite` is true)."""
    d = request.json
    take, frame, cam, oid = d["take"], int(d["frame"]), d["cam"], int(d["obj"])
    m = b64_png(d["png"]) > 0
    with LOCK:
        lab = load_label(take, frame, cam, m.shape)
        assert lab.shape == m.shape, (lab.shape, m.shape)
        before = int((lab == oid).sum())
        lab[lab == oid] = 0
        taken = (lab > 0) & m
        if d.get("overwrite"):
            lab[m] = oid
        else:
            lab[m & (lab == 0)] = oid
        Image.fromarray(lab).save(label_path(take, frame, cam), compress_level=3)
    log(take, frame, ev="save", cam=cam, obj=oid, area=int((lab == oid).sum()), area_before=before,
        overlapped=int(taken.sum()), overwrite=bool(d.get("overwrite")), strokes=d.get("strokes"))
    return jsonify(dict(ok=True, area=int((lab == oid).sum()), overlapped=int(taken.sum()),
                        present=sorted(int(x) for x in np.unique(lab) if x)))


@app.route("/api/clear", methods=["POST"])
def api_clear():
    d = request.json
    take, frame, cam, oid = d["take"], int(d["frame"]), d["cam"], int(d["obj"])
    with LOCK:
        p = label_path(take, frame, cam)
        if os.path.isfile(p):
            lab = np.array(Image.open(p))
            lab[lab == oid] = 0
            Image.fromarray(lab).save(p, compress_level=3)
    log(take, frame, ev="clear", cam=cam, obj=oid)
    return jsonify(dict(ok=True))


@app.route("/api/mark_empty", methods=["POST"])
def api_mark_empty():
    """The object is not visible in this camera: record the decision (an empty mask is the same as no mask for scoring,
    but the log shows the annotator looked)."""
    d = request.json
    take, frame = d["take"], int(d["frame"])
    p = label_path(take, frame, d["cam"])
    if d.get("undo"):                                # the mark was a slip: log it; drop an all-zero label file so the camera
        log(take, frame, ev="not_visible_undo", cam=d["cam"], obj=int(d["obj"]))   # no longer counts as visited
        with LOCK:
            if os.path.isfile(p) and not np.array(Image.open(p)).any():
                os.remove(p)
        return jsonify(dict(ok=True, removed_empty_label=not os.path.isfile(p)))
    log(take, frame, ev="not_visible", cam=d["cam"], obj=int(d["obj"]))
    if not os.path.isfile(p):                       # make the camera count as visited
        img = Image.open(os.path.join(inst_dir(take, frame), "images", f"{d['cam']}.jpg"))
        W, H = img.size
        Image.fromarray(np.zeros((H, W), np.uint8)).save(p, compress_level=3)
    return jsonify(dict(ok=True))


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    ROOT = os.path.abspath(a.root)
    app.run(host=a.host, port=a.port, threaded=True)


if __name__ == "__main__":
    main()
