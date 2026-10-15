#!/usr/bin/env python3
"""Protocol inventory, part 1 (host, read-only): which information sources exist on disk for each of the 17 scenes.

Reads only metadata / directory listings / objects_labels.json / the c_ini reference-frame label map (the protocol input).
No scoring-camera GT and no keyframe prediction is read.
"""
import collections, json, os, sys
REPO = "/home/sjpark/Documents/SCSegmentation"
RAW = os.path.join(REPO, "Data", "MVSeg")
S1 = os.path.join(REPO, "stage1", "Data")
CFG = json.load(open(os.path.join(REPO, "SCSam3", "demo", "MVSeg.json")))
SETS = json.load(open(os.path.join(REPO, "docs", "raw", "muvod_object_sets.json")))
scenes = sorted(d for d in os.listdir(S1) if os.path.isfile(os.path.join(S1, d, "meta.json")))
out = {}
tot = collections.Counter()
for s in scenes:
    meta = json.load(open(os.path.join(S1, s, "meta.json")))
    ref, c = meta["ref_frame"], meta["ref_cam"]
    vcams = sorted(os.listdir(os.path.join(RAW, s, "Video")))
    nfr = {v: len([f for f in os.listdir(os.path.join(RAW, s, "Video", v)) if f.endswith(".jpg")]) for v in (c,)}
    vfr = sorted(int(f[:-4]) for f in os.listdir(os.path.join(RAW, s, "Video", c)) if f.endswith(".jpg"))
    mcams = sorted(d for d in os.listdir(os.path.join(RAW, s, "Mask")) if os.path.isdir(os.path.join(RAW, s, "Mask", d)))
    mfr = sorted(int(f[:-4]) for f in os.listdir(os.path.join(RAW, s, "Mask", c)) if f.endswith(".png"))
    labels = json.load(open(os.path.join(RAW, s, "Mask", "objects_labels.json")))
    lab_s1 = json.load(open(os.path.join(S1, s, "objects_labels.json")))
    assert labels == lab_s1, s
    seed_ids = [int(i) for i in SETS[s]["seed_ids_c_ini"]]
    names = collections.Counter(v["label"] for v in labels.values())
    basic_names = collections.Counter(labels[str(i)]["label"] for i in seed_ids if str(i) in labels)
    types = collections.Counter(v["type"] for v in labels.values())
    # objects of the basic set whose label is shared with >= 1 other labelled object of the scene
    shared_basic = sum(1 for i in seed_ids if str(i) in labels and names[labels[str(i)]["label"]] > 1)
    pf = os.path.join(S1, s, "Cache", "pi3_frames")
    pi3_frames = sorted(os.listdir(pf)) if os.path.isdir(pf) else []
    pi3_cover = all(set(json.load(open(os.path.join(pf, t, "MANIFEST.json")))["cams"]) == set(meta["cams"]) and
                    all(os.path.isfile(os.path.join(pf, t, f"{v}.npz")) for v in meta["cams"]) for t in pi3_frames)
    sf = os.path.join(S1, s, "Cache", "sam3_feats")
    feats = {t: len(os.listdir(os.path.join(sf, t))) for t in sorted(os.listdir(sf)) if os.path.isdir(os.path.join(sf, t))} if os.path.isdir(sf) else {}
    feats_allcam_frames = sum(1 for t, n in feats.items() if n >= len(meta["cams"]))
    rec = {
        "ref_frame": ref, "c_ini": c, "rig_cams_video": len(vcams), "stage1_cams": len(meta["cams"]),
        "video_frames_c_ini": len(vfr), "video_first_last": [vfr[0], vfr[-1]],
        "annotated_window": [mfr[0], mfr[-1]], "n_annotated_frames": len(mfr),
        "c_ini_rgb_frames_before_ref": sum(1 for f in vfr if f < ref),
        "c_ini_rgb_frames_after_ref_in_window": sum(1 for f in mfr if f > ref),
        "c_ini_rgb_frames_after_window": sum(1 for f in vfr if f > mfr[-1]),
        "mask_cams": mcams, "n_labelled_objects": len(labels), "n_basic_objects": len(seed_ids),
        "n_complete_only_objects": len(labels) - len(seed_ids),
        "n_distinct_label_names": len(names), "types": dict(types),
        "basic_objects_with_shared_label": shared_basic,
        "pi3_frames": len(pi3_frames), "pi3_frames_all_cams": pi3_cover,
        "sam3_feats_frames": len(feats), "sam3_feats_frames_all_cams": feats_allcam_frames,
        "label_names": dict(names),
    }
    out[s] = rec
    tot["rig_cams"] += len(vcams); tot["labelled"] += len(labels); tot["basic"] += len(seed_ids)
    tot["shared_basic"] += shared_basic; tot["complete_only"] += len(labels) - len(seed_ids)
    for k, v in types.items():
        tot["type_" + k] += v
allnames = collections.Counter()
for s in out:
    allnames.update(out[s]["label_names"].keys())
print(f"{'scene':20s} ref  c_ini        rig s1  vid  win      bef aftW aftV  lab bas cmpl names shrdB  pi3fr allc sam3fr allc")
for s, r in out.items():
    print(f"{s:20s} {r['ref_frame']:3d}  {r['c_ini']:12s} {r['rig_cams_video']:3d} {r['stage1_cams']:3d} {r['video_frames_c_ini']:4d} "
          f"{r['annotated_window'][0]:3d}-{r['annotated_window'][1]:3d} {r['c_ini_rgb_frames_before_ref']:4d} {r['c_ini_rgb_frames_after_ref_in_window']:4d} "
          f"{r['c_ini_rgb_frames_after_window']:4d} {r['n_labelled_objects']:4d} {r['n_basic_objects']:3d} {r['n_complete_only_objects']:4d} "
          f"{r['n_distinct_label_names']:5d} {r['basic_objects_with_shared_label']:5d}  {r['pi3_frames']:5d} {str(r['pi3_frames_all_cams']):5s} "
          f"{r['sam3_feats_frames']:5d} {r['sam3_feats_frames_all_cams']:4d}")
print("totals", dict(tot), "distinct label names over the 17 scenes:", len(allnames))
json.dump({"scenes": out, "totals": dict(tot), "distinct_names": sorted(allnames)}, open(sys.argv[1], "w"), indent=1)
