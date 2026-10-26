"""Collect the development-instant results (scores/*.json + rig diagnostics + relay/id3d bookkeeping of the AB arm) into
scripts/croom/runs/<name>.json.  Measurement only.

    python stage1/scripts/croom/summarize_dev.py --out scripts/croom/runs/T04_dev_summary.json --instant "T04 #1000" \
        --scenes CROOM_T04_001000_r00,... [--variant-suffix _t06]
"""
import argparse
import datetime
import json
import os

S1 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
W = os.path.join(S1, "geomem", "runs", "_croom")
ARM_KEYS = ("n_pairs", "J_all", "J_near", "J_far", "wrong_all", "wrong_near", "wrong_far", "empty_all", "fail_all", "swap_pattern")


def load(p):
    return json.load(open(p)) if os.path.isfile(p) else None


def one(scene):
    sc = load(os.path.join(W, "scores", f"{scene}.json"))
    rig = load(os.path.join(W, "rig", f"{scene}.pi3.ringorder.json")) or load(os.path.join(W, "rig", f"{scene}.pi3.json"))
    ab = os.path.join(S1, "Data", scene, "Result", "croom_AB")
    relay, conf = load(os.path.join(ab, "relay.json")), load(os.path.join(ab, "id3d_conf.json"))
    out = dict(scene=scene, ref_cam=sc["ref_cam"] if sc else None, objects=sc["objects"] if sc else None,
               ring_order_used=bool(sc and sc.get("ring_order")))
    if conf:
        out.update(rig_ok=conf["rig_ok"], rig_phi_median=round(relay["rig"]["phi_median"], 4) if relay and relay.get("rig") else None,
                   id3d_summary=conf["summary"])
    if relay:
        out.update(n_S=len(relay["S"]), n_F=len(relay["F"]), relay_active=relay["active"],
                   n_relay=len(relay["order"]) if relay["active"] else 0, n_unreachable=len(relay["unreachable"]))
    if rig:
        out.update(ring=dict(order=rig.get("ring_order", "index"), breaks_after=rig["ring"]["breaks_after"],
                             az_jumps_after=rig["ring"]["az_jumps_after"], ring_ok=rig["ring"]["ring_ok"]))
    if sc:
        out["arms"] = {a: {k: v for k, v in s.items() if k in ARM_KEYS} for a, s in sc["arms"].items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--instant", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--variant-suffix", default=None, dest="suffix", help="also collect <scene><suffix> as 'variant'")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    res = dict(instant=a.instant, date=str(datetime.date.today()), note=a.note,
               checkpoint="geoset_pred_ft fold 2 seed 0 (as T06)", refs={})
    for s in a.scenes.split(","):
        res["refs"][s] = one(s)
        if a.suffix and os.path.isdir(os.path.join(S1, "Data", s + a.suffix)):
            res["refs"][s]["variant" + a.suffix] = one(s + a.suffix)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    for s, r in res["refs"].items():
        print(s, "rig_ok", r.get("rig_ok"), "phi", r.get("rig_phi_median"), "relay", r.get("n_relay"),
              {k: round(v["J_all"], 3) for k, v in r.get("arms", {}).items()})
        v = r.get("variant" + (a.suffix or ""))
        if v:
            print("   " + a.suffix, "rig_ok", v.get("rig_ok"), "phi", v.get("rig_phi_median"), "relay", v.get("n_relay"),
                  {k: round(x["J_all"], 3) for k, x in v.get("arms", {}).items()})


if __name__ == "__main__":
    main()
