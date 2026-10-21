#!/usr/bin/env bash
# Cart results -> NAS (docs/cart.md; user: "당연히 나스에 저장해줘").  Waits for Data/Cart/done/render_f120, copies the
# results with parallel rsync, verifies (PNG trees by content digest, every other file by sha256), then removes the local
# result copies (the source JPEGs in Data/Cart/src stay).  Writes Data/Cart/done/sync_f120.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
C="${ROOT}/Data/Cart"; KEY=cart_f120; M=SegMaskSam3XW1CGPS4MFsCs10kG2
NAS="${NAS:-${ROOT}/PlenopticServer2/SegProject/Data/Cart}"
ts() { date '+%F %T'; }
until [[ -f "${C}/done/render_f120" ]]; do sleep 60; done
echo "$(ts) sync to ${NAS}"
mkdir -p "${NAS}/${KEY}" "${NAS}/labels"
cp "${C}/cart.json" "${NAS}/"
rsync -rt "${C}/review" "${NAS}/"
rsync -rt "${C}/${KEY}/Mask" "${C}/${KEY}/SOURCE.json" "${NAS}/${KEY}/"
for tree in "${M}" export/labels export/fg viz/frames; do
  mkdir -p "${NAS}/${KEY}/${tree}"
  (cd "${C}/${KEY}/${tree}" && ls -d cam* | xargs -P 16 -I{} rsync -rt {} "${NAS}/${KEY}/${tree}/")
done
rsync -t "${C}/${KEY}/${M}"/MANIFEST*.json "${NAS}/${KEY}/${M}/"
rsync -t "${C}/${KEY}/export/export.json" "${NAS}/${KEY}/export/"
rsync -rt "${C}/${KEY}/viz/cams" "${C}/${KEY}/viz/mosaic.mp4" "${NAS}/${KEY}/viz/"
rsync -rt "${C}/labels/seg_labels_cart_f120_2699" "${NAS}/labels/"
echo "$(ts) copied; verifying"
EVAL_DIR="${ROOT}/eval" PYTHONDONTWRITEBYTECODE=1 python3 - "${C}" "${NAS}" "${KEY}" "${M}" <<'EOF'
import hashlib, os, sys
sys.path.insert(0, os.environ["EVAL_DIR"])
import manifest as Mf
C, N, KEY, M = sys.argv[1:]
bad = []
a, b = Mf.content_digest(os.path.join(C, KEY, M), jobs=16), Mf.content_digest(os.path.join(N, KEY, M), jobs=16)
print("masks", a[0] == b[0], len(a[1]), len(b[1]), flush=True)
if a[0] != b[0] or len(a[1]) != len(b[1]):
    bad.append("masks")
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()
for sub in (os.path.join(KEY, "export"), os.path.join(KEY, "viz"), "labels"):
    n = 0
    for dp, _, fs in os.walk(os.path.join(C, sub)):
        for f in fs:
            p = os.path.join(dp, f)
            q = os.path.join(N, os.path.relpath(p, C))
            if not os.path.isfile(q) or os.path.getsize(p) != os.path.getsize(q) or (f.endswith((".npy", ".json", ".mp4")) and sha(p) != sha(q)):
                bad.append(os.path.relpath(p, C))
            n += 1
    print(sub, n, "files checked", flush=True)
print("mismatches:", len(bad), bad[:10])
sys.exit(1 if bad else 0)
EOF
st=$?
[[ ${st} -eq 0 ]] || { echo "$(ts) ABORT: NAS copy differs; local copy kept"; exit 1; }
touch "${C}/done/sync_f120"
rm -rf "${C}/${KEY}/${M}" "${C}/${KEY}/export" "${C}/${KEY}/viz" "${C}/labels"
echo "$(ts) verified; local result copies removed (source JPEGs kept in ${C}/src)"
