#!/usr/bin/env bash
# Background replacement of the whole cart take (docs/cart.md §6), after the overlay render: waits for
# Data/Cart/done/render_f120, composites every frame of all 24 cameras into the outdoor panorama (lythwood_terrace 16K,
# yaw 270; user 2026-10-21) with cart/bg_cart.py comp, writes Data/Cart/done/bg_f120.  Run it in a container
# (scsam3, --user uid:gid) so a session kill on the host cannot stop it:
#   docker run -d --name cart_bg --user U:G -e HOME=/tmp -v /:/host --entrypoint bash scsam3 /host<repo>/cart/bg_post.sh /host<repo>
set -uo pipefail
ROOT="$1"; C="${ROOT}/Data/Cart"; KEY=cart_f120; PANO=lythwood_terrace; YAW=270
OUT="${C}/${KEY}/bg/${PANO}_y${YAW}"
ts() { date '+%F %T'; }
until [[ -f "${C}/done/render_f120" ]]; do sleep 60; done
echo "$(ts) bg composite -> ${OUT}"
cd "${ROOT}" && PYTHONDONTWRITEBYTECODE=1 python cart/bg_cart.py comp "${C}/bg" --pano "${C}/bg/pano/${PANO}_16k_tm.jpg" \
  --yaw "${YAW}" --fg "${C}/${KEY}/export/fg" --out "${OUT}" --workers 10 > "${C}/logs/bg_f120.log" 2>&1 \
  && touch "${C}/done/bg_f120" && echo "$(ts) bg done" || { echo "$(ts) bg failed (log ${C}/logs/bg_f120.log)"; exit 1; }
