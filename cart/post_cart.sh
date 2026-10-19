#!/usr/bin/env bash
# After the cart run (docs/cart.md): label / foreground PNGs, the native-resolution label volume, overlay videos.
#   waits for Data/Cart/done/run_f120; writes Data/Cart/done/{export,render}_f120.  Host python (stage1/.venv).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
C="${ROOT}/Data/Cart"; KEY=cart_f120; M=SegMaskSam3XW1CGPS4MFsCs10kG2; PY="${ROOT}/stage1/.venv/bin/python"
ts() { date '+%F %T'; }
until [[ -f "${C}/done/run_f120" ]]; do sleep 60; done
echo "$(ts) export: labels / fg PNGs + labels.npy"
if [[ ! -f "${C}/done/export_f120" ]]; then
  PYTHONDONTWRITEBYTECODE=1 "${PY}" "${ROOT}/cart/export_cart.py" "${C}/${KEY}" "${M}" "${C}/${KEY}/export" --start 120 \
    --npy "${C}/labels/seg_labels_cart_f120_2699" --workers 16 > "${C}/logs/export_f120.log" 2>&1 \
    && touch "${C}/done/export_f120" || { echo "$(ts) export failed (log ${C}/logs/export_f120.log)"; exit 1; }
fi
echo "$(ts) export done: $(tail -n 1 "${C}/logs/export_f120.log" | cut -c1-200)"
echo "$(ts) render: per-camera and 24-camera overlay videos"
if [[ ! -f "${C}/done/render_f120" ]]; then
  PYTHONDONTWRITEBYTECODE=1 nice -n 10 "${PY}" "${ROOT}/cart/render_cart.py" "${C}/${KEY}" "${C}/${KEY}/export" \
    "${C}/${KEY}/viz" --start 120 --workers 12 > "${C}/logs/render_f120.log" 2>&1 \
    && touch "${C}/done/render_f120" || { echo "$(ts) render failed (log ${C}/logs/render_f120.log)"; exit 1; }
fi
echo "$(ts) render done: ${C}/${KEY}/viz"
