# S1-E14 design (2026-10-16) — good-seed boundary error, DEV-A only

docs/stage1-E14-prereg.md summarises these. Every number comes from DEV-A (non-keyframe frames ref+4..+20 of each
scene, the 2 non-reference annotated cameras, 3 300 pairs; docs/stage1-dev-protocol.md). The scored keyframe (TEST) and
DEV-B were never decoded or scored. Large outputs (the round-2 logit dump, per-pair candidate rows, gradient dumps,
DEV decodes of the E9/E12 checkpoints) stay git-ignored under stage1/geomem/runs/_e14_*; the scripts refer to those paths.

| folder | what |
|---|---|
| `boundary-anatomy/` | where the good-seed error sits: ±2 px band oracle 0.9767 (63 % of the 0.0634 gap), FN 72 %, the 288-grid round trip 24.5 %, unclaimed misses 51 % (`report_anat.txt`, `report_part.txt`) |
| `logitdump/` | the round-2 low-res logit dump of `geoset_rep_x` on DEV-A (code + verification: re-thresholding reproduces the adopted PNGs with 0 px difference on 3 952 object-views, on GPU and on the host) |
| `threshold-upsample/` | the threshold shift (pos = L > −τ under the adopted exclusivity) and the logit upsamplers; `PREREG.md` (sha256-stamped before scoring), `summary.json`, `report.txt`, `cand/cmp_*.json`, `cand/bil2_tau0.5.json` (the per-pair rows the Z-emul check compares against), `decomp.json` (owner of the added pixels) |
| `multimask/` | multimask token selection / averaging (I3 +0.0012, I4 +0.0008; per-object token oracle) |
| `zoom-xcl/` | E10 zoom followed by E11 exclusivity, and the 2×2 factorial (`compare_arms.json`) |
| `train-band/` | E9 §5(b) band-BCE balance estimate from the E9 D1/D2 and E12 C0′ checkpoints decoded on DEV-A (no training): leak audit, training-log flow identity, gradient balance, dose estimate |
| `synth/` | the synthesis checks: τ power on 5 pseudo-keyframes and frame halves (`tau_power.json`), good-seed size bins, shift mechanism invariants (`shift_invariants.json`) |
| `design_workflow_result.json` | the whole workflow result (analyst maps, measurements, verdicts, synthesis) |
