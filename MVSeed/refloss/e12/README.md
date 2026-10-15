# S1-E12 design analyses (2026-10-15, read-only on the adopted stage-1 output `geoset_pred_ft_rep_x_e11` and the training data)

Scripts and their text outputs from the design workflows (docs/stage1-E12-prereg.md section 1 quotes them).
Paths inside the scripts point at the repository or at the session scratchpad they were written in; the large JSON outputs were not kept.

| file | where | what |
|---|---|---|
| `design_findings.md`, `design_proposals.md`, `design_synthesis.json` | — | the analyst reports (training-pipeline code map, visibility signal, mechanism prior), the three design proposals, the three judge verdicts and the synthesis the registration is built from |
| `train_vis.py`, `vis_report.py` → `vis_report.txt` | scsam3 (GPU, 213 s) / host | reference-invisible share of the supervised target GT over the trainer's 1 122 samples (47.9 %); keyframe recall of invisible GT by class for the adopted output (0.913; 22 under-cover pairs 0.611) |
| `vis_cost.py` → `vis_cost.out` | scsam3 | cost of computing visibility inside the training loop; projected reference GT covers 47.4 % of target GT |
| `lossmass_train.py` → `lossmass_train_report.txt` | scsam3 | focal / dice loss mass already on reference-invisible pixels (0.726 / region 0.761) — why the visibility-weighted loss arm was dropped |
| `oracle.py` | host | J oracle of adding all missed reference-invisible GT (+0.0410 over 664; outside +0.0094, occluded +0.0283, nodepth +0.0034) and the over-extension risk grid |
| `cut_stats.py` → `cut_stats.out` | host | the registered half-plane cut rule on the training reference label maps: eligible rows 0.613 (98.3 % of GT px), realised dropped object-cell fraction 0.442 |
| `kf_elig.py` → `kf_elig.out` | host | cut-eligible keyframe pairs (object ≥ 20 reference cells): 437/664, good 407/605, under-cover 15/22 |
| `gpu_probe.py` → `gpu_probe.out` | scsam3 | per-object per-cell `-inf` gate: other objects bitwise unchanged, pointer-only rows finite, zero gate == None path; mask cost +0.075 s per step; backward nondeterminism (107/108 gradient tensors differ) |
| `retrain40.out`, `retrain40b.out` | scsam3 | 40-step retrains of the adopted recipe: step-1 loss identical (0.13690), step 2 diverges — why the zero-dose control C0' is trained |
| `null_compose.py`, `null_report.py` → `null_report.txt`; `host_compose_check.py` → `host_compose_check.out`; `shell_reach.py` | scsam3 / host | the add-only "composition shell" measured on the keyframe (D1 +0.0011) — left OUT of E12 because it was designed on the test pairs |
| `keyframe_vis.py` | scsam3 (GPU, ~30 s per 6 folders) | **chain stage `vis`**: per-pair GT / hit / FP pixels of each result folder by reference visibility class, and the pixels one folder adds / removes relative to another with their GT overlap → `MVSeed/runs/e12/vis.json` (judge: recall per class, V1-V4) |
| `notcut_share.py` → `MVSeed/runs/e12/notcut_share.json` | host | post-hoc Zt "not cut" share of M+ by replaying each fold's sample order against the reference label maps (0.48 %; replay mismatches 0) |
