# S1-E13 dev-only analyses (2026-10-15) — new information sources for the under-cover block

docs/stage1-E13-dev.md summarises these. Every number here comes from the DEV set (non-keyframe frames ref+4..+20,
2 non-reference annotated cameras, 3 300 pairs); the scored keyframe was never read. Large outputs (dev decode masks,
proposals, detections) stay git-ignored under stage1/geomem/runs/_e13_*; scripts refer to those paths.

| folder | what |
|---|---|
| `protocol-inventory/` | which sources the MUVOD protocol allows (c_ini mask, all rig RGB + Pi3, c_ini video) vs separate-track (label names, a second annotated camera) vs forbidden; their reach into the beyond-c_ini classes |
| `dev-route/`, `dev/` | the DEV set and the scratch decode of the adopted preset geoset_rep_x on non-keyframe frames (`dev_decode.py`, `dev_score.py`, `dev_report.txt`: J&F 0.9078, under-cover population 140 pairs) |
| `sam3modes/` | SAM 3 single-image prompting modes in this repo (point / box / mask / text / same-image exemplar) with a GPU probe on 3 dev frames |
| `sam3-prompt-extend/` | candidate C (protocol-compatible): SAM 3 prompted from the visible reprojection; `REGISTRATION.txt` fixed before the run; −0.0012 all-pair |
| `geometry-grow/` | candidate D (protocol-compatible): multi-view geometric / colour growth; −0.0048 all-pair, every sensitivity CI < 0 |
| `text-concept/` | candidate E1 (protocol change): label-name concept prompts; registered arms null, a post-hoc env-only arm +0.0008 |
| `second-reference/` | candidate E2 (protocol change, diagnostic): a second annotated reference view; +0.0067 all-pair, but closes only ~10 % of the extent deficit |
| `judge-e13/` | the judges' independent re-checks |
| `design_workflow_result.json` | the whole workflow result (analyst maps, measurements, verdicts, synthesis) |
