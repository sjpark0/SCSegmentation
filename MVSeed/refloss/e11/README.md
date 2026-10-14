# S1-E11 design analyses (2026-10-15, read-only on the adopted stage-1 output `geoset_pred_ft_rep_e8`)

Scripts and their text summaries from the design workflow (docs/stage1-E11-prereg.md section 1 quotes them).
Paths inside the scripts point at the repository; the big JSON outputs were not kept.

| script | where | what |
|---|---|---|
| `overlap_oracle.py` → `overlap_oracle.txt` | host (`stage1/.venv`) | overlaps between object masks per scored view; GT-owner oracle (+0.00542 [+0.00346, +0.00747]); GT-free mask-only rules (larger / smaller / bbox centre / centroid / erase) |
| `extent_vis.py`, `extent_vis_report.py` → `extent_vis_report.txt` | scsam3 (GPU, 316 s) | missed GT pixels of the 22 under-cover failing pairs split by reference visibility (91 % invisible); neighbour-seed coverage of the missed region (19 %) |
| `host_anatomy.py`, `aggregate.py` → `aggregate.txt`; `gpu_timing.py` | host / scsam3 | reference-visibility replica, gate rows, round-2 overlaps and exclusivity variants, neighbour choice; round-1 vs round-2 decode time |
| `coupling.py`, `coupling2.py`, `coupling3.py`, `make_table.py` → `coupling*.txt` | host | stage-2 coupling of the seeds; failing cost 1.353 by group / scene / frame 0 vs frames 1-20; per-pair decode-only vs new-information verdicts |
