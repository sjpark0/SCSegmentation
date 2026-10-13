# Reference-camera loss diagnosis scripts (docs/stage2-D-refloss.md)

Descriptive, no registered rule.  Run from a scratch work directory (outputs land in the cwd; the
large JSONs, ~40 MB, are not committed).  Container scripts take `<method> <out.json>`.

| script | where | output | what |
|---|---|---|---|
| `refmap.py` | host | `refmap.json` | ref loss per method, per object (committed) |
| `ceil.py` | host | stdout | subtract the 288-grid ceiling (`MVSeed/runs/grid_ceiling.json`) |
| `kinds.py` | host | stdout | frame-level loss by failure type |
| `anatomy2.py` | scsam3 | `anat2_H.json` | FN/FP, boundary-band oracles (2/4/8 px, bp/2, bp), dilate/erode, copy-GT0 |
| `guided.py` | scsam3 | `gf_H.json` | guided-filter refinement probe |
| `snapfeat2.py` | scsam3 | `snap2_H.json` | GT-free snap features (IoU to frame-0 output, image change) + snapped J&F |
| `sim2.py` | host | stdout | snap-rule grid, LOSO estimate (reads `anat2_H.json`, `snap2_H.json`) |
| `nonref.py` | host | stdout | non-reference loss: E8 headline vs GT-seed headline (E7), F0/N0, seed bins, per-frame curve (docs/stage2-D-nonref.md) |

`refmap.py`/`ceil.py` read `refmap.json` from the cwd; `sim2.py` expects the two container outputs there.

The static/moving split of docs/stage2-D-nonref.md §3 reads `anat2_H.json` (non-reference rows, `Jc`/`Fc` = copy-GT0).
