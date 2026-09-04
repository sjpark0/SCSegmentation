## SCSam3 run-log inventory — `/home/sjpark/Documents/SCSegmentation/SCSam3/logs/`

113 files: 14 run directories (102 per-dataset logs) + 11 top-level driver logs. Facts below read from those files and from the runner scripts.

### Runners and what they write

| runner | log dir prefix | python entry | algorithm(s) |
|---|---|---|---|
| `SCSam3/runMVSegAll.sh` | `mvseg-<stamp>/<algo>-<ds>.log` | `runMVSeg.py` | `OneStage`, `OneStageNew` |
| `SCSam3/runMVOptThree.sh` | `mvopt-<stamp>/<ds>.log` | `runMVSeg.py --algo ${ALGO:-MVOpt}` | `MVOpt` (or `ALGO=` override) |
| `SCSam3/runForSam2Three.sh` | `forsam2-<stamp>/<ds>.log` | `runMVSegForSam2.py` | `ForSam2New` |
| `SCSam3/waitAndRunMVSeg.sh` | `auto-<stamp>.log` (calls `runMVSegAll.sh`) | — | — |

Stage markers come from the prints in `SCSam3/runMVSeg.py:310,319,322,338,356,358` and `SCSam3/runMVSegForSam2.py:177,187,193,209,211`:
`sessions ready` / `states ready` → `prompted` → `cross-view     done` → (`spatial model  retired`, MVOpt only) → `  frame N written` ×21 → `done -> <out>`.

### Directory ↔ driver log

| log dir | started | driver log | algorithm (from log `algorithm` line) | output folder | datasets | ok / fail |
|---|---|---|---|---|---|---|
| `mvseg-20260903-030244` | 2026-09-03 03:02:44 | `auto-20260902-235743.log` (smoke test) | OneStageNew | — | 1 | 0/1 |
| `mvseg-20260903-030512` | 03-03 03:05:12 | none | OneStageNew | SegMaskSam3OneStageNew | 1 | 1/0 |
| `mvseg-20260903-030756` | 03-03 03:07:56 | none | OneStage | SegMaskSam3OneStage | 7 | 5/2 |
| `mvseg-20260903-085425` | 03-03 08:54:25 | none | OneStage | SegMaskSam3OneStage | 1 | 1/0 |
| `mvseg-20260903-085851` | 03-03 08:58:51 | `sweep-20260903-085851.log` | OneStage ×15, OneStageNew ×7 | both | 22 | 20/2 |
| `mvseg-20260903-104328` | 03-03 10:43:28 | none | OneStage | SegMaskSam3OneStage | 1 | 1/0 |
| `mvseg-20260903-104557` | 03-03 10:45:57 | `sweep-full-104557.log` | OneStage ×15, OneStageNew ×7 | both | 22 | 20/2 |
| `mvseg-20260903-114432` | 03-03 11:44:32 | `sweepN-114432.log` | OneStageNew | SegMaskSam3OneStageNew | 8 | 7/1 |
| `forsam2-20260903-125826` | 03-03 12:58:26 | `forsam2-run-125826.log` | ForSam2New | SegMaskSam3ForSam2New | 3 | 0/3 |
| `forsam2-20260903-130304` | 03-03 13:03:04 | `forsam2-run-130304.log` | ForSam2New | SegMaskSam3ForSam2New | 3 | 0/3 |
| `mvopt-20260904-010657` | 03-04 01:06:57 | `mvopt-run-010657.log` | MVOpt | SegMaskSam3MVOpt | 3 | 3/0 |
| `mvopt-20260904-060418` | 03-04 06:04:18 | `mvopt-rest-060418.log` | MVOpt | SegMaskSam3MVOpt | 12 | 12/0 |
| `mvopt-20260904-083026` | 03-04 08:30:26 | `onestage-recheck-083026.log` | **OneStage** (not MVOpt) | SegMaskSam3OneStage_recheck | 15 | 15/0 |
| `forsam2-20260904-090459` | 03-04 09:04:59 | `f2n-retry-090459.log` | ForSam2New | SegMaskSam3ForSam2New | 3 | 0/3 |

Naming caveat: `mvopt-20260904-083026` is produced by `runMVOptThree.sh` (hence the `mvopt-` prefix) but was invoked with `ALGO=OneStage`; every log inside says `algorithm      OneStage  (demoSCSam3OneStage)`.

Also present, not part of any run dir: `logs/sam2-recheck-CoffeeMartini.log` — a SAM 2 (`/opt/sam2`) baseline recheck of CoffeeMartini, truncated at `propagate in video: 7% | 20/300`, with no `done`, no `Traceback`, no error text.

### Full run table

`result` = whether the log contains `done -> `. Peak column reports the two GiB figures the CUDA OOM message actually carries — *`Process N has X GiB memory in use` / `Of the allocated memory Y GiB is allocated by PyTorch`* — against a total capacity of 47.36 GiB in every case. **Caveat: no log reports a true peak; these are the instantaneous figures at the moment of the failed allocation.**

| log dir | date | algorithm | dataset | result | failure stage | peak GiB (in-use / PyTorch-alloc) |
|---|---|---|---|---|---|---|
| mvseg-20260903-030244 | 09-03 03:02 | OneStageNew | Frog | FAIL | **model build**, before any stage marker — `build_scsam3.py:735 → pkg_resources.resource_filename` → `TypeError: expected str, bytes or os.PathLike object, not NoneType` | n/a |
| mvseg-20260903-030512 | 09-03 03:05 | OneStageNew | Frog | ok | — | — |
| mvseg-20260903-030756 | 09-03 03:07 | OneStage | AlexaMeadeExhibit | FAIL | **temporal tracking** (`cross-view done` present, 1 frame written, dies at frame 1) — `SCSam3TrackerPredictor.py:1393 _apply_object_wise_non_overlapping_constraints`, tried to allocate 1.21 GiB | 46.07 / 44.65 |
| mvseg-20260903-030756 | 09-03 03:07 | OneStage | AlexaMeadeFacePaint | ok | — | — |
| mvseg-20260903-030756 | 09-03 03:07 | OneStage | Barn | ok | — | — |
| mvseg-20260903-030756 | 09-03 03:07 | OneStage | Blocks | ok | — | — |
| mvseg-20260903-030756 | 09-03 03:07 | OneStage | Breakfast | ok | — | — |
| mvseg-20260903-030756 | 09-03 03:07 | OneStage | Carpark | ok | — | — |
| mvseg-20260903-030756 | 09-03 03:07 | OneStage | CoffeeMartini | FAIL | **temporal tracking**, log truncated at 12/22 frames written (last line `55%|12/22 [32:04<1:21:41, 490.18s/it]`); **no Traceback, no OOM text, no `Killed`** | none reported |
| mvseg-20260903-085425 | 09-03 08:54 | OneStage | AlexaMeadeExhibit | ok | — | — |
| mvseg-20260903-085851 | 09-03 08:58 | OneStage | AlexaMeadeExhibit … Welder (all 15) | ok (15/15) | — | — |
| mvseg-20260903-085851 | 09-03 08:58 | OneStageNew | AlexaMeadeExhibit | FAIL | **temporal tracking** (`cross-view done, 45 views` present, 0 frames written) — `SCSam3TrackerPredictorNewMem.py:1926 _apply_object_wise_non_overlapping_constraints`, tried to allocate 976.00 MiB | 46.11 / 44.87 |
| mvseg-20260903-085851 | 09-03 08:58 | OneStageNew | AlexaMeadeFacePaint | ok | — | — |
| mvseg-20260903-085851 | 09-03 08:58 | OneStageNew | Barn | ok | — | — |
| mvseg-20260903-085851 | 09-03 08:58 | OneStageNew | Blocks | ok | — | — |
| mvseg-20260903-085851 | 09-03 08:58 | OneStageNew | Breakfast | ok | — | — |
| mvseg-20260903-085851 | 09-03 08:58 | OneStageNew | Carpark | ok | — | — |
| mvseg-20260903-085851 | 09-03 08:58 | OneStageNew | CoffeeMartini | FAIL | **temporal tracking**, truncated at 12/22 frames (`55%|12/22 [16:49<39:50]`); **no Traceback / OOM / `Killed`**; driver `sweep-20260903-085851.log` also ends mid-line on this row | none reported |
| mvseg-20260903-104328 | 09-03 10:43 | OneStage | Blocks | ok | — | — |
| mvseg-20260903-104557 | 09-03 10:45 | OneStage | AlexaMeadeExhibit … Welder (all 15) | ok (15/15) | — | — |
| mvseg-20260903-104557 | 09-03 10:45 | OneStageNew | AlexaMeadeExhibit | FAIL | **temporal tracking** (cross-view done, 0 frames) — `SCSam3TrackerPredictorNewMem.py:1926`, tried to allocate 976.00 MiB | 46.11 / 44.87 |
| mvseg-20260903-104557 | 09-03 10:45 | OneStageNew | AlexaMeadeFacePaint | ok | — | — |
| mvseg-20260903-104557 | 09-03 10:45 | OneStageNew | Barn | ok | — | — |
| mvseg-20260903-104557 | 09-03 10:45 | OneStageNew | Blocks | ok | — | — |
| mvseg-20260903-104557 | 09-03 10:45 | OneStageNew | Breakfast | ok | — | — |
| mvseg-20260903-104557 | 09-03 10:45 | OneStageNew | Carpark | ok | — | — |
| mvseg-20260903-104557 | 09-03 10:45 | OneStageNew | CoffeeMartini | FAIL | **temporal tracking**, truncated at 13/22 frames (`59%|13/22 [05:53<06:40]`); **no Traceback / OOM / `Killed`**; driver `sweep-full-104557.log` also ends mid-line here | none reported |
| mvseg-20260903-114432 | 09-03 11:44 | OneStageNew | Dog, Fencing, Frog, MATF, Painter, PoznanStreet, Welder | ok (7/7) | — | — |
| mvseg-20260903-114432 | 09-03 11:44 | OneStageNew | FlameSteak | FAIL | **temporal tracking**, truncated at 11/22 frames (`50%|11/22 [25:21<1:14:43, 407.58s/it]`); **no Traceback / OOM / `Killed`**; driver `sweepN-114432.log` ends mid-line here | none reported |
| forsam2-20260903-125826 | 09-03 12:58 | ForSam2New | AlexaMeadeExhibit | FAIL | **video/session load**, no stage marker at all — `SCSam3Video.py:51 LoadVideo_Folder_MVSeg` → `TypeError: SCSam3TrackerPredictorNewMem.init_state() got an unexpected keyword argument 'video_path'` | n/a |
| forsam2-20260903-125826 | 09-03 12:58 | ForSam2New | CoffeeMartini | FAIL | same `init_state(video_path=…)` TypeError | n/a |
| forsam2-20260903-125826 | 09-03 12:58 | ForSam2New | FlameSteak | FAIL | same `init_state(video_path=…)` TypeError | n/a |
| forsam2-20260903-130304 | 09-03 13:03 | ForSam2New | AlexaMeadeExhibit | FAIL | **temporal tracking — seeding** (`cross-view done` present; `runMVSegForSam2.py:195 RunNaiveTracking` → `:160 add_new_mask` → `SCSam3TrackerPredictorNewMem.py:489`), tried to allocate 508.00 MiB | 46.31 / 45.45 |
| forsam2-20260903-130304 | 09-03 13:03 | ForSam2New | CoffeeMartini | FAIL | same path, tried to allocate 880.00 MiB | 45.94 / 44.54 |
| forsam2-20260903-130304 | 09-03 13:03 | ForSam2New | FlameSteak | FAIL | same path, tried to allocate 796.00 MiB | 46.00 / 44.69 |
| mvopt-20260904-010657 | 09-04 01:06 | MVOpt | AlexaMeadeExhibit | ok | — | — |
| mvopt-20260904-010657 | 09-04 01:06 | MVOpt | CoffeeMartini | ok | — | — |
| mvopt-20260904-010657 | 09-04 01:06 | MVOpt | FlameSteak | ok | — | — |
| mvopt-20260904-060418 | 09-04 06:04 | MVOpt | AlexaMeadeFacePaint, Barn, Blocks, Breakfast, Carpark, Dog, Fencing, Frog, MATF, Painter, PoznanStreet, Welder | ok (12/12) | — | — |
| mvopt-20260904-083026 | 09-04 08:30 | OneStage | all 15 datasets | ok (15/15) | — | — |
| forsam2-20260904-090459 | 09-04 09:04 | ForSam2New | AlexaMeadeExhibit | FAIL | **temporal tracking — seeding** (`RunNaiveTracking:160 add_new_mask` → `…NewMem.py:500`), 0 frames written, tried to allocate 600.00 MiB | 45.84 / 44.73 |
| forsam2-20260904-090459 | 09-04 09:04 | ForSam2New | CoffeeMartini | FAIL | **temporal tracking — propagation** (1 frame written, `runMVSegForSam2.py:199 next(tracking_result)` → `sam3/sam/rope.py:79 apply_rotary_enc`), tried to allocate 670.00 MiB | 45.81 / 44.48 |
| forsam2-20260904-090459 | 09-04 09:04 | ForSam2New | FlameSteak | FAIL | **temporal tracking — seeding** (`add_new_mask` → `…NewMem.py:500`), tried to allocate 838.00 MiB | 45.61 / 44.83 |

Rows collapsed with "all 15" / explicit dataset lists are all-ok groups; every failing dataset has its own row. Per-log mtimes give finish times; the `date` column is the directory's start stamp.

### Datasets that have ever failed vs always succeeded

| dataset | logs | ok | fail | verdict |
|---|---|---|---|---|
| AlexaMeadeExhibit | 11 | 5 | 6 | **ever failed** (OneStage ×1, OneStageNew ×2, ForSam2New ×3) |
| CoffeeMartini | 10 | 4 | 6 | **ever failed** (OneStage ×1, OneStageNew ×2, ForSam2New ×3) |
| FlameSteak | 8 | 4 | 4 | **ever failed** (OneStageNew ×1, ForSam2New ×3) |
| Frog | 7 | 6 | 1 | **ever failed** — once only, and not a memory failure (the 03:02 model-build TypeError) |
| AlexaMeadeFacePaint | 7 | 7 | 0 | always succeeded |
| Barn | 7 | 7 | 0 | always succeeded |
| Blocks | 8 | 8 | 0 | always succeeded |
| Breakfast | 7 | 7 | 0 | always succeeded |
| Carpark | 7 | 7 | 0 | always succeeded |
| Dog | 5 | 5 | 0 | always succeeded |
| Fencing | 5 | 5 | 0 | always succeeded |
| MATF | 5 | 5 | 0 | always succeeded |
| Painter | 5 | 5 | 0 | always succeeded |
| PoznanStreet | 5 | 5 | 0 | always succeeded |
| Welder | 5 | 5 | 0 | always succeeded |

### Cross-cutting observations

- **No log anywhere under `logs/` contains the string `Killed`** (`grep -rlai killed` returns nothing), so no failure is documented as an OOM-kill by the container/kernel.
- **Cross-view propagation never failed.** Every OOM-or-truncation failure has `cross-view     done, N views have masks` in the log; the only failures before that point are the two non-memory bugs (`mvseg-20260903-030244` model build, `forsam2-20260903-125826` `init_state(video_path=…)`). All 15 memory-related failures are in temporal tracking.
- **Four failures have no recorded reason** (`mvseg-20260903-030756/OneStage-CoffeeMartini`, `mvseg-20260903-085851/OneStageNew-CoffeeMartini`, `mvseg-20260903-104557/OneStageNew-CoffeeMartini`, `mvseg-20260903-114432/OneStageNew-FlameSteak`): the log simply stops mid-progress-bar between 11/22 and 13/22 frames, and in three of the four the *driver* log stops mid-row too. The runner-script comments (`runForSam2Three.sh:3-4`, `runMVOptThree.sh:2-4`) attribute this class of event to host RAM exhaustion rebooting the machine, but **nothing in the logs themselves confirms that** — treat the cause as unrecorded.
- The `--memory=90g` container cap and `SCSAM3_TRIM_CACHED_OUTPUTS=1` appear only in `runForSam2Three.sh` / `runMVOptThree.sh`, not in `runMVSegAll.sh`; all four unexplained truncations are from `runMVSegAll.sh` runs.
- Every OOM message reports the same GPU total capacity, **47.36 GiB**, with 45.50 MiB – 888.12 MiB free at failure.
- `mvopt-20260904-060418` covers the 12 datasets *not* in `mvopt-20260904-010657`; together they are one 15-dataset MVOpt sweep with **zero failures**, and `mvopt-20260904-083026` is a 15-dataset OneStage recheck with **zero failures** — the only two complete clean sweeps in the directory.