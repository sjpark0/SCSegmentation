# MVSeg mask-output inventory

Root: `/home/sjpark/Documents/SCSegmentation/Data/MVSeg/` — 24 dataset directories (+ `__pycache__`), 35 GB, **282,187 PNG files** total (`find . -name '*.py' | wc -l` → `find . -name "*.png" | wc -l` = 282187).

## 0. On-disk layout convention (verified, not assumed)

Two distinct nestings coexist.

**A. Ground truth — `<Dataset>/Mask/<cam>/<frame:06d>.png`**
- 8-bit **grayscale, single channel**, pixel value = **object id**; 0 = background/unlabelled. Verified by decoding: `AlexaMeadeExhibit/Mask/camera_0001/000000.png` → 2560×1920, colortype 0, bitdepth 8, 33 unique values `[1..33]`; `Barn/Mask/v0/000000.png` → 30 unique values `[1..30]`.
- Frame filenames are **zero-padded to 6 digits** and start at the dataset's `start_frame`, *not* always 0: `Painter` = `000040..000060`, `MATF` = `000030..000050`, `PoznanStreet` = `000090..000110`, `Frog` = `000180..000200`; the other 11 = `000000..000020`. All 15 main datasets have exactly 21 frames × 3 cameras.
- Sibling `Mask/objects_labels.json` maps `"<id>" -> {label, type}`.

**B. Every result directory — `<Dataset>/<Method>/<cam>/<frame>/<objid>.png`**
- 8-bit grayscale **per-object binary**, values `{0, 255}` only. Verified: `AlexaMeadeExhibit/SegMask/camera_0001/0/1.png` → 2560×1920, 2 unique values `[0,255]`; same for `SegMaskSam3MVOpt`, `ConMask`, `Blocks1/SegMask_SA3D`.
- Frame directory names are the **un-padded integer** matching the GT stem (`Painter/SegMaskSam3OneStage/v6/` = `40..60`, GT `000040.png..000060.png`). Object files are `<id>.png`, un-padded.
- This is exactly what `Data/MVSeg/eval_jf.py:4-8` documents and `eval_jf.py:143-161` reads (`os.path.join(ds_dir, method, cam, pf, f"{obj}.png")`, thresholded `pm > 127`).
- Writer side confirms it: `SCSam2/demo/MaskConvertMVSeg.py:32-36` and `SCSam2/demo/sam2_MVSeg_recheck.py:73,80`.

**Camera directory names are dataset-specific**, driven by `prefix`/`prefix1` in `SCSam2/demo/MVSeg.json`: `camera_%04d` (AlexaMeade*, Dog, Welder), `v%d` (Barn, Breakfast, Carpark, Fencing, Frog, Painter, PoznanStreet), `cam%d`/`cam%02d` (Blocks, CoffeeMartini, FlameSteak), `S1_CAM_%d` (MATF).

**`ConMask` is not a method** — it is the GT re-encoded into layout B, written by `SCSam2/demo/MaskConvertMVSeg.py:26-36` (`mask = (img == i+1) * 255`, looping `i` to `np.max(img)` **per frame**, which is why its counts are ≤ the full object grid).

## 1. Dataset directories

| Group | Directories | GT layout |
|---|---|---|
| Main MVSeg (15) | AlexaMeadeExhibit, AlexaMeadeFacePaint, Barn, Blocks, Breakfast, Carpark, CoffeeMartini, Dog, Fencing, FlameSteak, Frog, MATF, Painter, PoznanStreet, Welder | layout A, 3 cams × 21 frames |
| COLMAP/SA3D variants (9) | Blocks1, Blocks1_SA3D, Blocks1_SA3D_YOLO, Carpark1, Carpark1_SA3D, Carpark1_SA3D_YOLO, Fencing1, Fencing1_SA3D, Fencing1_SA3D_YOLO | **different**: `Mask/` holds flat PNGs `<frame>_p<n>.png` (binary `{0,255}`, e.g. `Blocks1/Mask/00_p1.png` 1920×1080), **no camera subdirectories** |

Consequence: the 9 variant datasets are invisible to `eval_jf.py` — `eval_jf.py:217-218` lists `Mask/` subdirectories to get cameras and finds none, so zero jobs are generated for them.

## 2. (a) Dataset × method coverage matrix — main 15 datasets

Cell = PNG count. `exp` = `max_object_id × 3 cams × 21 frames` (the full grid). `—` = directory absent. All result dirs hold **3 camera subdirs** except where noted.

| Dataset | labels / max id | exp | SegMask | SegMask1 | SegMaskNew | SegMaskNew1 | SegMaskNew2 | SegMaskNew3 | Sam3OneStage | Sam3OneStageNew | Sam3OneStage_recheck | Sam3MVOpt | ConMask |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AlexaMeadeExhibit | 33 / 33 | 2079 | 2079 | 2079 | 2079 | 2079 | 2079 | 2079 | 1942 | **—** | 1942 | 1911 | 2079 |
| AlexaMeadeFacePaint | 14 / 14 | 882 | 756 | 882 | **1260 (5 cams)** | 882 | 882 | 882 | 630 | 630 | 630 | 630 | 840 |
| Barn | 30 / 30 | 1890 | 1890 | 1890 | 1890 | 1890 | 1890 | 1890 | 1890 | 1890 | 1890 | 1890 | 1890 |
| Blocks | 22 / 22 | 1386 | 1323 | 1386 | 1323 | 1386 | 1386 | 1386 | 882 | 882 | 882 | 882 | 1239 |
| Breakfast | 30 / 30 | 1890 | 1890 | 1890 | 1890 | 1890 | 1890 | 1890 | 1438 | 1441 | 1438 | 1441 | 1890 |
| Carpark | 22 / 22 | 1386 | 1386 | 1386 | 1386 | 1386 | 1386 | 1386 | 1386 | 1386 | 1386 | 1386 | 1386 |
| CoffeeMartini | 65 / 66 | 4158 | 4158 | 4158 | 4158 | 4158 | 4158 | 4158 | 2772 | **—** | 2772 | 2755 | 3570 |
| Dog | 9 / 9 | 567 | 441 | 567 | 441 | 567 | 567 | 567 | 483 | 483 | 483 | 483 | 525 |
| Fencing | 8 / 8 | 504 | 504 | 504 | 504 | 504 | 504 | 504 | 504 | 504 | 504 | 504 | 504 |
| FlameSteak | 68 / 68 | 4284 | 3843 | 4284 | 3843 | 4284 | 4284 | 4284 | 2457 | **—** | 2457 | 2457 | 3759 |
| Frog | 5 / 5 | 315 | 315 | 315 | 315 | 315 | 315 | 315 | 315 | 315 | 315 | 315 | 315 |
| MATF | 44 / 44 | 2772 | **1848 (cam S1_CAM_10 empty)** | 2772 | 2772 | 2772 | 2772 | 2772 | 1323 | 1323 | 1323 | 1323 | 2352 |
| Painter | 27 / 28 | 1764 | 1701 | 1764 | 1701 | 1764 | 1764 | 1764 | 1505 | 1491 | 1491 | 1491 | 1722 |
| PoznanStreet | 24 / 24 | 1512 | 1512 | 1512 | 1512 | 1512 | 1512 | 1512 | 1390 | 1406 | 1390 | 1406 | 1512 |
| Welder | 16 / 16 | 1008 | 1008 | 1008 | 1008 | 1008 | 1008 | 1008 | 945 | 941 | 945 | 941 | 1008 |

Single-dataset methods (CoffeeMartini only): `SegMaskSam2Recheck` = 4158 PNG (complete grid, 3 cams × 21 frames); `SegMaskSam3ForSam2New` = 198 PNG (3 cams × **1 frame** × 66 objects — only frame dir `0` exists).

**Missing combinations:** `SegMaskSam3OneStageNew` absent for **AlexaMeadeExhibit, CoffeeMartini, FlameSteak** (3/15). These are precisely the three datasets `SCSam3/runMVOptThree.sh:2` names as "the three datasets no SAM 3 variant could finish". Every other main dataset has all 10 shared method dirs.

**Coverage matrix — 9 variant datasets** (all 4 methods present in all 9, 3 cams × 21 frames each):

| Dataset | Mask (GT, flat) | ConMask | SegMask_SA3D | SegMask_SAM2 | SegMaskNew_SA3D | SegMaskNew_SAM2 |
|---|---|---|---|---|---|---|
| Blocks1 / Blocks1_SA3D / Blocks1_SA3D_YOLO | 20 each | 1239 each | 126 | 126 | 126 | 126 |
| Carpark1 / Carpark1_SA3D / Carpark1_SA3D_YOLO | 45 each | 1386 each | 315 | 315 | 315 | 315 |
| Fencing1 / Fencing1_SA3D / Fencing1_SA3D_YOLO | 20 each | 504 each | 126 | 126 | 126 | 126 |

## 3. (b) Total PNG counts per method directory name

Across **all 24** datasets (`find <dir> -name '*.png' | wc -l`, summed):

| Method dir | # datasets | Total PNG | On-disk (`du -ch`) |
|---|---|---|---|
| ConMask | 24 | 33,978 | 339 M |
| SegMask1 | 15 | 26,397 | 266 M |
| SegMaskNew1 | 15 | 26,397 | 266 M |
| SegMaskNew2 | 15 | 26,397 | 266 M |
| SegMaskNew3 | 15 | 26,397 | 266 M |
| SegMaskNew | 15 | 26,082 | 265 M |
| SegMask | 15 | 24,654 | 252 M |
| SegMaskSam3OneStage | 15 | 19,862 | 214 M |
| SegMaskSam3OneStage_recheck | 15 | 19,848 | 214 M |
| SegMaskSam3MVOpt | 15 | 19,815 | 213 M |
| SegMaskSam3OneStageNew | 12 | 12,692 | 122 M |
| SegMaskSam2Recheck | 1 | 4,158 | 52 M |
| SegMask_SA3D | 9 | 1,701 | — |
| SegMask_SAM2 | 9 | 1,701 | — |
| SegMaskNew_SA3D | 9 | 1,701 | — |
| SegMaskNew_SAM2 | 9 | 1,701 | — |
| Mask (GT) | 24 | 1,200 | 26 M |
| SegMaskSam3ForSam2New | 1 | 198 | 2.5 M |
| **`(Copy)` duplicates (6 dirs)** | 6 | **7,308** | **78 M** |

## 4. (c) Leftover / temporary artifacts

| Path | Size | mtime | Assessment |
|---|---|---|---|
| `AlexaMeadeExhibit/SegMask (Copy)` | 28 M | 2025-09-30 | 2079 PNG, identical count to `SegMask`. GUI copy-paste leftover — delete. |
| `AlexaMeadeExhibit/SegMaskNew (Copy)` | 28 M | 2025-10-14 | 2079 PNG. Same. |
| `Breakfast/SegMaskNew (Copy)` | 13 M | 2025-09-23 | 1890 PNG. Same. |
| `Carpark1_SA3D_YOLO/SegMask_SA3D (Copy)` | 2.8 M | 2025-10-20 | 315 PNG. Same. |
| `Carpark1_SA3D_YOLO/SegMask_SAM2 (Copy)` | 2.8 M | 2025-10-20 | 315 PNG. Same. |
| `Carpark1_SA3D_YOLO/SegMaskNew_SA3D (Copy)` | 2.8 M | 2025-10-20 | 315 PNG. Same. |
| `Carpark1_SA3D_YOLO/SegMaskNew_SAM2 (Copy)` | 2.8 M | 2025-10-20 | 315 PNG. Same. |
| `*/SegMaskSam3OneStage_recheck` (15 dirs) | **214 M total** | 2026-09-04 08:31–08:50 | **Full byte copies, not hardlinks** (`stat -c %i` gives different inodes for `AlexaMeadeExhibit/.../camera_0001/0/1.png`). `diff -rq` vs `SegMaskSam3OneStage` is **empty for 14/15 datasets**; only `Painter` differs, and only by 14 files present in `SegMaskSam3OneStage` and absent in `_recheck` (`v6/47/17.png` … `v6/60/17.png`). No script in the repo writes this name (`grep -rn recheck --include=*.py --include=*.sh` only hits `SCSam2/demo/sam2_MVSeg_recheck.py`, which writes `SegMaskSam2Recheck`). Manual re-run/copy — document or delete. |
| `MATF/SegMask/S1_CAM_10/30/` | 8.0 K | 2025-10-14 | **Empty scaffold**: the camera dir contains one frame dir `30` holding **zero** PNGs. This is why `MATF/SegMask` = 1848 instead of 2772 — one full camera is missing. Broken output, not just a leftover. |
| `CoffeeMartini/SegMaskSam3ForSam2New` | 2.5 M | 2026-09-04 09:14 | Only **1 of 21 frames** written (frame dir `0` per camera). Aborted/in-progress run of `SCSam3/runMVSegForSam2.py` (its `DEFAULT_OUT` at line 27). |
| `Data/MVSeg/jf_raw_sam3_onestage.STALE-prefix-bug.json` | 75 KB | 2026-09-03 | Self-labelled stale result file (not a mask dir, but sits in the same tree). |

No directories matching `base`, `patched`, `trim`, `tmp`, or `bak` exist under `Data/MVSeg/` (`find . -maxdepth 2 -type d -iname ...` returned none). Note `SCSAM3_TRIM_CACHED_OUTPUTS=1` in `SCSam3/runMVOptThree.sh:14` is an env var, not a directory.

`Data/` is gitignored (`.gitignore:181`), so none of this is version-controlled.

## 5. Ambiguities / contradictions found — stated, not resolved

1. **`SegMaskSam3MVOpt` vs `SegMaskSam3OneStageNew` are byte-identical.** `diff -rq` returns 0 differences for **all 12** datasets where both exist. MVOpt additionally covers the 3 datasets OneStageNew lacks. mtimes show MVOpt was written later and per-dataset (Barn: OneStageNew 2026-09-03 09:34, MVOpt 2026-09-04 06:12), consistent with a genuine re-run of a memory-optimized implementation that produces the same masks (`SCSam3/runMVSeg.py:41-43` defines both as separate `--algo` values). I cannot tell from the filesystem alone whether these are two algorithms that coincide or one algorithm run twice — the totals in §3 double-count them either way.
2. **`objects_labels.json` count ≠ max object id** for two datasets: `CoffeeMartini` has 65 entries with ids `1..66` (**id 55 absent**), `Painter` has 27 entries with ids `1..28` (**id 4 absent**). Result dirs disagree about which convention to follow: `CoffeeMartini/SegMask/cam02/0/` writes all of `1..66` (including unlabelled 55); `Painter/SegMask/v0/40/` writes `1..27` (omits 28) while `Painter/SegMask1/v0/40/` writes `1..28`. GT pixels confirm the gaps (`Painter/Mask/v0/000040.png` unique = `[1,2,3,5,...,27]`, no 4). The "exp" column in §2 uses max id; using label count instead lowers it by 63 for both.
3. **`AlexaMeadeFacePaint/SegMaskNew` has 5 camera dirs** (`camera_0005..camera_0009`, 252 PNG each) while the GT and every other method for that dataset have only 3 (`camera_0007..0009`). The two extra cameras have no ground truth and are silently ignored by `eval_jf.py` (which iterates GT cameras), but they inflate the raw count to 1260.
4. **`SegMaskSam3OneStage_recheck` provenance is unrecorded** — see the table above. Whether it or `SegMaskSam3OneStage` is the intended input to `jf_raw_sam3_onestage*.json` cannot be determined from the tree.

Key files: `/home/sjpark/Documents/SCSegmentation/Data/MVSeg/eval_jf.py`, `/home/sjpark/Documents/SCSegmentation/SCSam2/demo/MaskConvertMVSeg.py`, `/home/sjpark/Documents/SCSegmentation/SCSam2/demo/MVSeg.json`, `/home/sjpark/Documents/SCSegmentation/SCSam3/runMVSeg.py`, `/home/sjpark/Documents/SCSegmentation/SCSam3/runMVSegAll.sh`, `/home/sjpark/Documents/SCSegmentation/SCSam3/runMVOptThree.sh`, `/home/sjpark/Documents/SCSegmentation/SCSam3/runMVSegForSam2.py`, `/home/sjpark/Documents/SCSegmentation/SCSam2/demo/sam2_MVSeg_recheck.py`.