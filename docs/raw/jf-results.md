## Authoritative J&F results — MVSeg

All facts below were read from files under `/home/sjpark/Documents/SCSegmentation/Data/MVSeg/` on 2026‑09‑04. `Data/` is git-ignored (`/home/sjpark/Documents/SCSegmentation/.gitignore:172`), so **file mtimes are the only provenance available** — there is no git history for any `jf_*.json`.

---

### 1. The two tools

**`Data/MVSeg/eval_jf.py`** (8943 B, mtime 2026‑09‑02 14:59) — produces the raw files.

| Aspect | Fact | Cite |
|---|---|---|
| Expected layout | GT `Mask/<cam>/<frame>.png` (grey, pixel = object id); prediction `<method>/<cam>/<frame>/<id>.png` | `eval_jf.py:6-11` |
| Sequence unit | one `(dataset, camera)` pair = one DAVIS sequence | `eval_jf.py:29-33` |
| J | mask IoU; `union == 0 → 1.0` | `eval_jf.py:36-39` |
| F | DAVIS boundary F, `BOUND_TH = 0.008`, `bound_pix = ceil(0.008 * ‖(h,w)‖)` | `eval_jf.py:33`, `:128-129` |
| Missing prediction file | counted in `missing_files` and **scored as an empty prediction**, not skipped | `eval_jf.py:158-161` |
| Output per camera | `J_all`, `F_all`, `J_inner`, `F_inner` (per-object means), `missing_files` | `eval_jf.py:175-181` |
| `inner` | `slice(1,-1)` — drops first and last frame, as the DAVIS code does | `eval_jf.py:174` |
| CLI | `datasets…`, `--methods` (default the 6 SAM 2 dirs), `--out` (default `jf_raw.json`) | `eval_jf.py:187-195` |
| Dataset auto-selection | only dirs that have `Mask/` **and every** requested method dir | `eval_jf.py:198-201` |
| Requires | `numpy`, `cv2` (run inside the `scsam3` container) | `eval_jf.py:16-19` |

**`Data/MVSeg/report_jf.py`** (6783 B, mtime 2026‑09‑03 13:24) — aggregates. Stdlib only (verified: runs on system `python3` 3.12.3, which has **no** `cv2`).

| Option | Meaning | Cite |
|---|---|---|
| `--raw f1 f2 …` | one or more `eval_jf.py` outputs, **pooled** (default `jf_raw.json`); paths are resolved relative to `report_jf.py`'s own directory | `report_jf.py:96-97`, `:24`, `:118-120` |
| `--methods …` | which result folders to show, **in that order** (default: every method present in the pooled files) | `report_jf.py:98-99`, `:121` |
| `--datasets …` | restrict to these datasets | `report_jf.py:100-101`, `:122-123` |
| `--common` | keep only datasets **every selected method covers with a non-null result**; prints the count and the dropped names | `report_jf.py:102-105`, `:124-134` |
| `--no-missing` | suppress the trailing list of objects with no exported mask | `report_jf.py:106-107`, `:171` |
| Two aggregations | `as-is` (missing exports = empty predictions, the DAVIS protocol) and `exported only` (drop objects with no output file at all) | `report_jf.py:11-17`, `:137-146` |
| How "absent" is decided | **by listing the on-disk directory `<ROOT>/<ds>/<method>/<cam>/`**, not from a bookkeeping file | `report_jf.py:27-49` |

⚠ That last row is load-bearing — see §5.

---

### 2. What actually exists on disk

24 dirs under `Data/MVSeg/` contain a `Mask/`; 15 of them carry the SAM 2/SAM 3 result folders. The other 9 (`Blocks1`, `Blocks1_SA3D`, `Blocks1_SA3D_YOLO`, `Carpark1`, `Carpark1_SA3D`, `Carpark1_SA3D_YOLO`, `Fencing1`, `Fencing1_SA3D`, `Fencing1_SA3D_YOLO`) carry **none** of them.

Result-folder inventory (count = number of the 15 scored datasets that have it):

| Folder | # datasets | Note |
|---|---|---|
| `SegMask`, `SegMask1`, `SegMaskNew`, `SegMaskNew1`, `SegMaskNew2`, `SegMaskNew3` | 15 each | SAM 2 baselines |
| `SegMaskSam3OneStage` | 15 | SAM 3 |
| `SegMaskSam3MVOpt` | 15 | SAM 3 |
| `SegMaskSam3OneStageNew` | **12** | **absent for `AlexaMeadeExhibit`, `CoffeeMartini`, `FlameSteak`** |
| `SegMaskSam3OneStage_recheck` | 15 | no `jf_*.json` exists for it; mtimes 2026‑09‑04 08:31–08:50; byte-identical to `SegMaskSam3OneStage` on `Barn` (md5 of all 1890 PNGs) |
| `SegMaskSam2Recheck` | 1 (`CoffeeMartini`) | scored in `jf_recheck.json` |
| `SegMaskSam3ForSam2New` | 1 (`CoffeeMartini`, 198 PNGs, 2026‑09‑04 09:14) | not scored in any `jf_*.json` |
| `SegMask (Copy)`, `SegMaskNew (Copy)`, `SegMask*_SA3D*`, `SegMask*_SAM2*` | 1–9 | stray/other-method, never referenced by any `jf_*.json` |

---

### 3. Inventory of `jf_*.json`

| File | Bytes | mtime | Method(s) | Datasets | Status |
|---|---|---|---|---|---|
| `jf_raw.json` | 469 546 | 2026‑09‑02 14:29 | the 6 SAM 2 dirs (45 entries each, 270 total) | 15, all non-null | **CURRENT** — the only SAM 2 source |
| `jf_raw_sam3_onestage.STALE-prefix-bug.json` | 75 255 | 2026‑09‑03 10:42 | `SegMaskSam3OneStage` (45) | 15 | **SUPERSEDED** by `jf_sam3_onestage.json` — see §4 |
| `jf_sam3_onestage.json` | 78 822 | 2026‑09‑03 13:17 | `SegMaskSam3OneStage` (45) | 15 | **CURRENT** |
| `jf_sam3_onestagenew.json` | 53 890 | 2026‑09‑03 13:19 | `SegMaskSam3OneStageNew` (36) | **12** (no `AlexaMeadeExhibit`/`CoffeeMartini`/`FlameSteak`) | Superset-superseded by `_full`, but its 36 entries are **bit-identical** to the corresponding entries in `_full`. Still useful: it is the file that makes `--common` yield the 12-dataset subset |
| `jf_sam3_onestagenew_full.json` | 78 636 | 2026‑09‑04 01:47 | `SegMaskSam3OneStageNew` (45) | 15 | **CURRENT** for 15-dataset coverage — but see the caveat in §5 (its 3 extra datasets have no masks on disk any more) |
| `jf_sam3_mvopt.json` | 24 692 | 2026‑09‑04 01:46 | `SegMaskSam3MVOpt` (9) | 3 (`AlexaMeadeExhibit`, `CoffeeMartini`, `FlameSteak`) | **SUPERSEDED** by `_all` — its 9 entries are bit-identical to the same entries in `_all` |
| `jf_sam3_mvopt_all.json` | 78 366 | 2026‑09‑04 06:49 | `SegMaskSam3MVOpt` (45) | 15, all non-null | **CURRENT** |
| `jf_recheck.json` | 9 470 | 2026‑09‑03 14:03 | `SegMaskSam2Recheck` (3) | 1 (`CoffeeMartini`) | Side probe. J/F per camera: cam02 J 0.9262 / F 0.9755, cam10 J 0.7170 / F 0.7670, cam16 J 0.7289 / F 0.8036; `missing_files` 0 everywhere. Not part of the main tables |
| `jf_summary.json` | 17 323 | 2026‑09‑02 14:34 | 6 SAM 2 methods, 15 datasets | — | Derived snapshot of `jf_raw.json` only; pre-dates all SAM 3 work |
| `missing.json` | 23 698 | 2026‑09‑02 14:31 | — | — | Bookkeeping from the 2026‑09‑02 SAM 2 run |
| `jf_report.txt` | 12 874 | 2026‑09‑02 14:33 | 6 SAM 2 methods | 15 | Frozen text report; its SAM 2 numbers match §6 exactly (spot-checked rows `AlexaMeadeExhibit`…`Carpark`) |

---

### 4. The `*.STALE-prefix-bug.json` file

`jf_raw_sam3_onestage.STALE-prefix-bug.json` vs the current `jf_sam3_onestage.json`: same 45 `(dataset, camera)` keys, **41 of 45 entries byte-identical**. Exactly 4 differ:

| dataset/camera | stale J | current J | stale F | current F | stale `missing_files` | current | objects × frames |
|---|---|---|---|---|---|---|---|
| `Blocks/cam0` | 0.0095 | 0.5753 | 0.0095 | 0.6022 | 420 | 147 | 20 × 21 = 420 |
| `Blocks/cam4` | 0.0030 | 0.7716 | 0.0030 | 0.8135 | 336 | 42 | 16 × 21 = 336 |
| `Painter/v0` | 0.0267 | 0.8303 | 0.0267 | 0.8819 | 525 | 57 | 25 × 21 = 525 |
| `Painter/v6` | 0.0311 | 0.8440 | 0.0317 | 0.8906 | 162 | 42 | 26 × 21 = 546 |

What the content says was wrong:

- The failure is confined to **`Blocks` and `Painter`** — and those are **exactly the two of the 15 datasets whose config has `"prefix1": 0`** (no zero-padding of the camera number) in `/home/sjpark/Documents/SCSegmentation/SCSam3/demo/MVSeg.json`. Every other dataset has `prefix1` ≥ 2 (e.g. `FlameSteak`, `CoffeeMartini`: `"prefix": "cam", "prefix1": 2`; `AlexaMeadeExhibit`: `"camera_"`, 4). The camera folder name is built as `f"{prefix}{c:0{prefix1}d}"` — `cam_name()` at `/home/sjpark/Documents/SCSegmentation/SCSam3/runMVSeg.py:55-56`, used for the output path at `runMVSeg.py:349`. That is what "prefix bug" names.
- Within those two datasets, **only the non-reference cameras are broken**. `Blocks`'s reference camera is `cam9` and `Painter`'s is `v15` (`/home/sjpark/Documents/SCSegmentation/docs/tools/ds_info.json`), and those two entries are bit-identical between stale and current. `cam0`/`cam4` and `v0`/`v6` are the broken ones.
- For `Blocks/cam0`, `Blocks/cam4` and `Painter/v0` the stale run reports **100 % of `(frame, object)` mask files unmatched** (`missing_files` == objects × frames), and every per-object J is 0.0 except the handful of entries that are non-zero purely because the GT is empty in some frames (`crop_box → None → J = F = 1.0`, `eval_jf.py:170-173`) — the same non-zero values, 0.190 / 0.048 / 0.095, appear in the *current* file at the same object indices. `Painter/v6` had 162/546 unmatched yet still J ≈ 0 on every genuinely-scored object, i.e. files were present but did not correspond to that camera.

**Honest limit:** the buggy mask output was overwritten in place (`Blocks/SegMaskSam3OneStage/cam0/0` dir mtime 2026‑09‑03 10:44, its PNGs 10:50 — *after* the stale eval at 10:42), and `SCSam3/runMVSeg.py` is untracked, so the pre-fix source is unavailable. I can state the symptom and the `prefix1: 0` correlation as fact; I **cannot** state the exact line that was wrong.

**Conclusion: do not use this file.** `jf_sam3_onestage.json` supersedes it.

---

### 5. Two contradictions you must know before quoting the tables

**(i) `SegMaskSam3MVOpt` ≡ `SegMaskSam3OneStageNew`, numerically.**
`jf_sam3_onestagenew_full.json` and `jf_sam3_mvopt_all.json` have **byte-identical `result` blocks for all 45 `(dataset, camera)` entries** — only the `method` string differs. This is not a bookkeeping error: I md5-compared every prediction PNG on all 12 datasets where both output folders exist and they are **identical file-for-file** (630 / 1890 / 882 / 1441 / 1386 / 483 / 504 / 315 / 1323 / 1491 / 1406 / 941 PNGs respectively, zero differing hashes). This is consistent with the code comment "*MVOpt is OneStageNew with the memory fixes*" (`SCSam3/runMVSeg.py:44-46`) being value-neutral. **The two SAM 3 "New" columns below are therefore one result, reported twice.**

**(ii) `SegMaskSam3OneStageNew` "exported only" is `nan` on 15 datasets — a directory-listing artifact, not a score.**
`report_jf.py:27-49` derives "absent" from the live directory listing. The `SegMaskSam3OneStageNew` folders for `AlexaMeadeExhibit`, `CoffeeMartini` and `FlameSteak` **no longer exist on disk**, although `jf_sam3_onestagenew_full.json` still holds their scores. So every object in those 9 cameras is flagged absent (370 of the method's 427 flagged `(cam, object)` pairs), the "exported only" lists come out empty, and `mean([])` yields `nan`. On the 12 datasets where the folders do exist, the absent sets of all three SAM 3 methods are **identical** (57 `(cam, object)` pairs each).

Absent-object counts used by the "exported only" aggregation (from `find_absent` over the 4 current raw files):

| Method | cameras with absent objects | absent (cam, obj) pairs | total (cam, obj) pairs |
|---|---|---|---|
| SegMask | 8 | 26 | 1031 |
| SegMask1 | 0 | 0 | 1031 |
| SegMaskNew | 7 | 15 | 1031 |
| SegMaskNew1 / New2 / New3 | 0 | 0 | 1031 |
| SegMaskSam3OneStage | 20 | 91 | 1031 |
| SegMaskSam3OneStageNew | 25 | **427** (370 of them the artifact above) | 1031 |
| SegMaskSam3MVOpt | 21 | 94 | 1031 |

**Recommendation: quote the `as-is` numbers.** They are the DAVIS protocol, they are computed from the stored raw scores alone, and they are immune to the deleted-directory artifact.

---

### 6. Exact commands (reproducible)

Run from `/home/sjpark/Documents/SCSegmentation/Data/MVSeg`, system `python3` 3.12.3, no third-party packages needed.

```bash
cd /home/sjpark/Documents/SCSegmentation/Data/MVSeg

# (a) all 15 MVSeg datasets
python3 report_jf.py \
  --raw jf_raw.json jf_sam3_onestage.json jf_sam3_onestagenew_full.json jf_sam3_mvopt_all.json \
  --methods SegMask SegMask1 SegMaskNew SegMaskNew1 SegMaskNew2 SegMaskNew3 \
            SegMaskSam3OneStage SegMaskSam3OneStageNew SegMaskSam3MVOpt \
  --no-missing

# (b) 12-dataset common subset, via --common
#     (uses the 12-dataset jf_sam3_onestagenew.json so the intersection is 12)
python3 report_jf.py \
  --raw jf_raw.json jf_sam3_onestage.json jf_sam3_onestagenew.json jf_sam3_mvopt_all.json \
  --methods SegMask SegMask1 SegMaskNew SegMaskNew1 SegMaskNew2 SegMaskNew3 \
            SegMaskSam3OneStage SegMaskSam3OneStageNew SegMaskSam3MVOpt \
  --common --no-missing

# (b') same 12 datasets named explicitly, from the 15-dataset files — verified
#      byte-identical output to (b) apart from the "common subset:" banner line
python3 report_jf.py \
  --raw jf_raw.json jf_sam3_onestage.json jf_sam3_onestagenew_full.json jf_sam3_mvopt_all.json \
  --methods SegMask SegMask1 SegMaskNew SegMaskNew1 SegMaskNew2 SegMaskNew3 \
            SegMaskSam3OneStage SegMaskSam3OneStageNew SegMaskSam3MVOpt \
  --datasets AlexaMeadeFacePaint Barn Blocks Breakfast Carpark Dog Fencing Frog \
             MATF Painter PoznanStreet Welder \
  --no-missing
```

`--common` on run (b) prints: `common subset: 12 datasets   (dropped: AlexaMeadeExhibit, CoffeeMartini, FlameSteak)`.

Column legend used below — **SAM 2**: `M` = SegMask, `M1` = SegMask1, `MN` = SegMaskNew, `MN1` = SegMaskNew1, `MN2` = SegMaskNew2, `MN3` = SegMaskNew3. **SAM 3**: `S3-1S` = SegMaskSam3OneStage, `S3-1SN` = SegMaskSam3OneStageNew, `S3-MVO` = SegMaskSam3MVOpt.

---

### 7 (a). 15 datasets — `as-is`, all frames

**J&F**

| dataset | M | M1 | MN | MN1 | MN2 | MN3 | S3‑1S | S3‑1SN | S3‑MVO |
|---|---|---|---|---|---|---|---|---|---|
| AlexaMeadeExhibit | 0.8910 | 0.8910 | 0.8913 | 0.8913 | 0.8916 | 0.8919 | 0.8618 | 0.8618 | 0.8618 |
| AlexaMeadeFacePaint | 0.8385 | 0.7843 | 0.8384 | 0.7845 | 0.7844 | 0.7844 | 0.7927 | 0.7928 | 0.7928 |
| Barn | 0.8567 | 0.8567 | 0.8569 | 0.8569 | 0.8569 | 0.8569 | 0.8858 | 0.8865 | 0.8865 |
| Blocks | 0.8248 | 0.7487 | 0.8243 | 0.7488 | 0.7487 | 0.7487 | 0.7484 | 0.7478 | 0.7478 |
| Breakfast | 0.7178 | 0.7178 | 0.7138 | 0.7138 | 0.7160 | 0.7167 | 0.7419 | 0.7421 | 0.7421 |
| Carpark | 0.9417 | 0.9417 | 0.9416 | 0.9416 | 0.9420 | 0.9420 | 0.9438 | 0.9434 | 0.9434 |
| CoffeeMartini | 0.8362 | 0.8362 | 0.8380 | 0.8380 | 0.8365 | 0.8352 | 0.8461 | 0.8440 | 0.8440 |
| Dog | 0.7662 | 0.9089 | 0.7654 | 0.9075 | 0.9086 | 0.9087 | 0.9241 | 0.9241 | 0.9241 |
| Fencing | 0.9273 | 0.9273 | 0.9321 | 0.9321 | 0.9337 | 0.9337 | 0.8862 | 0.9112 | 0.9112 |
| FlameSteak | 0.4627 | 0.7835 | 0.4627 | 0.7832 | 0.7833 | 0.7833 | 0.7736 | 0.7734 | 0.7734 |
| Frog | 0.9734 | 0.9734 | 0.9754 | 0.9754 | 0.9756 | 0.9755 | 0.9753 | 0.9743 | 0.9743 |
| MATF | 0.6108 | 0.6649 | 0.6649 | 0.6649 | 0.6650 | 0.6649 | 0.6840 | 0.6841 | 0.6841 |
| Painter | 0.8341 | 0.8618 | 0.8329 | 0.8713 | 0.8619 | 0.8618 | 0.8682 | 0.8681 | 0.8681 |
| PoznanStreet | 0.8543 | 0.8543 | 0.8646 | 0.8646 | 0.8672 | 0.8672 | 0.8662 | 0.8787 | 0.8787 |
| Welder | 0.8781 | 0.8781 | 0.8778 | 0.8778 | 0.8781 | 0.8784 | 0.8419 | 0.8437 | 0.8437 |
| **AVERAGE** | **0.8142** | **0.8419** | **0.8187** | **0.8434** | **0.8433** | **0.8433** | **0.8427** | **0.8451** | **0.8451** |

**J**

| dataset | M | M1 | MN | MN1 | MN2 | MN3 | S3‑1S | S3‑1SN | S3‑MVO |
|---|---|---|---|---|---|---|---|---|---|
| AlexaMeadeExhibit | 0.8200 | 0.8200 | 0.8206 | 0.8206 | 0.8210 | 0.8217 | 0.7970 | 0.7971 | 0.7971 |
| AlexaMeadeFacePaint | 0.8209 | 0.7751 | 0.8210 | 0.7752 | 0.7751 | 0.7751 | 0.7781 | 0.7784 | 0.7784 |
| Barn | 0.8074 | 0.8074 | 0.8079 | 0.8079 | 0.8079 | 0.8079 | 0.8355 | 0.8353 | 0.8353 |
| Blocks | 0.7954 | 0.7298 | 0.7953 | 0.7303 | 0.7300 | 0.7300 | 0.7282 | 0.7272 | 0.7272 |
| Breakfast | 0.6807 | 0.6807 | 0.6772 | 0.6772 | 0.6792 | 0.6799 | 0.6970 | 0.6976 | 0.6976 |
| Carpark | 0.9030 | 0.9030 | 0.9030 | 0.9030 | 0.9037 | 0.9037 | 0.9036 | 0.9029 | 0.9029 |
| CoffeeMartini | 0.8067 | 0.8067 | 0.8084 | 0.8084 | 0.8071 | 0.8060 | 0.8149 | 0.8142 | 0.8142 |
| Dog | 0.7782 | 0.9086 | 0.7777 | 0.9072 | 0.9081 | 0.9083 | 0.9189 | 0.9194 | 0.9194 |
| Fencing | 0.8769 | 0.8769 | 0.8809 | 0.8809 | 0.8826 | 0.8825 | 0.8342 | 0.8561 | 0.8561 |
| FlameSteak | 0.4472 | 0.7535 | 0.4471 | 0.7532 | 0.7534 | 0.7534 | 0.7361 | 0.7355 | 0.7355 |
| Frog | 0.9726 | 0.9726 | 0.9739 | 0.9739 | 0.9744 | 0.9744 | 0.9745 | 0.9739 | 0.9739 |
| MATF | 0.5765 | 0.6232 | 0.6234 | 0.6234 | 0.6234 | 0.6234 | 0.6405 | 0.6407 | 0.6407 |
| Painter | 0.8074 | 0.8403 | 0.8052 | 0.8500 | 0.8404 | 0.8404 | 0.8454 | 0.8451 | 0.8451 |
| PoznanStreet | 0.8106 | 0.8106 | 0.8205 | 0.8205 | 0.8230 | 0.8231 | 0.8205 | 0.8320 | 0.8320 |
| Welder | 0.8561 | 0.8561 | 0.8559 | 0.8559 | 0.8561 | 0.8562 | 0.7969 | 0.8008 | 0.8008 |
| **AVERAGE** | **0.7840** | **0.8110** | **0.7879** | **0.8125** | **0.8124** | **0.8124** | **0.8081** | **0.8104** | **0.8104** |

**F**

| dataset | M | M1 | MN | MN1 | MN2 | MN3 | S3‑1S | S3‑1SN | S3‑MVO |
|---|---|---|---|---|---|---|---|---|---|
| AlexaMeadeExhibit | 0.9621 | 0.9621 | 0.9621 | 0.9621 | 0.9621 | 0.9621 | 0.9267 | 0.9265 | 0.9265 |
| AlexaMeadeFacePaint | 0.8560 | 0.7936 | 0.8558 | 0.7938 | 0.7937 | 0.7937 | 0.8072 | 0.8072 | 0.8072 |
| Barn | 0.9061 | 0.9061 | 0.9059 | 0.9059 | 0.9059 | 0.9060 | 0.9361 | 0.9376 | 0.9376 |
| Blocks | 0.8543 | 0.7675 | 0.8533 | 0.7673 | 0.7675 | 0.7675 | 0.7685 | 0.7683 | 0.7683 |
| Breakfast | 0.7549 | 0.7549 | 0.7505 | 0.7505 | 0.7529 | 0.7535 | 0.7867 | 0.7867 | 0.7867 |
| Carpark | 0.9803 | 0.9803 | 0.9802 | 0.9802 | 0.9804 | 0.9804 | 0.9839 | 0.9839 | 0.9839 |
| CoffeeMartini | 0.8658 | 0.8658 | 0.8675 | 0.8675 | 0.8659 | 0.8644 | 0.8772 | 0.8739 | 0.8739 |
| Dog | 0.7543 | 0.9093 | 0.7532 | 0.9078 | 0.9090 | 0.9090 | 0.9292 | 0.9287 | 0.9287 |
| Fencing | 0.9777 | 0.9777 | 0.9832 | 0.9832 | 0.9847 | 0.9848 | 0.9382 | 0.9663 | 0.9663 |
| FlameSteak | 0.4783 | 0.8135 | 0.4784 | 0.8133 | 0.8132 | 0.8132 | 0.8111 | 0.8113 | 0.8113 |
| Frog | 0.9741 | 0.9741 | 0.9768 | 0.9768 | 0.9769 | 0.9767 | 0.9761 | 0.9748 | 0.9748 |
| MATF | 0.6451 | 0.7065 | 0.7063 | 0.7063 | 0.7067 | 0.7065 | 0.7275 | 0.7275 | 0.7275 |
| Painter | 0.8609 | 0.8833 | 0.8607 | 0.8926 | 0.8834 | 0.8833 | 0.8910 | 0.8912 | 0.8912 |
| PoznanStreet | 0.8980 | 0.8980 | 0.9088 | 0.9088 | 0.9114 | 0.9114 | 0.9119 | 0.9254 | 0.9254 |
| Welder | 0.9001 | 0.9001 | 0.8997 | 0.8997 | 0.9001 | 0.9005 | 0.8869 | 0.8866 | 0.8866 |
| **AVERAGE** | **0.8445** | **0.8728** | **0.8495** | **0.8744** | **0.8743** | **0.8742** | **0.8772** | **0.8797** | **0.8797** |

**`exported only`, 15 datasets, J&F** (per-dataset rows for the 3 dropped-folder datasets are unusable — see §5(ii)):

| dataset | M | M1 | MN | MN1 | MN2 | MN3 | S3‑1S | S3‑1SN | S3‑MVO |
|---|---|---|---|---|---|---|---|---|---|
| AlexaMeadeExhibit | 0.8910 | 0.8910 | 0.8913 | 0.8913 | 0.8916 | 0.8919 | 0.8618 | **nan** | 0.8905 |
| AlexaMeadeFacePaint | 0.9223 | 0.7843 | 0.9222 | 0.7845 | 0.7844 | 0.7844 | 0.9342 | 0.9344 | 0.9344 |
| Barn | 0.8567 | 0.8567 | 0.8569 | 0.8569 | 0.8569 | 0.8569 | 0.8858 | 0.8865 | 0.8865 |
| Blocks | 0.8410 | 0.7487 | 0.8405 | 0.7488 | 0.7487 | 0.7487 | 0.9186 | 0.9179 | 0.9179 |
| Breakfast | 0.7178 | 0.7178 | 0.7138 | 0.7138 | 0.7160 | 0.7167 | 0.8737 | 0.8740 | 0.8740 |
| Carpark | 0.9417 | 0.9417 | 0.9416 | 0.9416 | 0.9420 | 0.9420 | 0.9438 | 0.9434 | 0.9434 |
| CoffeeMartini | 0.8362 | 0.8362 | 0.8380 | 0.8380 | 0.8365 | 0.8352 | 0.9379 | **nan** | 0.9357 |
| Dog | 0.8812 | 0.9089 | 0.8803 | 0.9075 | 0.9086 | 0.9087 | 0.9241 | 0.9241 | 0.9241 |
| Fencing | 0.9273 | 0.9273 | 0.9321 | 0.9321 | 0.9337 | 0.9337 | 0.8862 | 0.9112 | 0.9112 |
| FlameSteak | 0.4883 | 0.7835 | 0.4883 | 0.7832 | 0.7833 | 0.7833 | 0.9094 | **nan** | 0.9091 |
| Frog | 0.9734 | 0.9734 | 0.9754 | 0.9754 | 0.9756 | 0.9755 | 0.9753 | 0.9743 | 0.9743 |
| MATF | 0.7126 | 0.6649 | 0.6649 | 0.6649 | 0.6650 | 0.6649 | 0.9032 | 0.9033 | 0.9033 |
| Painter | 0.8450 | 0.8618 | 0.8438 | 0.8713 | 0.8619 | 0.8618 | 0.9366 | 0.9365 | 0.9365 |
| PoznanStreet | 0.8543 | 0.8543 | 0.8646 | 0.8646 | 0.8672 | 0.8672 | 0.8913 | 0.9042 | 0.9042 |
| Welder | 0.8781 | 0.8781 | 0.8778 | 0.8778 | 0.8781 | 0.8784 | 0.8419 | 0.8437 | 0.8437 |
| **AVERAGE** | **0.8378** | **0.8419** | **0.8354** | **0.8434** | **0.8433** | **0.8433** | **0.9083** | **nan** | **0.9126** |

Corresponding `exported only` J averages (15 ds): 0.8070 / 0.8110 / 0.8045 / 0.8125 / 0.8124 / 0.8124 / 0.8707 / nan / 0.8748; F averages: 0.8686 / 0.8728 / 0.8664 / 0.8744 / 0.8743 / 0.8742 / 0.9458 / nan / 0.9503.

---

### 7 (b). 12-dataset common subset — `as-is`, all frames

Dropped: `AlexaMeadeExhibit`, `CoffeeMartini`, `FlameSteak`. **Verified fact:** `collect()` aggregates per `(dataset, method)` independently (`report_jf.py:52-70`), so every per-dataset row is numerically identical to the corresponding row in §7(a) — only the AVERAGE row changes. J&F rows:

| dataset | M | M1 | MN | MN1 | MN2 | MN3 | S3‑1S | S3‑1SN | S3‑MVO |
|---|---|---|---|---|---|---|---|---|---|
| AlexaMeadeFacePaint | 0.8385 | 0.7843 | 0.8384 | 0.7845 | 0.7844 | 0.7844 | 0.7927 | 0.7928 | 0.7928 |
| Barn | 0.8567 | 0.8567 | 0.8569 | 0.8569 | 0.8569 | 0.8569 | 0.8858 | 0.8865 | 0.8865 |
| Blocks | 0.8248 | 0.7487 | 0.8243 | 0.7488 | 0.7487 | 0.7487 | 0.7484 | 0.7478 | 0.7478 |
| Breakfast | 0.7178 | 0.7178 | 0.7138 | 0.7138 | 0.7160 | 0.7167 | 0.7419 | 0.7421 | 0.7421 |
| Carpark | 0.9417 | 0.9417 | 0.9416 | 0.9416 | 0.9420 | 0.9420 | 0.9438 | 0.9434 | 0.9434 |
| Dog | 0.7662 | 0.9089 | 0.7654 | 0.9075 | 0.9086 | 0.9087 | 0.9241 | 0.9241 | 0.9241 |
| Fencing | 0.9273 | 0.9273 | 0.9321 | 0.9321 | 0.9337 | 0.9337 | 0.8862 | 0.9112 | 0.9112 |
| Frog | 0.9734 | 0.9734 | 0.9754 | 0.9754 | 0.9756 | 0.9755 | 0.9753 | 0.9743 | 0.9743 |
| MATF | 0.6108 | 0.6649 | 0.6649 | 0.6649 | 0.6650 | 0.6649 | 0.6840 | 0.6841 | 0.6841 |
| Painter | 0.8341 | 0.8618 | 0.8329 | 0.8713 | 0.8619 | 0.8618 | 0.8682 | 0.8681 | 0.8681 |
| PoznanStreet | 0.8543 | 0.8543 | 0.8646 | 0.8646 | 0.8672 | 0.8672 | 0.8662 | 0.8787 | 0.8787 |
| Welder | 0.8781 | 0.8781 | 0.8778 | 0.8778 | 0.8781 | 0.8784 | 0.8419 | 0.8437 | 0.8437 |
| **AVERAGE (J&F)** | **0.8353** | **0.8431** | **0.8407** | **0.8449** | **0.8449** | **0.8449** | **0.8465** | **0.8497** | **0.8497** |
| **AVERAGE (J)** | **0.8071** | **0.8153** | **0.8118** | **0.8171** | **0.8170** | **0.8171** | **0.8145** | **0.8175** | **0.8175** |
| **AVERAGE (F)** | **0.8635** | **0.8709** | **0.8695** | **0.8727** | **0.8727** | **0.8728** | **0.8786** | **0.8820** | **0.8820** |

`exported only`, 12 datasets — per-dataset J&F: AlexaMeadeFacePaint 0.9223/0.7843/0.9222/0.7845/0.7844/0.7844/0.9342/0.9344/0.9344; Barn, Carpark, Fencing, Frog, Welder identical to `as-is`; Blocks 0.8410/…/0.9186/0.9179/0.9179; Breakfast …/0.8737/0.8740/0.8740; Dog 0.8812/…/0.9241; MATF 0.7126/…/0.9032/0.9033/0.9033; Painter 0.8450/…/0.9366/0.9365/0.9365; PoznanStreet …/0.8913/0.9042/0.9042. Averages: **J&F 0.8626 / 0.8431 / 0.8595 / 0.8449 / 0.8449 / 0.8449 / 0.9096 / 0.9128 / 0.9128**; J 0.8339 / 0.8153 / 0.8306 / 0.8171 / 0.8170 / 0.8171 / 0.8746 / 0.8776 / 0.8776; F 0.8913 / 0.8709 / 0.8884 / 0.8727 / 0.8727 / 0.8728 / 0.9445 / 0.9479 / 0.9479. No `nan` here.

---

### 7 (c). Per-method J&F averages — all four aggregation combinations

**15 datasets**

| | M | M1 | MN | MN1 | MN2 | MN3 | S3‑1S | S3‑1SN | S3‑MVO |
|---|---|---|---|---|---|---|---|---|---|
| all frames, as-is | 0.8142 | 0.8419 | 0.8187 | 0.8434 | 0.8433 | 0.8433 | 0.8427 | 0.8451 | 0.8451 |
| all frames, exported only | 0.8378 | 0.8419 | 0.8354 | 0.8434 | 0.8433 | 0.8433 | 0.9083 | nan | 0.9126 |
| DAVIS (drop first/last), as-is | 0.8135 | 0.8410 | 0.8179 | 0.8425 | 0.8424 | 0.8424 | 0.8415 | 0.8439 | 0.8439 |
| DAVIS, exported only | 0.8371 | 0.8410 | 0.8346 | 0.8425 | 0.8424 | 0.8424 | 0.9073 | nan | 0.9117 |

**12-dataset common subset**

| | M | M1 | MN | MN1 | MN2 | MN3 | S3‑1S | S3‑1SN | S3‑MVO |
|---|---|---|---|---|---|---|---|---|---|
| all frames, as-is | 0.8353 | 0.8431 | 0.8407 | 0.8449 | 0.8449 | 0.8449 | 0.8465 | 0.8497 | 0.8497 |
| all frames, exported only | 0.8626 | 0.8431 | 0.8595 | 0.8449 | 0.8449 | 0.8449 | 0.9096 | 0.9128 | 0.9128 |
| DAVIS (drop first/last), as-is | 0.8344 | 0.8421 | 0.8397 | 0.8438 | 0.8438 | 0.8438 | 0.8451 | 0.8484 | 0.8484 |
| DAVIS, exported only | 0.8617 | 0.8421 | 0.8585 | 0.8438 | 0.8438 | 0.8438 | 0.9085 | 0.9118 | 0.9118 |

---

### 8. Open / ambiguous items

1. **MVOpt vs OneStageNew are the same numbers and the same PNGs** (§5(i)). Whether that is the intended "value-neutral memory fix" or an accidental relabel cannot be settled from the data alone — but the byte-identical masks on all 12 comparable datasets make an accidental JSON relabel unlikely, since the two output *folders* were written 21 hours apart (`Barn/SegMaskSam3OneStageNew` 2026‑09‑03 09:34 vs `Barn/SegMaskSam3MVOpt` 2026‑09‑04 06:12).
2. **`SegMaskSam3OneStageNew` masks for `AlexaMeadeExhibit`, `CoffeeMartini`, `FlameSteak` are gone from disk** while `jf_sam3_onestagenew_full.json` still reports them. Those three rows cannot be re-verified and their "exported only" values are `nan`.
3. **`SegMaskSam3OneStage_recheck` (15 datasets, 2026‑09‑04 08:31–08:50) has never been scored** — no `jf_*.json` mentions it. Byte-identical to `SegMaskSam3OneStage` on `Barn`; not checked on the other 14.
4. **`SegMaskSam3ForSam2New`** (CoffeeMartini only, 198 PNGs, 2026‑09‑04 09:14) is likewise unscored.
5. The exact code defect behind the `.STALE-prefix-bug.json` file is **not recoverable** (§4) — the buggy masks were overwritten and `SCSam3/runMVSeg.py` is untracked.