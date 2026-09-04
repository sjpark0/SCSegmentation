## Verdict

The belief survives on the **numbers**, but the proposed action is still unsafe for **provenance**. Overwrite is not needed to get what the user wants, and it destroys the only on-disk evidence that currently backs the equivalence claim.

---

## 1. Actual git state

| Fact | Value |
|---|---|
| Repo / branch | `/home/sjpark/Documents/SCSegmentation`, `main` @ `99c05c7` ("추가", 2026-03-12) |
| Remote | `origin` → github.com/sjpark0/SCSegmentation, `main` **0 ahead / 0 behind** (pushed) |
| Tags | **none at all** |
| `demoSCSam3OneStageNew` | 13 tracked `.py`; all 13 blob hashes **match HEAD exactly** (clean) |
| Its git history | **exactly one commit** (`99c05c7`). No earlier version to fall back on |
| `demoSCSam3MVOpt` | **entirely untracked**, 868 KB, 13 files; only `__pycache__` is gitignored |

Two things the working tree hides:

- **OneStageNew is 13 GB on disk**, not 868 KB — subdirs `0/`…`31/` (50 PNGs each, root-owned, Mar 9) plus an untracked `.dockerignore`. All gitignored (`*.png`), so **none of it is recoverable from git**. An `rm -rf && cp -r` overwrite deletes 13 GB permanently; a `cp -r` overlay leaves it.
- **The tree is already dirty elsewhere.** Five sibling packages have uncommitted edits (`demoSCSam3OneStage/SCSam3Video.py`, `.../SCSam3TrackerPredictor.py`, TwoStage, TwoStageNew, ForSam2, ForSam2New), plus untracked `.github/`, `figures/`, `SCSam3/sam3/`, `.tmp_probe/`, and five `run*.sh`. A `git add -A` after the overwrite would sweep all of that into one commit.

**What git would record** for the overwrite itself — 6 modified, 4 unchanged, and one deletion:

```
MODIFIED   SCSam3TrackerPredictor.py, SCSam3TrackerPredictorNewMem.py, SCSam3Video.py,
           SCSam3VideoInference.py, SCSam3VideoInferenceNewMem.py, io_utils.py
unchanged  SCSam3VideoPredictor.py, SCSam3VideoPredictorNewMem.py, build_scsam3.py,
           misc.py, sam3_demoVideo.py, test.py
DELETED    "SCSam3TrackerPredictorNewMem copy.py"   <- tracked, 97418 B, no MVOpt counterpart
```

That last file is a real tracked file (differs from `SCSam3TrackerPredictorNewMem.py`) that MVOpt does not have. A `cp -r` leaves it behind as a stale orphan now inconsistent with its siblings; a directory replace deletes it.

Because the change lands as an ordinary modification of the same paths, **the name `OneStageNew` stops meaning "the baseline code"** with nothing in the tree marking the transition.

---

## 2. Are the published numbers still reproducible? — Yes, and there is hard evidence

Published baseline = `Data/MVSeg/jf_sam3_onestagenew.json`: 36 records, **12 datasets**, method `SegMaskSam3OneStageNew`. All 12 output dirs exist (122 MB, 12 692 PNGs).

I byte-compared every one against MVOpt:

```
IDENTICAL  AlexaMeadeFacePaint(630) Barn(1890) Blocks(882) Breakfast(1441) Carpark(1386)
IDENTICAL  Dog(483) Fencing(504) Frog(315) MATF(1323) Painter(1491) PoznanStreet(1406) Welder(941)
```

`diff -rq` reports **zero differences on all 12 datasets** — identical file counts, byte-identical PNGs. And this held with the memory path fully engaged: those MVOpt dirs came from `logs/mvopt-20260904-060418/` via `runMVOptThree.sh`, which sets `-e SCSAM3_TRIM_CACHED_OUTPUTS=1` — the one env-gated branch MVOpt adds (`demoSCSam3MVOpt/SCSam3VideoInferenceNewMem.py:38`, `:546`). That is the strongest equivalence evidence obtainable short of re-running.

**So the risk is not to the numbers. It is that this evidence is the only thing anchoring the claim, and it lives entirely outside git** — `Data` and `*.png` are both in `.gitignore`. The 122 MB of baseline masks are unbacked-up on one disk.

The three missing datasets are the reason MVOpt exists: OneStageNew died on all three (logs: `AlexaMeadeExhibit` at 0/22, `CoffeeMartini` stalled at 13/22, `FlameSteak` at 11/22 @ 407 s/it), and their `SegMaskSam3OneStageNew` dirs have since been deleted.

---

## 3. Provenance landmine found: `jf_sam3_onestagenew_full.json` is mislabeled

This file is **not** an evaluation of OneStageNew. Proven arithmetically:

- 45 records = 36 (`jf_sam3_onestagenew.json`) + 9 (`jf_sam3_mvopt.json`), exactly
- the 12-dataset records carry over **verbatim**
- the 3 gap datasets' records are **MVOpt's records verbatim**, with `method` rewritten to `"SegMaskSam3OneStageNew"`
- and those 3 datasets have **no `SegMaskSam3OneStageNew` directory on disk at all**

`jf_sam3_mvopt_all.json` is the same 45 records with the opposite label (`onestagenew_full == mvopt_all` ignoring `method`: **True**), so its 12-dataset half was never independently measured from MVOpt dirs either — it is correct only because the masks happen to be byte-identical.

Anyone reporting 15-dataset "OneStageNew" numbers from that file is reporting 3 MVOpt results under a false label. Today that is defensible because the byte-identity is checkable. **After the overwrite it is no longer checkable.**

---

## 4. Output-directory audit — `Data/MVSeg`

15 real datasets (plus 6 legacy `Blocks1*/Carpark1*/Fencing1*` with no SAM 3 output at all):

| dir | coverage | scored in |
|---|---|---|
| `SegMaskSam3OneStage` | 15/15 | `jf_sam3_onestage.json` |
| `SegMaskSam3OneStageNew` | **12/15** (no AlexaMeadeExhibit, CoffeeMartini, FlameSteak) | `jf_sam3_onestagenew.json` ← **published** |
| `SegMaskSam3MVOpt` | 15/15 | `jf_sam3_mvopt.json` (3) + `..._all.json` (15) |
| `SegMaskSam3OneStage_recheck` | 15/15, **214 MB** | **nothing — orphaned** |
| `SegMaskSam3ForSam2New` | 1/15 (CoffeeMartini) | **nothing — orphaned** |
| `SegMaskSam2Recheck` | 1/15 (CoffeeMartini) | `jf_recheck.json` |

`SegMaskSam3OneStage_recheck` (214 MB, Sep 4 08:30) was made by `demoSCSam3OneStage` via `runMVOptThree.sh` with `ALGO=OneStage OUT=SegMaskSam3OneStage_recheck` — already orphaned before any overwrite.

**What the overwrite makes ambiguous:** `runMVSeg.py` maps `--algo OneStageNew` → `demoSCSam3OneStageNew` and defaults `--out` to `SegMaskSam3OneStageNew`. After the copy, that command runs MVOpt code into the baseline's directory name. **Nothing on disk records which code wrote a mask dir** — no manifest, no commit hash, no run stamp; provenance is only mtimes and the gitignored `logs/` tree. `runMVSeg.py` does refuse a non-empty out dir without `--overwrite` — but `runMVOptThree.sh` passes `--overwrite` unconditionally and `waitAndRunMVSeg.sh` calls `runMVSegAll.sh --overwrite`. One careless rerun silently replaces the published baseline masks under the baseline's own name, unrecoverably.

Secondary: `ALGOS`/`DEFAULT_OUT` would then hold two keys pointing at byte-identical code but different output dirs — a duplicate guaranteed to drift.

---

## 5. Recommended sequencing

**Do not overwrite. Keep both — they cost 868 KB.**

1. **Tag the baseline first.** There are zero tags; the baseline's only anchor is a commit titled "추가".
   `git tag -a baseline-onestagenew-jf 99c05c7 -m "code behind jf_sam3_onestagenew.json (12/15 MVSeg)"` and push it.
2. **Commit MVOpt at its own path, with explicit paths — never `git add -A`** (five sibling packages have unrelated uncommitted edits):
   `git add SCSam3/demoSCSam3MVOpt SCSam3/runMVSeg.py SCSam3/runMVSegAll.sh SCSam3/runMVOptThree.sh SCSam3/waitAndRunMVSeg.sh`
3. **Snapshot the 122 MB of published masks outside the repo**, with `jf_sam3_onestagenew.json`. `Data` and `*.png` are gitignored; this is the sole copy of the evidence for the equivalence claim.
4. **Fix the mislabeled JSON now**, independent of the code decision: rename `jf_sam3_onestagenew_full.json` to something honest (`jf_sam3_onestagenew12_plus_mvopt3.json`) or regenerate it with truthful per-record `method` fields.
5. If the real goal is "stop maintaining two copies," deprecate rather than bury: drop the `OneStageNew` key from `ALGOS`, or leave the baseline files in place with a `DEPRECATED` note. Repointing `ALGOS["OneStageNew"] → demoSCSam3MVOpt` while `DEFAULT_OUT["OneStageNew"]` still says `SegMaskSam3OneStageNew` recreates exactly the ambiguity in §4.
6. **If the overwrite happens anyway:** copy file-by-file, never `rm -rf` the directory (13 GB of untracked demo output in `0/`…`31/` plus untracked `.dockerignore`), and make an explicit decision about `SCSam3/demoSCSam3OneStageNew/SCSam3TrackerPredictorNewMem copy.py`.

**Key paths:** `/home/sjpark/Documents/SCSegmentation/SCSam3/runMVSeg.py` (`ALGOS`, `DEFAULT_OUT`, the `hasattr(sc, "RetireSpatialPredictor")` guard), `/home/sjpark/Documents/SCSegmentation/SCSam3/runMVOptThree.sh:14-16`, `/home/sjpark/Documents/SCSegmentation/SCSam3/demoSCSam3MVOpt/SCSam3VideoInferenceNewMem.py:38`, `/home/sjpark/Documents/SCSegmentation/Data/MVSeg/eval_jf.py:199-203`, `/home/sjpark/Documents/SCSegmentation/Data/MVSeg/jf_sam3_onestagenew_full.json`.