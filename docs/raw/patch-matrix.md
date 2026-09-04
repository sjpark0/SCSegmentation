## Scope & method

Seven package folders exist under `/home/sjpark/Documents/SCSegmentation/SCSam3/`. All Python lives at depth 1 (`find … -mindepth 2 -name '*.py'` returns nothing outside `__pycache__`); the numbered dirs `0/`…`31/` are PNG output. Counts below come from `grep -c` on exact patterns, per file.

## Raw marker counts per file

Only files with ≥1 hit are listed. `S2a` counts the *presence* of the line `inference_state["mask_inputs_per_obj"][obj_id][frame_idx] = mask` — the S2 change is its **absence**, so `S2a=0` inside an applicable file means the change is applied.

| Folder | File | S1 | S2a | S3 | S4 | S5 | S6 | S7 |
|---|---|---|---|---|---|---|---|---|
| ForSam2 | `SCSam3TrackerPredictor.py` | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| ForSam2 | `SCSam3VideoInference.py` | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| ForSam2 | `io_utils.py` | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| ForSam2New | `SCSam3TrackerPredictor.py` | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| ForSam2New | `SCSam3TrackerPredictorNewMem.py` | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| ForSam2New | `SCSam3VideoInference.py` | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| ForSam2New | `io_utils.py` | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| MVOpt | `SCSam3TrackerPredictor.py` | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| MVOpt | `SCSam3TrackerPredictorNewMem.py` | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| MVOpt | `SCSam3Video.py` | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| MVOpt | `SCSam3VideoInference.py` | **3** | 0 | 0 | 0 | 0 | 0 | 0 |
| MVOpt | `SCSam3VideoInferenceNewMem.py` | **3** | 0 | 0 | 0 | 0 | 0 | 6 |
| MVOpt | `io_utils.py` | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| OneStage | `SCSam3TrackerPredictor.py` | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| OneStage | `SCSam3Video.py` | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| OneStage | `SCSam3VideoInference.py` | **3** | 0 | 0 | 0 | 0 | 0 | 0 |
| OneStage | `io_utils.py` | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| **OneStageNew** | `SCSam3VideoInference.py` | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| **OneStageNew** | `SCSam3VideoInferenceNewMem.py` | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| TwoStage | `SCSam3TrackerPredictor.py` | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| TwoStage | `SCSam3VideoInference.py` | **3** | 0 | 0 | 0 | 0 | 0 | 0 |
| TwoStage | `io_utils.py` | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| TwoStageNew | `SCSam3TrackerPredictor.py` | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| TwoStageNew | `SCSam3TrackerPredictorNewMem.py` | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| TwoStageNew | `SCSam3VideoInference.py` | **3** | 0 | 0 | 0 | 0 | 0 | 0 |
| TwoStageNew | `io_utils.py` | 0 | 0 | 0 | 0 | 1 | 0 | 0 |

**S1 counting caveat (important):** a count of `_THIS_FRAME_HAS_OUTPUTS_` = 1 is **baseline**, not the change. Every folder has one upstream occurrence in `_run_single_frame_inference` (`demoSCSam3OneStageNew/SCSam3VideoInference.py:398`, `demoSCSam3ForSam2/SCSam3VideoInference.py:406`). The S1 *change* adds **two more** — one at the tail of `add_tracker_new_points` and one at the tail of `add_tracker_new_mask` — so `count == 3` means PRESENT and `count == 1` means ABSENT.

## Presence matrix

| Folder | S1 sentinel | S2 maskinputs | S3 meta | S4 nonoverlap | S5 fp32 | S6 retire | S7 trim |
|---|---|---|---|---|---|---|---|
| `demoSCSam3ForSam2` | N/A | N/A | PRESENT | PRESENT | PRESENT | ABSENT | N/A |
| `demoSCSam3ForSam2New` | N/A | N/A | PRESENT | PRESENT | PRESENT | ABSENT | N/A |
| `demoSCSam3MVOpt` | PRESENT | PRESENT | PRESENT | PRESENT | PRESENT | PRESENT | PRESENT |
| `demoSCSam3OneStage` | PRESENT | PRESENT | PRESENT | PRESENT | PRESENT | PRESENT (inert, see below) | N/A |
| `demoSCSam3OneStageNew` | ABSENT | ABSENT | ABSENT | ABSENT | ABSENT | ABSENT | ABSENT |
| `demoSCSam3TwoStage` | PRESENT | PRESENT | PRESENT | PRESENT | PRESENT | ABSENT | N/A |
| `demoSCSam3TwoStageNew` | PRESENT | PRESENT | PRESENT | PRESENT | PRESENT | ABSENT | N/A |

**`demoSCSam3MVOpt` is the only folder carrying all seven.** `demoSCSam3OneStageNew` carries none — it is the unpatched reference.

### Why the NOT-APPLICABLEs

- **S1 / S2 in `ForSam2`, `ForSam2New`** — the anchors do not exist. `def add_tracker_new_mask` is absent from both (`grep -rn "def add_tracker"`: only `add_tracker_new_points` at `demoSCSam3ForSam2/SCSam3VideoInference.py:1406`). And their `add_tracker_new_points` never touches `previous_stages_out` at all: the only occurrences in `demoSCSam3ForSam2/SCSam3VideoInference.py` are lines 105, 172, 196, 232, 233, 241, 406 — all in the upstream init/propagate code, none in the prompt path. The obj-registration block that S2 patches (`inference_state["mask_inputs_per_obj"][obj_id] = {}`) is likewise absent; the file's single `mask_inputs_per_obj` hit is a *read* on a `tracker_state` at line 1693, not the video-level dict. Structurally these are the SAM2-style packages (see below) whose tail of `add_tracker_new_points` goes straight from `_build_tracker_output` to `_cache_frame_outputs` with no ledger bookkeeping (`demoSCSam3ForSam2/SCSam3VideoInference.py:1586-1613`).
- **S7 in `ForSam2`, `ForSam2New`, `OneStage`, `TwoStage`, `TwoStageNew`** — the change lives in `SCSam3VideoInferenceNewMem.py`, which exists only in `demoSCSam3MVOpt` and `demoSCSam3OneStageNew`. The other five have no `*NewMem` inference module and therefore no multi-session `propagate_in_video` loop to hang the trim on. (They do have a `cached_frame_outputs` cache in their single-session `SCSam3VideoInference.py` — 23 hits in OneStage/TwoStage/TwoStageNew, 11 in ForSam2* — so the idea is portable, but the marker's host file does not exist.)

### S6 nuance — `demoSCSam3OneStage` is a partial/inert hit

`demoSCSam3OneStage/SCSam3Video.py:31` defines `RetireSpatialPredictor`, byte-for-byte the same body as MVOpt's, **but that folder has no spatial predictor**: `__init__` (lines 16-29) builds only `self.predictor`, and the sole `self.predictor_spatial =` in the file is `= None` at line 63 inside the method itself. So `getattr(self, "predictor_spatial", None)` is `None` and the method returns at line 48 without freeing anything. `uses_spatial_predictor` appears only inside the docstring (line 42), never as an assignment. `demoSCSam3MVOpt/SCSam3Video.py` has both halves for real: `self.uses_spatial_predictor = True` at line 23 and a working `RetireSpatialPredictor` at line 37 that closes `session_id_statial`, nulls `predictor.model`, and calls `gc.collect()` + `torch.cuda.empty_cache()`.

For `ForSam2`, `ForSam2New`, `TwoStage`, `TwoStageNew` the change *is* applicable — each really does construct a separate `self.predictor_spatial` (`demoSCSam3ForSam2/SCSam3Video.py:19`, `demoSCSam3TwoStage/SCSam3Video.py:22`) — it simply has not been applied.

## Anchor evidence (exact hunks)

`diff demoSCSam3OneStageNew/SCSam3TrackerPredictor.py demoSCSam3MVOpt/SCSam3TrackerPredictor.py` yields exactly two hunks — S3 and S4, nothing else:

- **S3** replaces `mask_inputs_per_frame[frame_idx] = mask_inputs_video_res` (OneStageNew line 407) with `torch.empty(mask_inputs_video_res.shape, dtype=torch.bool, device="meta")` (MVOpt lines 416-417). This is why `mask_inputs_per_frame` counts 6 in OneStageNew vs 7 in every patched tracker.
- **S4** inserts the bool fast path before the general `torch.where`, at `demoSCSam3MVOpt/SCSam3TrackerPredictor.py:1412-1416`, inside `_apply_object_wise_non_overlapping_constraints` (def at line 1387). Note the task named the anchor `_apply_non_overlapping_constraints`; no such `def` exists in any folder — that name is only the `super()` call at line 1397. The enclosing method is `_apply_object_wise_non_overlapping_constraints`, and it exists in **all 12** tracker files across all 7 folders, so S4 is applicable everywhere.

`diff demoSCSam3OneStageNew/SCSam3VideoInference.py demoSCSam3MVOpt/SCSam3VideoInference.py` yields exactly three hunks — S1 twice (lines 1656-1665 → 1656-1661, and 1905-1914 → 1907-1912) and S2 once (line 1880 → 1876-1882, replaced by a 7-line `[MEM]` comment). The patched code carries a searchable `[MEM]` tag: 1 hit each in `MVOpt/SCSam3VideoInference.py`, `MVOpt/SCSam3VideoInferenceNewMem.py`, `OneStage/…`, `TwoStage/…`, `TwoStageNew/SCSam3VideoInference.py`; 0 in OneStageNew and 0 in ForSam2*.

`diff demoSCSam3OneStageNew/io_utils.py demoSCSam3MVOpt/io_utils.py` is a single 17-line insertion ending in `img = img.to(torch.float32)` (MVOpt line 222, ForSam2* line 223).

**S7** is `TRIM_CACHED_OUTPUTS` at `demoSCSam3MVOpt/SCSam3VideoInferenceNewMem.py:38` (env-gated on `SCSAM3_TRIM_CACHED_OUTPUTS`), `def _trim_cached_frame_outputs` at line 529, called at line 1316 inside the multi-session `propagate_in_video` (def line 1123). `demoSCSam3OneStageNew/SCSam3VideoInferenceNewMem.py` has the same multi-session `propagate_in_video` (def line 1087) and 23 `cached_frame_outputs` hits but zero trim hits. The gate is turned on externally at `SCSam3/runMVOptThree.sh:14`.

## Cross-check: MVOpt vs OneStageNew is *only* the memory work

`diff -rq --exclude=__pycache__ demoSCSam3OneStageNew demoSCSam3MVOpt` differs on exactly 5 `.py` files (`SCSam3TrackerPredictor.py`, `SCSam3TrackerPredictorNewMem.py`, `SCSam3Video.py`, `SCSam3VideoInference.py`, `SCSam3VideoInferenceNewMem.py`) plus one file only in OneStageNew: a stray backup `SCSam3TrackerPredictorNewMem copy.py` (never imported; 0 hits for any of the 7 markers). `build_scsam3.py`, `SCSam3VideoPredictor.py`, `SCSam3VideoPredictorNewMem.py`, `sam3_demoVideo.py`, `misc.py`, `test.py` are byte-identical (md5 match). Every differing hunk is one of S1–S7.

## What each folder IS (from its entry point `sam3_demoVideo.py` + `SCSam3Video.__init__`)

Baseline for comparison: **`demoSCSam3OneStageNew`** — two SAM 3 session predictors (`build_scsam3_video_predictor` for the cross-view pass at `SCSam3Video.py:16`, `build_scsam3_video_predictor_newmem` for the temporal pass at line 19); text prompt on view 0 (`sam3_demoVideo.py:24 sc.AddText(0,"person")`); temporal propagate is multi-session shared-memory (`session_ids=self.session_ids, spatial_idx=m`, `SCSam3Video.py:223-228`); output drained round-robin, 50 frames × 32 views interleaved (`sam3_demoVideo.py:50-59`).

| Folder | One-line identity vs OneStageNew |
|---|---|
| `demoSCSam3MVOpt` | Algorithmically identical to OneStageNew (same builders, byte-identical `sam3_demoVideo.py` and `build_scsam3.py`); it is OneStageNew **plus all seven memory changes**, notably the retired cross-view model. |
| `demoSCSam3OneStage` | Collapses the two predictors into **one** `build_scsam3_video_predictor` (`SCSam3Video.py:18`) used for both the cross-view and the temporal pass, so temporal propagate is single-session (`session_id=self.session_ids[m]`, line 256) and the output is drained **view-by-view, not interleaved** (`sam3_demoVideo.py:50-59`). |
| `demoSCSam3TwoStage` | Splits the two stages across two *different model types*: stage 1 is the full SAM 3 session predictor (`predictor_spatial`, text prompt via `handle_request`), stage 2 is the **raw tracker** `self.model.tracker` from `build_scsam3_video_model` with the SAM2-style `init_state`/`propagate_in_video` generator API (`SCSam3Video.py:18-23, 185, 205`); output view-by-view, tuple-unpacked (`sam3_demoVideo.py:51`). |
| `demoSCSam3TwoStageNew` | TwoStage but on `build_scsam3_video_model_newmem`, so the stage-2 tracker propagates over the **whole state list** with `spatial_idx=m` (shared cross-view memory) instead of one state at a time, seeds zero masks for objects missing in a view, and drains output interleaved 50×32 (`SCSam3Video.py:18, 197-212`; `sam3_demoVideo.py:51-58`). |
| `demoSCSam3ForSam2` | Pure SAM2-style package: **both** stages are `model.tracker` objects (`SCSam3Video.py:14-21`) — no sessions, no `handle_request` (0 hits), no `AddText` (0 hits), point prompts only; the cross-view stage is a second tracker whose `init_state(original_states=…, perms=…)` builds an N-view pseudo-video (line 43). |
| `demoSCSam3ForSam2New` | ForSam2 on `build_scsam3_video_model_newmem`; the only diffs are the builder and the shared-memory propagate signature (`self.inference_state, spatial_idx=m`, `SCSam3Video.py:151-154`) with interleaved 50×32 output (`sam3_demoVideo.py:52-59`). |

## Ambiguities / things worth flagging

1. **S2 is stated as an absence, which is unsafe as a lone grep.** In `ForSam2`/`ForSam2New` the line is absent because the entire enclosing method (`add_tracker_new_mask`) and the registration block do not exist — a naive "absence ⇒ applied" reading would score them PRESENT. I gated on the presence of `inference_state["mask_inputs_per_obj"][obj_id] = {}` (2 hits in each of the five SAM3-session folders, 0 in ForSam2*) to separate "removed" from "never existed".
2. **S6's two halves disagree in `demoSCSam3OneStage`** (method yes, flag no, and the method is a no-op). I scored it PRESENT per the task's literal "and/or" wording but it buys nothing at runtime. If the doc wants "actually frees memory", OneStage should read ABSENT.
3. **An unrelated pre-existing divergence in `io_utils.py`** that is *not* one of the seven: MVOpt/OneStage do `img_mean = img_mean.cuda()` (local rebind, lines 202-203) where TwoStage/TwoStageNew do `self.img_mean = self.img_mean.cuda()`. `md5` groups io_utils into four distinct variants; do not treat io_utils equality as an S5 proxy.
4. **Runner coverage is uneven.** `SCSam3/runMVSeg.py:38-40` wires only `OneStage`, `OneStageNew`, `MVOpt`; `SCSam3/runMVSegForSam2.py:26` wires only `ForSam2New`. `TwoStage`, `TwoStageNew`, `ForSam2` have no runner and are reachable only through their in-folder `sam3_demoVideo.py`.
5. **`demoSCSam3MVOpt` is entirely untracked in git** (0 tracked files, 12 untracked `.py`); every other demo folder is tracked (9–14 files each). The memory-fixed package is not yet committed.