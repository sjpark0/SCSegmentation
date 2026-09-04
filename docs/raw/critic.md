> **시점 안내 (2026-09-04).** 이 문서는 `demoSCSam3OneStageNew`가 **무패치 원본**이던 시점의
> 조사 결과입니다. 이후 그 폴더는 `demoSCSam3MVOpt`의 동결 스냅샷으로 바뀌었으므로,
> "OneStageNew에는 없다 / 무패치다" 류의 서술은 **현재 상태가 아닙니다.**
> 원본은 git 태그 `baseline-onestagenew`. 배경: `docs/README.md`

## COMPLETENESS CRITIQUE

### A. Change-list re-derivation — what the seven sites do not cover

I re-derived the diff independently (`diff -rq --exclude=__pycache__`, plus per-file md5). **The seven sites do cover all 14 hunks in the six `.py` files** — the hunk-level enumeration is complete. But the *change set* is larger than the diff file:

| Delta | Status | Impact |
|---|---|---|
| `demoSCSam3OneStageNew/.dockerignore` (`*` / `!*.py`) has **no counterpart in NEW** | **untracked in git** (`git status: ?? …/.dockerignore`) | An `rsync -a --delete` copy destroys it with no git baseline to restore from. `SCSam3/Dockerfile` only `COPY ./sam3`, so it is vestigial — but it is the one deleted file that is *not* recoverable. |
| `SCSam3TrackerPredictorNewMem copy.py` absent in NEW | tracked (13 files under `SCSam3/demoSCSam3OneStageNew/` are in the index) | Recoverable; never imported. Fine. |
| Output dirs `0/`…`31/` (50 PNGs each) absent in NEW | untracked (`.gitignore:183 *.png`) | Write-only output of `sam3_demoVideo.py:48` (`os.makedirs(f"{m}")`) and `:59` (`cv2.imwrite(f"{spatial_idx}/{frame_idx}.png")`). Regenerable. Fine. |

The one thing the *nonpy-diff* sweep should have flagged and evidently did not: **`.dockerignore` is the only deleted artifact with no recovery path.**

Also: **S3's own reader enumeration is incomplete and its line citations are stale.** The comment at `demoSCSam3MVOpt/SCSam3TrackerPredictorNewMem.py:407-415` lists five readers of `mask_inputs_per_frame`. It omits two: `_reset_tracking_results` at `:144` and at `:1042` (`for v in inference_state["mask_inputs_per_obj"].values(): v.clear()`). Both are `.clear()`, so the "keys only" conclusion survives — but the audit inherited a reader list the author had not actually completed. Every line number in that comment is also pre-edit (off by +11: cited `:267/:951/:759/:964/:1775/:1808` vs. actual `:267/:962/:770/:975/:1786/:1819`), so an auditor spot-checking the citations lands on the wrong lines.

**S1 verified independently and cleanly.** `previous_stages_out` is a **list**, not a dict (`SCSam3VideoInferenceNewMem.py:170`, `SCSam3VideoInference.py:164`: `[None] * num_frames`). The OLD `if frame_idx not in inference_state["previous_stages_out"]` was therefore a *value* membership test on a list, and its `else: .update(out)` branch was unreachable. Every reader (`:194`, `:231`, `:239`, and upstream `sam3/sam3/model/sam3_video_inference.py:190/227/235`) tests only `is None`. The NEW sentinel matches upstream `sam3_video_inference.py:400` exactly. No gap.

**S5 verified independently.** `img -= self.img_mean` appears at exactly one site in `io_utils.py` (`:204`); the other two loader classes (`AsyncVideoFrameLoader.__getitem__:118`, `AsyncVideoFrameLoaderFile.__getitem__:153`) return raw BGR uint8 and never normalize. So the cast at `:222` is not one of several normalization paths. Worth noting only that `AsyncVideoFrameCPUToGPU` is applied by `runMVSeg.py:143`, not by `SCSam3Video.LoadVideo_Folder:79` — after the copy, the in-package demo entry point is untouched by S5.

---

### B. Risk classes nobody looked for

**1. Post-retirement object state (S6) — the docstring's rationale is inverted for its only real caller.**
`demoSCSam3MVOpt/SCSam3Video.py:46-48` claims the attribute is set to `None` rather than deleted "so that an older caller sniffing `hasattr(self, "predictor_spatial")` still takes the multi-session branch." The actual caller in this repo is:

```
runMVSeg.py:144    return getattr(self, "predictor_spatial", self.predictor)
```

That is a **defaulted** `getattr`. Setting the attribute to `None` (`SCSam3Video.py:69`) makes `sc.spatial` return `None` — the default is never reached. `del`-ing the attribute, the option the docstring explicitly rejects, is the one that would make that fallback work. The runner survives today only because all four `self.spatial` uses (`runMVSeg.py:180, 184, 190, 210`) precede the retire at `:335-337`. Every inherited method that touches the predictor directly — `AddPoint` (`SCSam3Video.py:172`), `AddText` (`:186`), `InitializeSegmentation` (`:216`), `LoadVideo_Folder` (`:79`), `LoadVideo_Folder_MVSeg` (`:90`) — becomes `AttributeError: 'NoneType' object has no attribute 'handle_request'` after retirement. This is a genuine new failure surface introduced into OLD by the copy, and "memory-fix-plus-new-api / 0 refutations" does not reflect it.

**2. Exception safety (S6).** `RetireSpatialPredictor` (`SCSam3Video.py:37-72`) has no `try/finally`. If `predictor.handle_request(close_session)` at `:58-60` raises, `self.session_id_statial` is left stale and `self.predictor_spatial` non-`None`. More importantly, the copy inserts a **brand-new abort point at `runMVSeg.py:336`, between the expensive `PropagateAcrossViews` and `TrackForward`** — a failure there discards the entire cross-view pass. OLD had no such call.

**3. Multi-GPU.** `SCSam3Video.py:15` builds both predictors with `gpus_to_use = range(torch.cuda.device_count())`, and the inference code is genuinely multi-rank (`SCSam3VideoInferenceNewMem.py` imports `torch.distributed`, shards objects via `tracker_metadata["obj_ids_per_gpu"]` at `:1440-1446`, and gates on `self.rank == 0` at `:1173` and `:1277`). `torch.cuda.empty_cache()` at `SCSam3Video.py:71` releases the caching allocator **for the current device only**. The claimed "~3.2 GiB parameters + ~1.6 GiB autocast" reclaim is therefore a single-device claim; on an N-GPU run the spatial model's blocks on ranks ≥ 1 stay cached. Nobody measured this per-device.

**4. Torch-version pinning of the S4 argument.** The whole S4 equivalence rests on `torch.clamp(bool_tensor, max=0)` promoting to int64 and yielding zeros — a claim measured once by hand (`docs/analysis/SCHEDULE.md:188`, torch 2.10.0+cu128) with no regression test. `SCSam3/Dockerfile:2` pins a **floating tag** `pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel`, and the measured build (`+cu128`) is not even the tagged one (`cuda13.0`). Note the direction: the fast path *removes* the version-sensitive promotion, so NEW is the safer half — but the argument justifying it is unpinned.

**5. Error paths / silent-wrong-answer in S7 (see C2 below).** Not a crash class — a `.get(frame_idx, {})` class.

Determinism/RNG and memory-format/stride I checked and found nothing: no RNG anywhere on these paths; `keep = pixel_level_non_overlapping_masks > 0` is freshly allocated and contiguous, and `pred_masks` at the call site (`SCSam3VideoInferenceNewMem.py:513-516`, `out_binary_masks.unsqueeze(1)` → `(N,1,H,W)`) has identical shape, so `keep &= pred_masks` cannot silently broadcast-truncate where `torch.where` would have broadcast up.

---

### C. Empirical coverage — the existing evidence does not cover every changed path

**C1. S7 has zero output-equivalence evidence in either direction. This is the headline gap.**

The env var is set in exactly one place in the repo:

```
runMVOptThree.sh:14        -e SCSAM3_TRIM_CACHED_OUTPUTS=1 \
```

and `runMVSegAll.sh:58-60` — the script that produced the 12-dataset comparison — runs `docker run … python runMVSeg.py "${ds}" --algo "${algo}"` with **no `-e` at all** (and `ALGOS=(OneStage OneStageNew)` at `:25`). Therefore:

- The **12 byte-identical datasets ran with `TRIM_CACHED_OUTPUTS = False`** (`SCSam3VideoInferenceNewMem.py:38`), so `_trim_cached_frame_outputs` returned at its first line (`:549-550`) every time. Lines **`:551-559` never executed**.
- The **3 datasets that only NEW completes ran with trim ON** — and OLD OOMs on all three, so there is **no baseline to diff against**.

The 2×2 of {algo} × {trim} has an empty cell, and it is the only cell that could validate S7. The audit's `bp:false / new-feature / 1 refutation` verdict is correct in kind but understates it: this is not merely "a new feature," it is *the only changed code path with no empirical evidence whatsoever*, and it is simultaneously the change that produced the headline result (3 previously-impossible datasets).

**C2. Two static defects in S7 that the single recorded refutation may not name.**

Both concern the guard at `SCSam3VideoInferenceNewMem.py:550-552`, which returns early only if a **prior** `propagation_full` appears in `action_history`:

- *Silent empty output on `propagation_fetch`.* `propagate_in_video` reaches the trim call at `:1316` only for `propagation_partial` / `propagation_fetch` (`:1143-1152` returns early for full). The fetch branch at `:1171-1200` replays the **entire** `processing_order` out of the cache with `inference_states[spatial_idx]["cached_frame_outputs"].get(frame_idx, {})` at `:1174-1176`. After a trimmed partial walk, every past frame is gone and `.get(..., {})` returns an empty dict — so a subsequent fetch yields **zero masks for every trimmed frame, with no exception**. The guard looks backwards for a full propagation; it cannot see a *forthcoming* fetch.
- *The "strictly-past" guard does not mean "already consumed."* The comprehension at `:557-559` drops all `f < keep_frame_idx`. On a partial walk starting at frame 50, the very first trim call deletes cached frames 0-49 that an **earlier** walk populated and this walk never touched. The docstring at `:541-543` ("only strictly-past frames are dropped, never one the walk has not reached yet") conflates index order with consumption order, and the case it breaks is precisely the interactive case it claims to protect.

**C3. Changed code paths the 15 runs did not exercise.**

| Path | Why not reached |
|---|---|
| `SCSam3VideoInferenceNewMem.py:551-559` (trim body) | trim off in all 12 comparison runs (C1) |
| `:557` `reverse=True` half of the comprehension | `runMVSeg.py:234` always sends `propagation_direction="forward"` |
| `:1171-1200` `propagation_fetch` | the runner consumes each generator exactly once (`runMVSeg.py:245-247`, `:340-341`); a fetch never occurs |
| `SCSam3TrackerPredictorNewMem.py:1786` + `_map_keys` `:1819` | inside `remove_object`; MVSeg never removes an object — these are two of the readers S3's safety argument depends on |
| `:962` `.pop` / `:975` `in` on the meta-placeholder dict | reached only via `clear_all_points_in_frame`, called from `:1789` inside `remove_object` |
| `:144` / `:1042` `_reset_tracking_results` (`v.clear()`) | `runMVSeg.py:253` issues `reset_session` *before* any prompt, so the dicts are empty when cleared |
| `SCSam3Video.py:79/90/172/186/216` after retirement | runner never calls them post-retire — but the copy makes them reachable-and-broken |
| `self.rank != 0` branches with S4's bool return | unknown GPU count in the recorded runs; not established |

Note what *is* covered and should be credited: both tracker twins are exercised. `SCSam3TrackerPredictor.py` (non-NewMem) backs `predictor_spatial`, so its S3 site (`:407-417`) fires during `AddReferenceMask` and its S4 fast path (`:1409-1416`) fires via `SCSam3VideoInference.py:506` whenever `n_obj > 1`. S6 is covered end-to-end by the 12 byte-identical runs, since MVOpt *did* retire (`runMVSeg.py:335`) and OLD did not.

**C4. Empirical checks that would settle what static analysis left open.**

- **E1 (highest value, ~1 dataset, minutes).** Fill the empty cell: re-run one already-passing dataset as `ALGO=MVOpt runMVOptThree.sh <ds>` (which sets `SCSAM3_TRIM_CACHED_OUTPUTS=1`) and `diff -rq` against `SegMaskSam3OneStageNew`. This is the only way to get an OLD-vs-NEW byte comparison with trimming live, and it is cheap.
- **E2 — poison tests, which test the *claim* rather than re-running the code.** For S3, replace the placeholder at `SCSam3TrackerPredictorNewMem.py:416-417` with a **wrong-shaped** meta tensor (`torch.empty((0,), dtype=torch.bool, device="meta")`); for S2, restore the removed store at `SCSam3VideoInferenceNewMem.py:2002` but store a bare `object()`. If a full dataset still comes out byte-identical, "keys only" / "write-only" is *proven* rather than argued from a reader list that has already been shown incomplete (§A). If either run changes or raises, the claim is false. This is the check that beats another equivalence run, because both S2 and S3 currently rest on grep-completeness arguments.
- **E3.** A ~20-line unit test in the container pinning S4: random bool masks vs. the deleted `torch.where(pixel > 0, pred_masks, torch.clamp(pred_masks, max=0)) > 0`, including the all-zero-`obj_scores` tie case (reachable: `SCSam3VideoInferenceNewMem.py:461` supplies `0.0` for absent tracker scores, making `argmax` at `sam3/sam3/model/sam3_tracker_base.py:1126` a first-index tiebreak) and the `batch_size == 1` early return at `sam3_tracker_base.py:1120`. Pin it so the floating Dockerfile tag cannot silently invalidate the equivalence.
- **E4.** Log `torch.cuda.memory_reserved(d)` for every visible device around `runMVSeg.py:335-337` on a ≥2-GPU run, to test the per-device `empty_cache()` gap in §B3.

---

### Bottom line for the copy decision

The user's belief ("memory bug-fixes only, zero behavioral change") is **defensible for S1-S5**, which I re-verified independently and which are covered end-to-end by the 12-dataset byte-identity. It is **not established for S6 and S7**:

- **S6** adds a new public method, a new flag, a new abort point in the runner's critical section, and an object state (`predictor_spatial is None`) in which five inherited methods raise — justified by a docstring whose stated rationale is backwards for its only real caller (`runMVSeg.py:144`).
- **S7** is a behavior-changing feature, default-off, whose enabled path has *no* output-equivalence evidence and which I believe silently returns empty masks on `propagation_fetch` (`:1174`) and destroys pre-start-frame cache entries on any second partial walk.

If the goal is "OLD gains the memory fixes with zero behavioral change," S1-S5 can be copied as-is; **S6 should be copied only alongside a fix to `runMVSeg.py:144` (or a `del` instead of `= None`), and S7 should be left out or copied with the fetch/second-walk hazard closed.** And `.dockerignore` should be preserved or committed before any `--delete` copy.