# Brief: SCSam3 multi-view segmentation (demoSCSam3OneStageNew)

Everything below was established empirically in this session. Treat it as ground
truth about behaviour; verify anything about *mechanism* against the code.

## Paths

- Code under review: `/home/sjpark/Documents/SCSegmentation/SCSam3/demoSCSam3OneStageNew/`
- Upstream SAM 3 it was forked from: `/home/sjpark/Documents/SCSegmentation/SCSam3/sam3/sam3/` (model/ subdir)
- Unified diffs isolating the author's changes (READ THESE FIRST, they are the contribution):
  `/tmp/claude-1000/-home-sjpark-Documents-SCSegmentation/9786e830-b342-4be7-bc11-e7701415bd8d/scratchpad/analysis/diff_*.patch`
  - `diff_SCSam3TrackerPredictorNewMem.patch` (613 changed lines) – the multi-view memory tracker. THE core contribution.
  - `diff_SCSam3VideoInferenceNewMem.patch` (499) – video inference wrapper for the NewMem tracker
  - `diff_SCSam3VideoInference.patch` (334) – the cross-view ("spatial") inference wrapper
  - `diff_SCSam3TrackerPredictor.patch` (76) – cross-view tracker (small changes)
  - `diff_SCSam3VideoPredictorNewMem.patch` (45), `diff_SCSam3VideoPredictor.patch` (23) – session/request layer
  - `diff_build_scsam3.patch` (208) – model construction
  - `diff_NewMem_over_Tracker.patch`, `diff_NewMem_over_Inference.patch` – what NewMem adds on top of the non-NewMem variant
- Orchestration: `SCSam3Video.py` (233 lines) and `sam3_demoVideo.py` (the demo driver)
- SAM 2 predecessor of the same idea (for lineage): `/home/sjpark/Documents/SCSegmentation/SCSam2/demo/SCSam2VideoPredictorNew.py`, `SCSam2VideoNew.py`
- MVSeg runner written this session (works, documents the real API): `/home/sjpark/Documents/SCSegmentation/SCSam3/runMVSeg.py`

## What the algorithm does (verified)

Input: N synchronized cameras (multi-view video), one text/point/mask prompt on ONE reference view at the start frame.

1. **Cross-view ("spatial") pass.** The start frame of every camera is stacked into a
   pseudo-video of N frames (view index = frame index). A plain SAM 3 video tracker
   (`predictor_spatial`) is prompted on the reference view and propagates in both
   directions across this pseudo-video, yielding a mask per object per view.
2. **Temporal pass with cross-view memory ("NewMem").** Each camera gets its own
   temporal session, seeded at the start frame with the mask from step 1. Tracking
   advances all cameras in lockstep (frame t for view 0..N-1, then t+1 ...). The
   NewMem tracker's `_prepare_memory_conditioned_features_multiple` conditions frame
   t of view v not only on view v's own memory bank (`num_maskmem=7`) but also on the
   `maskmem_features` of neighbouring views `spatial_idx + s_pos for s_pos in range(-4, 0)`
   at the same/previous frame – those are pulled from CPU to GPU and concatenated
   into the memory attention prompt. This is the "OneStageNew" variant. "OneStage"
   is the same without cross-view memory (each view tracked independently).

The design and the `range(-4,0)` window, `num_maskmem=7`, CPU offload of
`maskmem_features` are inherited unchanged from the author's SAM 2 version
(`SCSam2VideoPredictorNew.py`). SAM 3 checkpoint is 3.3 GB vs SAM 2 hiera-large 857 MB.

## Empirical results (MVSeg benchmark, DAVIS J&F, 3 scored cameras × 21 frames per dataset)

12 datasets where every method completed (AlexaMeadeExhibit, CoffeeMartini, FlameSteak excluded):

| method | J | F | J&F |
|---|---|---|---|
| SAM2 SegMaskNew1 (author's SAM 2 version) | 0.8171 | 0.8727 | 0.8449 |
| SAM3 OneStage (no cross-view memory) | 0.8145 | 0.8786 | 0.8465 |
| SAM3 OneStageNew (cross-view memory) | 0.8175 | 0.8820 | **0.8497** |

- The gain over SAM 2 is almost entirely **F (boundary)**; J is flat.
- OneStageNew vs OneStage differ materially only on Fencing (0.8862→0.9112) and PoznanStreet (0.8662→0.8787); elsewhere within ±0.002.
- Per-dataset vs SAM 2: wins Barn +0.030, Breakfast +0.028, Dog +0.017, MATF +0.019, PoznanStreet +0.014, FacePaint +0.008; losses **Welder −0.034 (J 0.856→0.801)**, **Fencing −0.021 (J 0.881→0.856)**; Blocks/Frog/Painter ≈ tie.
- Reference view + 2 other cameras are scored; the reference camera's own J is ~0.87–0.95, non-reference cameras 0.58–0.80 (Blocks: cam9 ref 0.876, cam4 0.772, cam0 0.575).

## Failures observed

- **OOM with OneStageNew** on the 3 heaviest datasets: AlexaMeadeExhibit (45 views × 33 objects @ 2560×1920), CoffeeMartini (18 × 66 @ 2704×2028), FlameSteak (21 × 68 @ 2704×2028). Cross-view pass completes; OOM occurs when seeding/tracking all N temporal sessions (46 GB allocated, <1 GB reserved-unused → genuine capacity, not fragmentation). The author's **SAM 2 version completes CoffeeMartini at 40 GB** with the identical algorithm structure (same window, same num_maskmem, same offload) → the gap is model size, not a wiring bug. OneStage (independent sessions, only the 3 scored cameras tracked) completes all 15.
- Two of those OOMs took the whole host down (hard reboot) until the container was capped with `--memory=90g`; then they fail as clean CUDA OOM.
- Timing: OneStage 35–190 s/dataset; OneStageNew 84–553 s/dataset (Welder, 46 views, 553 s).

## Bugs and dead code found in demoSCSam3OneStageNew (verified)

1. **Request key mismatch**: `SCSam3Video.InitializeSegmentation` and `RunNaiveTracking` send `start_frame_idx=...` but `SCSam3VideoPredictor*.handle_stream_request` reads `request.get("start_frame_index")` → the value is silently ignored (None). Consequences: (a) the demo cannot start tracking at an arbitrary frame; (b) `propagation_direction` defaults to `"both"`, so temporal tracking also runs backward over the entire video (e.g. 180 frames of Frog before the annotated range) — wasted compute, and the yield order (forward first, then backward) happens to make the demo's `next()` loop work only because start_frame=0 there.
2. **Passing an explicit `start_frame_index` to the cross-view propagation breaks it when the reference view is the last index** (Blocks: ref cam9 = view 9 of 10 → other views J≈0.003; Painter ref v15 = 15 of 16 → same). With start left implicit (None) both recover (Blocks 0.28→0.75). Root cause not established — suspected: the forward pass over a single frame + hotstart buffering / `tracking_bounds` state poisons the subsequent backward pass in `propagate_in_video` (direction "both" runs forward then backward on the same state).
3. **Dead API**: `SCSam3Video.LoadVideo_Folder_MVSeg` calls `predictor.init_state(video_path=...)` and appends to `self.inference_state` (never initialised); `AddMaskSingle` calls `self.predictor.add_new_mask(inference_state=self.inference_state_spatial ...)`. Both are SAM 2 API leftovers and raise. `AddMask`, `RunTracking` likewise.
4. **Objects absent in the reference view at the seed frame never get a masklet** (Blocks: 22 objects prompted, 15 registered). Structural to single-reference seeding; SAM 2 version has the same limit.
5. `SCSam3TrackerPredictorNewMem copy.py` – a 1930-line duplicate with tiny differences, left in the package.
6. `RunNaiveTracking` in OneStage (non-New) variant skips objects missing from a view (no zero-mask fallback) while OneStageNew adds a zero mask → the two variants seed differently.
7. Reference camera choice = the annotated camera whose GT holds the highest object id (max id, not object count).

## Not yet known (worth establishing)

- Whether the 4-neighbour spatial memory window is the right size/shape for a 360° camera ring (views wrap around; the window is one-sided: only *previous* views `s_pos<0`).
- Whether cross-view memory helps or hurts on the datasets where SAM 3 loses to SAM 2 (Welder, Fencing).
- Memory scaling law: per-object, per-view, per-pixel cost of the NewMem tracker.
- Whether hotstart / masklet-confirmation thresholds (designed for temporal video) are appropriate for the cross-view pseudo-video.
