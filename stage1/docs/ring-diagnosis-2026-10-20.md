## 1. Bottom line

1. **The collapse is not caused by bad π³ geometry. It is identity confusion in the learned star decoder once views leave the viewpoint range it was trained on.**
   - Far-half failures on T06 frame 900 are mostly whole-person identity swaps. 28 of 113 far pairs cover the wrong person, against 1 of 113 for the TCSVT chain. The main pattern is p2→p4 (9 pairs) and p4→p2/p6/p5.
   - π³ places every person correctly in 3D on the far side, so the geometry is right.
   - On the T06 views, collapse starts between a 47.8° viewpoint change (J 0.961) and 54.6° (J 0.745).
   - MUVOD never has a pair beyond about 58°. On T06, 58% of views are beyond 60° and 45% beyond 90°.
2. **The existing safeguards cannot catch it.**
   - The learned 3D-distance bias is weak, about a 1.1-nat swing, and the same across all 5 folds.
   - The `reproj` gate needs the predicted pixels to be visible from c_ini. Because of that it rejects 71–83% of *correct* far-view masks.
   - Round-2 neighbours of a far view are themselves far, so on the far half round 2 adds almost nothing.
3. **One fix works in a simulation on T06.** A 3D identity check that does not need visibility from c_ini flagged 27 of 28 wrong-person masks, with 2 false flags out of 152. Falling back to the chain for flagged objects raises mean J from 0.742 to 0.893. Adding object-level co-visibility routing gives 0.914–0.928. The ceiling (best of the two methods per pair) is 0.944, and the chain alone is 0.941.
   - **This is descriptive only.** T06 frame 900 has now been seen many times, so from here on it can only be a development set for out-of-domain data, not a test set.
4. **The honest consequence for the paper.** On 360° rigs the learned model currently adds nothing over the chain, even in the best case. A routing fix makes it *safe*. Only wide-baseline training (Fix C) could make it *better* there. The ICCV claims must be scoped to match.
5. **A new risk on MUVOD.** A routing rule based on object co-visibility would fire on 41% of MUVOD DEV-A (view, object) pairs, mostly on the fisheye rigs. It is unsafe on its own, so it must be measured on DEV-A and confirmed for non-inferiority on DEV-B.

## 2. Diagnosis — evidence

All analyses below are read-only. The scripts are in the scratchpad folder `ring_diag/`. They read the repo's π³ cache (`stage1/Data/T06_flag_ganggang_F/Cache/pi3`), the ground-truth masks (`Data/PlenopticFull/T06_flag_ganggang_F/Mask`), the scores (`plenoptic/full/jf_{lat,sub}.json`) and the frame-900 masks on the NAS. My J computation reproduces the official means exactly: 0.7423 for the latest config and 0.9407 for the submitted one.

**2.1 What fails.**
- Frame 900, mean J over the 184 non-reference pairs: 0.742 (latest) vs 0.941 (submitted). By ring distance, d 1–6 is the same for both (0.956); d 7–16 is 0.608 vs 0.932.
- Wrong-person masks on the far half: 28 vs 1. Empty masks: 4 vs 1.
- The swaps follow no simple mirror or image-x prior: the nearest unmirrored-x person explains 5 of 28 and the mirrored one 4 of 28, about chance level.
- In the reference view, p4 stands directly behind p2 (3D separation 0.336 normalised units). From the far side the depth order is reversed.
- The 5 folds give 0.737–0.773 (`docs/plenoptic-F.md` §4 and the s1map4k probe summary). That probe's raw outputs were deleted in the 10-20 cleanup. Its claim of 24–29 identity swaps per fold at 1080p survives only as a summary in `croom/plan_result.json` and cannot be re-checked.

**2.2 π³ is not the cause (checked).**
- Lifting each camera's ground-truth person pixels with that camera's own π³ depth into the c_ini frame puts the 3D centroid 0.01–0.24 normalised units from the same person's reference centroid. Persons are 0.256–0.678 apart.
- The nearest reference person is correct in 180 of 182 evaluable pairs. The two exceptions are 2.6k and 4.0k px at 4K.
- π³ does leave a 94° gap between cam_31 and cam_00 (other steps 4–18°). Whether that gap is physical is **unverified**, but it does not break correspondence at the object level.
- π³ confidence is a weaker failure signal than viewpoint change. At view level the AUC is 0.88 for confidence vs 0.95 for viewpoint angle. The stored validity mask (confidence > 0.1) covers at least 0.957 of every view, so it carries no information.

**2.3 Outside the training support (checked; angles estimated from π³).**
- Angle between c_ini and the target camera, measured at the reference view's median 3D point. MUVOD maximum per scene: CBABasketball 58.1°, MartialArts 53.6°, Blocks 47.0°; most scenes are under 25°.
- π³ angles on the fisheye scenes are unreliable.
- On T06: views up to 47.8° have J of at least 0.885; every view from 54.6° up has J of at most 0.82. View-level AUC for angle is 0.95 (31 views).
- Training pairs are only c_ini → the 2 annotated cameras, plus a randomly chosen predicted memory.

**2.4 Why nothing catches it (checked).**
- **The bias is weak.** Learned `geo_bias` read from the fold 0–4 checkpoints: about +0.53 to +0.60 at distance ≤ 0.1, about −0.5 at distance ≥ 0.6. That is too little to override appearance attention when the target shows the person's back.
- **The `reproj` gate is blind beyond about 90°** (THETA 0.5, OCC_TOL 0.01, `stage1/methods/geomem/run.py`). Feeding ground-truth masks in as predictions:
  - correct masks are rejected at 0.14 (d 1–3), 0.34 (d 4–6), 0.71 (d 7–9), 0.83 (d 10–12) and 0.76 (d 13–16);
  - swapped masks pass at most 1% of the time.

  So round 2 on far views runs mostly on c_ini memory alone.
- **Round 2 cannot rescue far views.** It takes its memory from the k=3 nearest cameras by π³ centre, and on the far half those cameras are also far.

**2.5 Failure predictors that need no ground truth (descriptive; contaminated by T06).**
- Object co-visibility: the share of the object's c_ini-visible surface that is unoccluded in the target, at 5% tolerance. AUC 0.80 at pair level. Pairs with co-visibility ≥ 0.5 have J 0.922 (2 failures in 86).
- The same statistic on MUVOD DEV-A (inputs only) is below 0.5 for 41% of pairs: AlexaMeade 0.68–0.75, Welder 0.73, Dog 0.70, MATF 0.49. Using it alone would send large parts of MUVOD to the chain.

**2.6 Cheap diagnostics still owed** (out-of-domain dev set, declared, none used to pick constants; under 0.5 GPU-h):
- D1: save round-1 masks (`--r1-out`) to see whether swaps come from round 1 or round 2.
- D2: untrained star — did MUVOD fine-tuning make far views worse?
- D3: fine-tuned attention with the bias zeroed.
- D4: `--xcl none`, to see whether the exclusivity rule chains the errors.
- D5: the all-17-scene checkpoint vs the fold-2 rule.

## 3. Proposed fixes

**Fix A — 3D identity check with per-object fallback (no training).**
- How it works: lift each predicted mask with the target view's π³ depth; take its median 3D point; flag the mask if the nearest reference-object centroid (from the c_ini mask) is not its own id. Constants on MUVOD DEV-A: minimum valid points, a distance-ratio margin, and abstaining when π³ is unreliable.
- What a flagged object gets: (a) its chain mask (simple, but the chain is TCSVT Stage 1, so present this as an ablation), or (b) a relay decode from Fix B (preferred).
- It replaces the `reproj` test as the trust signal for round-2 memories.
- Evidence (T06 frame 900, descriptive): 27 of 28 swaps flagged, 2 of 152 false flags, J 0.742 → 0.893.
- Risk: false flags on the fisheye MUVOD scenes, where π³ is poor. This is unmeasured and is the first thing to measure on DEV-A.

**Fix B — Learned relay decoding over a π³ view graph (no retraining).**
- Views within θ_s of c_ini are decoded exactly as now (star). Views beyond θ_s are decoded in order of increasing angle. Each one's memory is c_ini plus up to 3 already-decoded nearest views whose masks passed Fix A's check.
- `MemorySet` already accepts several predicted memories, and the checkpoint was trained with a predicted extra memory (`extra_mem_pred`).
- If θ_s is at or above the MUVOD support, MUVOD output is byte-identical. But collapse begins at 54.6°, so θ_s probably has to be below 58°. Pick it on DEV-A plus the T06 dev set and verify non-inferiority on DEV-B.
- Risk: errors accumulating along the relay, as in the chain; Fix A's check is the guard.
- Cost: decode time roughly 2× round 1 (estimate).

**Fix C — Wide-baseline training (the only fix that could beat the chain on rings).**
- Add training pairs with 60–180° viewpoint change and relay-style memories.
- Possible sources (licences and mask availability **unverified**):
  - CMU Panoptic Studio, a multi-person dome, with SAM 3 pseudo-masks prompted from projected 3D keypoints (`http://domedb.perception.cs.cmu.edu/`);
  - synthetic multi-person rings rendered from scans, or Kubric (arXiv 2203.03570);
  - the Ego-Exo4D correspondence training split.
- Optional: make the bias angle-aware, so geometry can outweigh appearance on back views.
- Keep the MUVOD 5-fold training. Train an all-17-scene checkpoint for out-of-domain tests and fix that rule before any out-of-domain test run.
- Not recommended:
  - **restricting the bias to near views** — the geometry is right and the bias is the only correct cue; keep it only as diagnostic D3;
  - **a π³-confidence gate as the main signal** — it is a weaker proxy for viewpoint change.

## 4. Validation without selecting on test data

- **Splits.**
  - MUVOD DEV-A (design), DEV-B (confirm once per registration), keyframes = TEST (once), as in `docs/stage1-dev-protocol.md`.
  - **Out-of-domain dev: T06 frame 900** (contaminated; declare it as dev).
  - **Out-of-domain test 1:** new ground truth on CROOM takes T03/T04/T05/T07 (on the NAS, `PlenopticServer1/CROOM/JPG/`), one frame each, all 32 cameras. Draw it blind to any predictions and commit sha256 hashes before any decode. Annotation cost is an **unverified estimate** (~15–25 human hours).
  - **Out-of-domain test 2 (public):** ScanNet++ results binned by viewpoint angle (0–45/45–90/90–135/135–180°, as in SegMASt3R), or the Ego-Exo4D correspondence val split.
- **Order for each registration:** constants from DEV-A plus the T06 dev set → commit the registration → DEV-B once. Non-inferiority margin δ = 0.003 J&F, set from the measured retrain noise (+0.0017; E14 §1.2): CI lower bound > −0.003, no scene below −0.01.
- **Then MUVOD TEST once.**
- **Then the out-of-domain tests once.** Pass if ΔJ > 0 with the camera-cluster CI lower bound > 0, and fewer wrong-person masks. Report every registered arm.
- **Limit DEV-B reuse to 2 registrations** (Fix A+B together, then Fix C). Treat the ablation as pre-declared measurement, not selection.
- **Weakness "architecture chosen on test":** re-decode the existing checkpoints (star, geomem_ft, geobias_only_ft, geoset_pred_ft round 1/2) on DEV-B (~1 GPU-h each) and report that table instead of the keyframe-based choice.
- **The fold rule for external scenes stays fixed** (fold 2) or is replaced up front by the all-scene checkpoint. Never pick the fold from out-of-domain scores.

## 5. No-geometry fine-tune ablation

During training the only use of geometry is GeoBias. SSL is used only in geomem_ssl, and the extra-memory camera is random. At decode, π³ is used in three places: the bias, the choice of round-2 neighbours (camera centres), and the `reproj` gate.

| Arm | Training | Decode |
|---|---|---|
| **NG** `nogeo_pred_ft` (new; code needed) | Same recipe and seeds as geoset_pred_ft but `use_bias=False`, `use_geometry=False` | Neighbours by rig index (NG-idx) or by SAM 3 global-feature cosine (NG-feat), with `--gate obj`, `--xcl outs` |
| NG+π³ | NG checkpoint | π³ neighbours + `reproj` gate (π³ only for routing and gating) |
| G | Current geoset_pred_ft | As now (C0/xs) |
| G-bias0 | geoset_pred_ft | Bias weights zeroed at decode (how much it relies on the bias) |
| Star | Existing untrained arm (0.888) | — |

- This is a 2×2 design: geometry in training on/off × π³ at decode on/off.
- **Seeds:** 3 per trained arm, because retrain noise (~0.0017) is close to the expected bias effect (geobias_only − geomem_ft J +0.008 in `stage1/docs/results-geomem2.md`).
- **Report:** J, F, J&F, failures (J<0.5), wrong-person count; on DEV-A, DEV-B, TEST and the out-of-domain sets.
- **Code changes needed:** add the arm to `stage1/geomem/arms.py` (SET2/PRED_MEM/NO_EMBED, not BIAS) and add a `--nbr {pi3,index,feat}` option in `stage1/methods/geomem/run.py`.

## 6. GPU-time estimates

**Past runs (RTX 6000 Ada 48 GB):**

| Job | Time |
|---|---|
| π³, 32 views at 4K | 21 s, 13.4 GB |
| Stage-1 T06 decode | 13.5–16.8 s + 6–9 s load, 10.9 GB |
| Keyframe decode, 17 scenes | 465 s, 20.7 GB |
| Round-1 / two-round decode | 146 s / 361 s |
| DEV-A decode + score | ~24 min |
| DEV-B decode / score | 40 min / 20 min per preset |
| geoset_pred_ft training | 32–37 min per fold (fold 2: 2,226 s), so 5 folds ≈ 2.7–3.1 h |
| Feature cache | ~3,500 views in 10 min |
| Stage-2 sweep | 1 h 43 min |

**Plan:**

| Work | GPU time |
|---|---|
| Diagnostics D1–D5 | < 0.5 h |
| Fix A+B | 5–7 h |
| Architecture re-check on DEV-B | 4–5 h |
| No-geometry ablation | 20–25 h with 3 seeds (8–10 h with 1) |
| Fix C | 10–29 h (data 2–6 h + 5–19 h training + 3–4 h evaluation) |
| Baselines | ≤ 1–2 h each (engineering 0.5–2 days each, code availability **unverified**) |
| **Total** | **≈ 50–80 GPU-h** |

The ICCV 2027 deadline is about early March 2027, roughly 19 weeks away (**unverified**).

## 7. TCSVT elements to keep out (target < 20% overlap)

**Keep out** — cite at most, never re-describe or reuse:
- the abstract's headline numbers;
- §III "Recovering the reference camera" and the MUVOD metric write-up;
- the two-stage baseline with Stage 1 as a pseudo-video chain — at most one baseline row, citing TCSVT;
- all of Stage 2: cross-view memory (mode C, window, gate, pointer, row shift), raw frame-0 seeding, zoom-gated crop retracking;
- the XMem comparison (79.35/75.63 vs 91.0/86.4) and the per-scene, margin-decomposition, selection-optimism (LOSO), runtime, oracle and seed-replacement tables;
- §§VII–VIII: GT-seed reruns, the first-frame deficit, the mask-grid round trip;
- the five seed-repair attempts (G-det, G-corr, G-geo-a SfM, **G-geo-b π³ reprojection**, G-gs, SeedVote) and the 41-vs-623 asymmetry;
- the F0/N0 terminology;
- §IX on the T06 ring: frame-900 GT seeding, text/click seeding, the ring-distance J figure and the occlusion events;
- the TCSVT figures (pipeline, seed failures, ring, event figures).

**Can be new in ICCV:**
- the learned stage 1 (GeoSet-2: c_ini-memory star decoding, π³ RBF bias, memory-attention fine-tune, predicted second memory, gates, E11/E14 decode rules);
- Fixes A–C;
- the DEV protocol;
- the new benchmarks and recent baselines;
- the keyframe metrics.

**Process:**
- Write from scratch.
- Run an n-gram overlap check against `paper/main.tex` at the `tcsvt-submitted` tag.
- Cite TCSVT anonymously and include it in the supplementary. This follows CVPR practice; the ICCV 2027 policy text is **unverified**.

**Conflict needing a decision:** `docs/venues-2027.md` plans to add the learned stage 1 to the TCSVT revision if reviewers ask. Doing that would move the ICCV paper's core into TCSVT. Decide before 12-15.

## 8. Unverified

- Whether the 94° π³ gap is physical.
- The 1080p identity-swap counts from the deleted probe.
- Licences and mask availability for CMU Panoptic, Ego-Exo4D, ScanNet++ and Kubric.
- Code availability for MV-SAM, 3AM, G²TAM, SegMASt3R/MuViSeg, O-MaMa, V²-SAM.
- The ICCV 2027 dates and dual-submission wording.
- The annotation hours.
- All Fix A–C effect sizes outside T06 (the T06 numbers are descriptive and the set is contaminated).

ITEMS
- Far-half collapse = identity swaps: T06 frame 900 far half (d7-16, 113 pairs): latest config mean J 0.608 with 28 wrong-person masks (p2->p4 x9, p4->p2 x7, p4->p6 x4, ...) and 4 empty; submitted chain 0.932 with 1 wrong-person mask. Not explained by an image-x or mirrored-x prior (5/28, 4/28). [/tmp/claude-1000/-home-sjpark-Documents-SCSegmentation/8f2081e6-71f6-4db0-ab8a-894e91d89bb3/scratchpad/ring_diag/swap_diag.py] verified=True
- Pi3 geometry is correct at object level: GT person pixels lifted with each camera's own Pi3 depth land 0.01-0.24 normalised units from the same person's reference centroid (persons 0.256-0.678 apart). Nearest person correct in 180/182 evaluable pairs; the 2 exceptions are 2.6k and 4.0k px at 4K. The 94 deg ring gap (cam_31-cam_00) exists, but whether it is physical is unverified. [/tmp/claude-1000/-home-sjpark-Documents-SCSegmentation/8f2081e6-71f6-4db0-ab8a-894e91d89bb3/scratchpad/ring_diag/ring_geo.py] verified=True
- Out of training viewpoint support: Pi3-estimated viewpoint change at the content: MUVOD maximum 58.1 deg (CBABasketball), no pair above 60. T06: 58% of views above 60 deg, 45% above 90, maximum 155. Collapse starts between 47.8 deg (J 0.961) and 54.6 deg (J 0.745); view-level AUC 0.95. Angles on the fisheye scenes are unreliable. [/tmp/claude-1000/-home-sjpark-Documents-SCSegmentation/8f2081e6-71f6-4db0-ab8a-894e91d89bb3/scratchpad/ring_diag/baselines.py] verified=True
- Learned 3D bias is weak: geo_bias read from geoset_pred_ft fold0-4 checkpoints without torch: about +0.53 to +0.60 at distance <= 0.1, about -0.5 at distance >= 0.6 (a ~1.1-nat swing); same across folds. [/tmp/claude-1000/-home-sjpark-Documents-SCSegmentation/8f2081e6-71f6-4db0-ab8a-894e91d89bb3/scratchpad/ring_diag/read_bias.py] verified=True
- reproj gate is blind on far views: With GT masks used as predictions, the gate rejects correct masks at 0.14 / 0.34 / 0.71 / 0.83 / 0.76 for d1-3 / 4-6 / 7-9 / 10-12 / 13-16; swapped masks pass at most 1%. Cause: the occlusion test (OCC_TOL 0.01) requires visibility from c_ini. [/tmp/claude-1000/-home-sjpark-Documents-SCSegmentation/8f2081e6-71f6-4db0-ab8a-894e91d89bb3/scratchpad/ring_diag/gate_diag.py] verified=True
- 3D identity check simulation (Fix A): Visibility-independent check (median 3D point of the predicted mask vs reference object centroids) flags 27/28 wrong-person masks with 2/152 false flags. Fallback to the chain mask for flagged objects: J 0.742 -> 0.893; plus object co-visibility < 0.25/0.5: 0.914/0.928; best-of-both per pair 0.944; chain alone 0.941. Descriptive only (T06 is contaminated). [/tmp/claude-1000/-home-sjpark-Documents-SCSegmentation/8f2081e6-71f6-4db0-ab8a-894e91d89bb3/scratchpad/ring_diag/hybrid.py] verified=True
- Co-visibility router unsafe on MUVOD alone: Object co-visibility, computed from inputs only on MUVOD DEV-A (38,461 view-object pairs), is below 0.5 for 41% (AlexaMeade 0.68-0.75, Welder 0.73, Dog 0.70, MATF 0.49). On T06 it has AUC 0.80. It must be combined with the identity check and measured on DEV-A. [/tmp/claude-1000/-home-sjpark-Documents-SCSegmentation/8f2081e6-71f6-4db0-ab8a-894e91d89bb3/scratchpad/ring_diag/objcovis_muvod.py] verified=True
- Pi3 confidence as a gate: Stored validity (confidence > 0.1) covers at least 0.957 of every T06 view, so it is uninformative. Median sigmoid confidence has view-level AUC 0.88 (Spearman 0.60), weaker than viewpoint angle (0.95). Not recommended as the primary signal. [/home/sjpark/Documents/SCSegmentation/stage1/Data/T06_flag_ganggang_F/Cache/pi3/MANIFEST.json] verified=True
- Fix B: learned relay decoding: Beyond threshold theta_s, decode views in order of increasing angle using memory = c_ini + up to 3 already-decoded nearest views that pass the identity check; replaces the reproj gate there. Existing MemorySet and extra_mem_pred training support multiple memories. Effect size untested. [/home/sjpark/Documents/SCSegmentation/stage1/methods/geomem/run.py] verified=False
- Fix C: wide-baseline training: MUVOD training pairs are c_ini -> 2 annotated cameras (at most 58 deg). Add 60-180 deg pairs: CMU Panoptic with SAM 3 pseudo-masks, synthetic rings, or the Ego-Exo4D correspondence split; optionally an angle-aware bias. Data licences unverified. 5-15 GPU-h per seed for training. [/home/sjpark/Documents/SCSegmentation/stage1/geomem/train.py] verified=False
- No-geometry fine-tune ablation: New arm nogeo_pred_ft = geoset_pred_ft recipe with use_bias=False and use_geometry=False (training geometry is only GeoBias; SSL is geomem_ssl-only; the memory camera is random). Decode with index or feature neighbours and --gate obj. 2x2 design with Pi3 at decode, plus G-bias0 and star; 3 seeds (retrain noise 0.0017). About 20-25 GPU-h. [/home/sjpark/Documents/SCSegmentation/stage1/geomem/arms.py] verified=False
- DEV protocol extended for out-of-domain data: MUVOD DEV-A/DEV-B/TEST as in the protocol doc. T06 frame 900 becomes the out-of-domain dev set (contaminated). New blind ground truth on CROOM T03/T04/T05/T07 is out-of-domain test 1; ScanNet++ angle bins or Ego-Exo4D val is test 2. Non-inferiority margin 0.003 J&F on DEV-B; at most 2 DEV-B registrations; fold rule fixed in advance. [/home/sjpark/Documents/SCSegmentation/docs/stage1-dev-protocol.md] verified=False
- Re-validate the architecture on DEV-B: The architecture was chosen on the 664 keyframe pairs before the DEV protocol existed. Re-decode existing checkpoints (star, geomem_ft, geobias_only_ft, geoset_pred_ft round 1/2) on DEV-B, about 1 GPU-h each, with no training. [/home/sjpark/Documents/SCSegmentation/stage1/docs/results-baseline-fix.md] verified=False
- GPU timing sources: Training 32-37 min per fold (results-geoset2 l.170; fold2 config 2226 s); keyframe decode 465 s, DEV-B decode 40 min and score 20 min (E14); Pi3 32 views 4K 21 s and T06 decode 13.5-16.8 s (croom plan_result.json); feature cache 3,500 views in 10 min (results-geoquery l.164); stage-2 sweep 1 h 43 min. [/home/sjpark/Documents/SCSegmentation/docs/stage1-E14.md] verified=True
- TCSVT keep-out list: Keep out: the chain Stage-1 description, all of Stage 2 (cross-view memory, frame-0 seeding, crop retracking), the XMem, decomposition, LOSO, oracle and runtime tables, the five seed-repair generators including the Pi3 reprojection one, SeedVote and F0/N0, §IX on the T06 ring (frame-900 GT/text/click seeding, ring-distance figure, occlusion events), and all TCSVT figures. [/home/sjpark/Documents/SCSegmentation/paper/main.tex] verified=True
- Overlap conflict: TCSVT revision: venues-2027.md plans to put the learned stage 1 into the TCSVT revision if reviewers ask; that would move the ICCV core into TCSVT. Needs a user decision before the 12-15 review. [/home/sjpark/Documents/SCSegmentation/docs/venues-2027.md] verified=True
- Probe evidence deleted: The s1map4k/s1map probe outputs (5-fold 4K J 0.737-0.773, 24-29 swaps per fold at 1080p) were removed in the 10-20 cleanup; only the summary in croom/plan_result.json remains. [/tmp/claude-1000/-home-sjpark-Documents-SCSegmentation/8f2081e6-71f6-4db0-ab8a-894e91d89bb3/scratchpad/croom/plan_result.json] verified=True
- Recent baselines to run: MV-SAM https://arxiv.org/abs/2601.17866, 3AM https://arxiv.org/abs/2601.08831, G2TAM https://arxiv.org/abs/2607.03789, SegMASt3R https://arxiv.org/abs/2510.05051, O-MaMa https://arxiv.org/abs/2506.06026, V2-SAM https://arxiv.org/abs/2511.20886 (taken from the repo survey; code availability unverified). [/home/sjpark/Documents/SCSegmentation/docs/mvimage-seg-survey.md] verified=False