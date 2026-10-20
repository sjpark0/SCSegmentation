# Ring diagnostics D1–D5: learned stage 1 on T06 frame 900

I ran D1–D4 and reproduced the published 0.742 exactly. D5 was not run. `metrics.json` is written, but `report.md` is not: the harness refused it ("Subagents should return findings as text, not write report files"), so the full report is below.

**Setup.** The model is GeoSet-2 preset `geoset_rep_xs` (`geoset_pred_ft`, fold 2, gate `reproj`, `--xcl outs`). The data is the 32-camera CROOM ring T06 at source frame 900, reference camera cam_06, 184 non-reference (camera, person) pairs. **T06 frame 900 is a declared, contaminated development set.** Everything here is descriptive; nothing here picks a constant or the fold.

## Bottom line

- **Reproduction is exact.** My own decode of the preset scores J 0.7423 at full resolution and 0.7424 with swap_diag's 1/4-resolution rule. It has 28 far wrong-person masks and 4 far empty masks. All 186 PNGs are byte-identical to the published frame-900 masks on the NAS.
- **D1: the swaps come from round 1.**
  - Round 1 already has 31 far swaps. 27 of the 28 final swaps were already wrong in round 1, 26 of them on the same wrong person.
  - Round 2 fixes 3 swaps, turns 1 into an empty mask, creates 1 new swap and moves 1 swap to a different wrong person.
  - That last case is errors chaining through the round-2 memory: the neighbour's own swapped mask passed the `reproj` gate.
  - Net effect of round 2 on the far half: J 0.577 → 0.607, swaps 31 → 28.
- **D2: MUVOD fine-tuning made far views much better, not worse.**
  - The untrained star decoder gets far J 0.402 with 51 swaps. The fine-tuned round 1 gets 0.577 with 31.
  - With the full preset decode, untrained weights get 0.450 with 47 swaps, against 0.608 with 28.
  - The untrained model collapses about 2 ring steps earlier (J 0.76–0.78 at d5–d6, against 0.96–0.97).
  - So the identity confusion comes from SAM 3's appearance matching. Fine-tuning reduces it but does not remove it.
- **D3: zeroing the 3D bias at decode makes things worse.**
  - J falls 0.742 → 0.701, far J 0.608 → 0.550, and far swaps rise 28 → 33. The biggest losses are at d6–d7.
  - Step by step along the preset decode: untrained 0.634 → fine-tuned attention without bias 0.701 → with bias 0.742.
  - The bias is weak, but the decoder relies on it and it helps.
- **D4: exclusivity does not chain the errors.**
  - With `--xcl none`, J is 0.7408 against 0.7424. The 28 far swaps are the same, as are the 4 empty masks, and no far pair changes status between the raw round-2 output and the final output.
  - The preset's `outs` mode combines two things. Exclusivity at L > 0 costs 0.0037 J; the −0.5 threshold shift adds 0.0053.
- **D5: not run.**
  - No all-17-scene checkpoint exists. Training one would take about 45 min on an idle GPU, at or over the 45 GPU-min limit, and the GPU is shared.
  - As a clearly labelled extra, I decoded the other four fold checkpoints. Each was trained on scenes that include CBABasketball, the MUVOD scene with the widest viewpoint change (58°).
  - They score J 0.737–0.773, against 0.742 for fold 2, with 22–25 far swaps against 28. This re-measures the deleted probe's 0.737–0.773 range exactly.
  - Every fold collapses on the far half. These numbers do not reopen the fixed fold-2 rule.

## Arms × metrics

All arms are scored with the swap_diag rules:
- 184 pairs: 71 near (d 1–6) and 113 far (d 7–16).
- J is computed at 1/4 resolution with nearest resizing; Jfull at 3840×2160.
- "wrong" means the predicted mask overlaps another person's GT more than any other person's. "empty" means it covers no person.
- r1 means round 1: a single decode using only the reference camera's (c_ini) memory.

| Arm | Diagnostic | Weights / decode | J | Jfull | J d1–6 | J d7–16 | wrong d1–6 | wrong d7–16 | empty d7–16 | J<0.5 |
|---|---|---|---|---|---|---|---|---|---|---|
| nas_chain | reference | TCSVT chain, NAS frame 900 | 0.9411 | 0.9407 | 0.956 | 0.932 | 0 | 1 | 1 | 2 |
| nas_lat | reference | published latest config, NAS frame 900 | 0.7424 | 0.7423 | 0.956 | 0.608 | 0 | 28 | 4 | 38 |
| **preset** | reproduction, D1 round 2 | fine-tuned fold 2, 2 rounds, reproj, outs | **0.7424** | **0.7423** | 0.956 | 0.608 | 0 | 28 | 4 | 38 |
| preset_r1 | D1 round 1 | same run, round 1 | 0.7154 | 0.7155 | 0.935 | 0.577 | 0 | 31 | 3 | 42 |
| star | D2a | untrained `--arm geomem`, 1 round | 0.5890 | 0.5888 | 0.886 | 0.402 | 2 | 51 | 2 | 65 |
| untr_r1 | D2b round 1 | untrained geoset_pred_ft, round 1 | 0.5882 | 0.5880 | 0.886 | 0.401 | 2 | 51 | 2 | 65 |
| untr | D2b | untrained geoset_pred_ft, preset decode | 0.6337 | 0.6335 | 0.925 | 0.450 | 1 | 47 | 2 | 60 |
| bias0_r1 | D3 round 1 | fine-tuned attention, bias = 0, round 1 | 0.6643 | 0.6644 | 0.907 | 0.512 | 2 | 36 | 4 | 50 |
| bias0 | D3 | fine-tuned attention, bias = 0, preset decode | 0.7009 | 0.7007 | 0.942 | 0.550 | 1 | 33 | 4 | 45 |
| xnone | D4 | preset with `--xcl none` (L > 0) | 0.7408 | 0.7407 | 0.954 | 0.607 | 0 | 28 | 4 | 39 |
| xout | D4 split | preset with `--xcl out` (L > 0) | 0.7371 | 0.7371 | 0.955 | 0.601 | 0 | 28 | 4 | 40 |
| preset_f0 | D5 proxy (extra) | fold 0, preset decode | 0.7438 | 0.7437 | 0.957 | 0.610 | 0 | 22 | 7 | 39 |
| preset_f1 | D5 proxy (extra) | fold 1 | 0.7593 | 0.7593 | 0.957 | 0.635 | 0 | 22 | 5 | 35 |
| preset_f3 | D5 proxy (extra) | fold 3 | 0.7732 | 0.7731 | 0.957 | 0.658 | 0 | 22 | 2 | 33 |
| preset_f4 | D5 proxy (extra) | fold 4 | 0.7373 | 0.7373 | 0.956 | 0.600 | 0 | 25 | 5 | 41 |

- **Near-half empty masks** are 0 for every arm.
- **Round 1 of the fold proxies (f0 / f1 / f3 / f4):** J 0.7275 / 0.7259 / 0.7499 / 0.6973, far swaps 24 / 26 / 26 / 29.
- **Pairs better / worse than the chain by more than 0.01:**

  | preset | preset_r1 | star | untr | bias0 | xnone |
  |---|---|---|---|---|---|
  | 9 / 85 | 5 / 113 | 0 / 131 | 4 / 103 | 8 / 93 | 7 / 87 |

Mean J by ring distance d = 1…16:

| Arm | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chain | .92 | .96 | .95 | .97 | .97 | .97 | .96 | .88 | .97 | .95 | .93 | .96 | .92 | .90 | .89 | .97 |
| preset | .92 | .96 | .95 | .97 | .97 | .96 | .78 | .60 | .69 | .58 | .57 | .55 | .51 | .56 | .61 | .62 |
| preset r1 | .92 | .95 | .95 | .96 | .96 | .86 | .77 | .57 | .67 | .60 | .51 | .48 | .47 | .56 | .58 | .59 |
| star | .92 | .95 | .94 | .96 | .78 | .76 | .55 | .38 | .44 | .32 | .40 | .38 | .31 | .38 | .43 | .47 |
| untr | .92 | .95 | .95 | .97 | .97 | .78 | .56 | .47 | .51 | .49 | .49 | .38 | .32 | .42 | .39 | .49 |
| bias0 | .92 | .96 | .95 | .97 | .97 | .87 | .68 | .57 | .57 | .46 | .51 | .54 | .49 | .53 | .57 | .60 |
| xnone | .92 | .95 | .95 | .97 | .97 | .96 | .77 | .59 | .68 | .61 | .57 | .55 | .51 | .56 | .61 | .63 |

## What each diagnostic shows

**Reproduction.** I checked this before reading any other arm.
- The `preset` decode produces exactly the 186 PNGs of `SegMaskSam3XW1CGPS4MSdgeoset_rep_xsFsCs10kG2Rs/cam_KK/900/` on the NAS, all byte-identical (180 non-reference masks and 6 reference masks; the 6 empty seeds have no PNG).
- So the full run's frame-900 output is exactly the stage-1 output. `--ref-snap` did not change cam_06 at frame 900.
- The checkpoint is fold 2, sha `5b9ef4885d9e…`.

**D1: round 1 against round 2.** Far-half status changes between steps:

| Step | ok→ok | wrong→wrong | wrong→ok | wrong→empty | ok→wrong | empty→empty |
|---|---|---|---|---|---|---|
| round 1 → raw round 2 (`xnone`) | 78 | 27 | 3 | 1 | 1 | 3 |
| raw round 2 → final (`outs`) | 81 | 28 | 0 | 0 | 0 | 4 |

- **Fixed by round 2:** cam_24/p6 (J 0.61), cam_26/p1 (0.86) and cam_27/p1 (0.50) became correct; cam_30/p6 became empty. Each had a correct neighbour memory for the same person that passed the gate.
- **New swap:** cam_24/p1 went from correct to p6, even though its gated-in neighbour memory (cam_25/p1) was correct.
- **Changed wrong person:** cam_18/p4 went from p1 to p6. Its neighbour cam_19's p4 mask was itself swapped to p6 and passed the gate. This is the one case seen of errors chaining through the round-2 memory.
- **Gate on round-1 masks:**
  - Swapped masks passing: 3 of 31 (cam_19/p4, cam_25/p4, cam_26/p4, all at d 12–13).
  - Correct masks passing: 55 of 71 near (77%) but only 23 of 79 far (29%).
  - As a result, 61% of far pairs get no neighbour memory for that person and decode from c_ini alone, against 15% of near pairs.

**D2: untrained star decoder.**
- `star` and `untr_r1` agree to within 0.001, as they should.
- Against the fine-tuned round 1, the untrained model is worse on both halves: near 0.886 vs 0.935, far 0.402 vs 0.577, far swaps 51 vs 31. It already collapses at d5 (0.78 vs 0.96).
- The main swap patterns are shared. Fine-tuning removes most p1→p3 swaps (10 → 2) but few p2→p4 swaps (11–12 → 9).

**D3: fine-tuned attention with the 3D bias zeroed.** This uses the new `--zero-bias` flag; the largest bias parameter before zeroing was 0.349 in absolute value.
- Round 1: 0.664 vs 0.715. Final: 0.701 vs 0.742.
- Far: 0.550 vs 0.608, with 33 swaps vs 28.
- In round 1, near J also drops (0.907 vs 0.935) and 2 near swaps appear.
- Of the preset's 28 swaps, 26 also occur without the bias, plus 8 new ones.

**D4: exclusivity off.**
- `xnone` is 0.0016 J below `preset`.
- It has the same 28 swaps (none new), the same 4 empty masks, and no far status changes. The preset run's own counters agree: exclusivity changed 159 masks and emptied none.
- Splitting the preset's `outs` mode: exclusivity at L > 0 costs 0.0037 J (`out` − `none`); the threshold shift adds 0.0053 (`outs` − `out`).

**D5: all-17-scene checkpoint, not run.**
- **Why not:**
  - `runs/geoset_pred_ft/` holds only folds 0–4.
  - Fold 2 trained in 2,226 s for 2,772 steps on 14 scenes. 17 scenes means about 3,366 steps, about 2,700 s ≈ 45 min on an idle GPU.
  - The GPU is shared: decodes took 28–30 s against the documented 13.5–16.8 s, so training would take roughly 90 min of wall time.
- **Extra proxy, descriptive only:** the other four folds (each trained on CBABasketball) score 0.737–0.773, with far J 0.600–0.658 and 22–25 far swaps. Every fold collapses on the far half. The ±0.02 spread mixes the effect of training scenes with training noise.

## Caveats

- **One scene, one frame.** The 184 pairs come from 31 cameras × 6 people and are correlated. No CI is computed, and the set is contaminated. Use these numbers to describe mechanisms, not to tune.
- **Swap rule.** Wrong-person masks are counted with swap_diag's 1/4-resolution argmax. Quarter- and full-resolution J agree to within 0.0004 for every arm.
- **D3 measures reliance, not training without the bias.** The attention was co-trained with the bias, so zeroing it at decode is not the same as training without it (that is the NG arm of §5).
- **D2b still uses geometry at decode.** It still uses π³ to pick round-2 neighbours and in the `reproj` gate. It is "untrained attention + the preset decode", not a geometry-free model.
- **D4 can only test effects within a view.** `outs` exclusivity acts only on round-2 outputs, never on memories, so it cannot spread errors across views by construction.
- **Shared GPU.** The GPU was shared with `cart_cart_f120` (about 15 GB, about 92% utilisation), which I did not touch; it was still running at the end.
  - Results are deterministic: a repeat decode was byte-identical.
  - Peak torch allocation was 10.9 GB per two-round decode and 4.2 GB for `star`.
  - The whole card peaked at 31.3 GB, including the other job's ~15.7 GB, so my share was about 15.6 GB, under the 20 GB limit.
- **Cost.** 11 decodes in total (including the check after the code change): about 5 min of decoding and about 9 min of wall time.

## Code change (D3 only, not committed)

`stage1/methods/geomem/run.py`:
- a new `--zero-bias` flag (`dest="zero_bias"`, default off);
- `zero_geo_bias(wrapper)`, shown below;
- a call in `decode_scene`, guarded so that the config gets new keys only when the flag is on;
- `check_gate_out(..., zero_bias=False)`, which refuses `--out` equal to the arm name when the flag is on;
- `main()` passes `args.zero_bias` through.

No in-process `Stage1` preset sets the flag.

```python
def zero_geo_bias(wrapper):
    import torch
    assert wrapper.use_bias and wrapper.geo_bias is not None, "--zero-bias needs an arm with the 3D bias (use_bias)"
    with torch.no_grad():
        before = max(float(p.detach().abs().max()) for p in wrapper.geo_bias.parameters())
        for p in wrapper.geo_bias.parameters():
            p.zero_()
    return before
...
    if getattr(args, "zero_bias", False):                # ring diagnosis D3; off = the unchanged path (no config key)
        config.update({"zero_bias": True, "zero_bias_max_abs_before": zero_geo_bias(wrapper)})
```

**Default behaviour is byte-identical.** I decoded `preset` before the change and `preset_post` after it. All 186 + 187 PNGs are byte-identical, and the RUN.json config, `round1_conf.json` and `xcl_conf.json` are identical.

**Tests:**
- New files: `stage1/tests/test_zero_bias.py` (host) and `stage1/tests_torch/test_zero_bias_torch.py` (container).
- Host suite: `cd stage1 && .venv/bin/python -m pytest -q -p no:cacheprovider` → 178 passed, 3 skipped (175 before plus the 3 new tests).
- Container: `scripts/docker_run.sh scsam3 python -m pytest -q -p no:cacheprovider tests_torch/test_zero_bias_torch.py tests_torch/test_bias.py` → 7 passed.

## Exact commands

- Inputs were copied from `stage1/Data/T06_flag_ganggang_F` to `ring_d15/data/T06_flag_ganggang_F` and checked with sha256.
- Each fold proxy has a directory `ring_d15/data_f{k}/T06_flag_ganggang_F` with symlinks to that copy and a `meta.json` whose `external_fold` is k.
- The NAS frame-900 masks (375 PNGs) were copied once to `ring_d15/nas900/`.

`ring_d15/run_arm.sh <datadir> <out> [args]` runs:

```bash
docker run --rm --gpus all --shm-size=32g --user "$(id -u):$(id -g)" -v /:/host \
  -w /host/home/sjpark/Documents/SCSegmentation/stage1 \
  -e HF_HOME=/host/home/sjpark/Documents/SCSegmentation/SCSam3/hf_cache -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/host/home/sjpark/Documents/SCSegmentation/stage1 -e HOME=/tmp \
  scsam3 python methods/geomem/run.py --scene T06_flag_ganggang_F --data /host$D15/<datadir> --out <out> [args]
```

The arms, run one at a time:

```bash
run_arm.sh data preset      --arm geoset_pred_ft --gate reproj --xcl outs --r1-out preset_r1        # before code change
run_arm.sh data preset_post --arm geoset_pred_ft --gate reproj --xcl outs --r1-out preset_post_r1   # after: byte-identity
run_arm.sh data star        --arm geomem --untrained                                                # D2a
run_arm.sh data untr        --arm geoset_pred_ft --untrained --gate reproj --xcl outs --r1-out untr_r1   # D2b
run_arm.sh data bias0       --arm geoset_pred_ft --zero-bias --gate reproj --xcl outs --r1-out bias0_r1 # D3
run_arm.sh data xnone       --arm geoset_pred_ft --gate reproj --xcl none --r1-out xnone_r1         # D4
run_arm.sh data xout        --arm geoset_pred_ft --gate reproj --xcl out  --r1-out xout_r1          # D4 split
for k in 0 1 3 4; do run_arm.sh data_f$k preset_f$k --arm geoset_pred_ft --gate reproj --xcl outs --r1-out preset_f${k}_r1; done
stage1/.venv/bin/python ring_d15/score.py ring_d15/scores_raw.json <arm>=<mask dir> ...
stage1/.venv/bin/python ring_d15/analyze.py
```

## Files

All in `/tmp/claude-1000/-home-sjpark-Documents-SCSegmentation/8f2081e6-71f6-4db0-ab8a-894e91d89bb3/scratchpad/ring_d15/`:
- `metrics.json`: per-arm metrics, definitions and run info, D1 transitions, swap overlaps, D5 status
- `pairs.json`: per-pair rows for every arm
- `scores_raw.json`
- `data/T06_flag_ganggang_F/Result/<arm>/`: masks, `RUN.json`, `round1_conf.json`, `xcl_conf.json`
- `data_f{0,1,3,4}/T06_flag_ganggang_F/Result/preset_f{k}{,_r1}/`
- `nas900/`, `logs/`
- scripts: `run_arm.sh`, `chain1.sh`, `score.py`, `analyze.py`

Repo changes, uncommitted:
- `/home/sjpark/Documents/SCSegmentation/stage1/methods/geomem/run.py`
- `/home/sjpark/Documents/SCSegmentation/stage1/tests/test_zero_bias.py`
- `/home/sjpark/Documents/SCSegmentation/stage1/tests_torch/test_zero_bias_torch.py`