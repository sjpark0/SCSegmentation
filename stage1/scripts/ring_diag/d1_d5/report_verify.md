I checked the other agent's work independently and the core numbers hold. The preset reproduces 0.742 exactly, and the default code path is unchanged. Several of its readings need correcting, mainly what the "swap" counts actually contain and how much of the gap they explain. I did not use the GPU and changed nothing in the repo; everything I wrote is under `ring_d15/verify/`.

## What I recomputed

My script re-scored all 21 arms from the PNGs and the full-resolution ground truth, using the same 184 pairs (the two pairs with no ground-truth person are cam_12/p5 and cam_15/p3; near d1–6 = 71, far d7–16 = 113).

| Arm | J (full res) | J (1/4 res) | J near | J far | wrong near / far | empty far | J<0.5 |
|---|---|---|---|---|---|---|---|
| nas_chain | 0.9407 | 0.9411 | 0.956 | 0.932 | 0 / 1 | 1 | 2 |
| nas_lat | 0.7423 | 0.7424 | 0.956 | 0.608 | 0 / 28 | 4 | 38 |
| preset | 0.7423 | 0.7424 | 0.956 | 0.608 | 0 / 28 | 4 | 38 |
| preset_r1 | 0.7155 | 0.7154 | 0.935 | 0.577 | 0 / 31 | 3 | 42 |
| star | 0.5888 | 0.5890 | 0.886 | 0.402 | 2 / 51 | 2 | 65 |
| untr | 0.6335 | 0.6337 | 0.925 | 0.450 | 1 / 47 | 2 | 59 (60 at 1/4 res) |
| bias0 | 0.7007 | 0.7009 | 0.942 | 0.550 | 1 / 33 | 4 | 45 |
| xnone | 0.7407 | 0.7408 | 0.954 | 0.607 | 0 / 28 | 4 | 39 |
| xout | 0.7371 | 0.7371 | 0.955 | 0.601 | 0 / 28 | 4 | 40 |

- **Other arms:** the rounds 1 (untr_r1, bias0_r1) and the fold proxies f0/f1/f3/f4 (0.7437 / 0.7593 / 0.7731 / 0.7373, far wrong 22 / 22 / 22 / 25) also match the report. So do the J-by-distance table and the better/worse-than-chain counts.
- **Swap counts are robust:** counting wrong-person masks at full resolution gives the same numbers. Counting by best IoU instead gives preset 27 and bias0 34.
- **Reproduction is exact:** my full-resolution J for the preset (0.7423189127315917) and the chain (0.9406727846145791) equal the official `Q1_mean_J` values to every digit.
- **Byte identity:**
  - preset matches the NAS published masks and preset_post (186/186).
  - preset_r1 matches preset_post_r1, xnone_r1 and xout_r1 (187/187).
  - star and untr_r1 differ in 183 of 189 PNGs, yet their J agrees to within 0.001.
- **D1 confirmed:**
  - Far-half changes from round 1 to final: ok→ok 78, wrong→wrong 27, wrong→ok 3, wrong→empty 1, ok→wrong 1, empty→empty 3.
  - 27 of the 28 final swaps were already wrong in round 1, 26 of them on the same person.
  - Exclusivity changes no far pair's status (81 / 28 / 4).
  - The six changed pairs are the ones the report names.
- **Gate figures confirmed** (recomputed from `round1_conf.json` and the π³ camera centres):
  - Correct round-1 masks pass 55/71 near and 23/79 far; wrong ones pass 3/31.
  - Pairs with no gated neighbour memory for that person: 15% near (11/71), 61% far (69/113).
  - cam_18's neighbours are cam_17, cam_19 and cam_16, which supports the one chaining case.
- **Code change:**
  - The default path is guarded by `getattr(args,"zero_bias",False)`, and the flag adds no config key when off.
  - No preset sets it, and the two in-process callers (`stage1.py`, `dev_decode.py`) build their arguments without it.
  - Timestamps show `preset` was decoded before the edit and `preset_post` after. Its RUN.json config, `round1_conf.json` and `xcl_conf.json` are identical to the pre-edit run's.
  - Host tests: 178 passed, 3 skipped (I ran them). The container tests run on CPU only, read-only: 7 passed.
  - The `MVSeed/runs/e4c_project.json` change dates from 2026-09-12 and is unrelated.

## Discrepancies and over-claims

1. **"Wrong-person mask" is not the same as an identity swap.** Of the preset's 28 far wrong masks:
   - 11 are fragments under 25% of the person's ground-truth area (10 under 10%);
   - 8 are oversized (more than 1.5× the area) and spill onto another person;
   - only about 9 are similar-size swaps.

   For the untrained star decoder, 32 of its 51 are oversized. So "fine-tuning cuts swaps 51 → 31" is largely fewer oversized masks.

   The three wrong masks that passed the gate are small pieces: about 3k, 6k and 156k px against 0.8–1.4M px of ground truth. That includes the one "chaining" case, whose cam_19/p4 memory is a 3k-px fragment.
2. **Swaps are not the whole gap.** The preset is 0.198 J below the chain:
   - 0.153 of it (77%) comes from wrong or empty pairs;
   - 0.046 comes from pairs where the right person was found but the mask is worse (far correct masks J 0.842 against 0.945 for the chain on the same pairs).
3. **D2 "collapses about 2 ring steps earlier" compares unlike things** (one round against two). Like for like it is about 1 step:
   - round 1 vs round 1: d5 0.78 vs 0.96, d6 0.76 vs 0.86;
   - same two-round decode: d6 0.78 vs 0.96, d7 0.56 vs 0.78.

   Smaller slips:
   - **Swap patterns:** p2→p4 on the far half is 10 → 9, not "11–12 → 9"; p1→p3 10 → 2 is correct.
   - **D3 "plus 8 new swaps":** on the far half it is 7 new; the 8th is a near-half swap.
   - **star vs untr_r1 "as they should":** they agree in J, not in bytes. With the bias starting at zero, I would expect the residue to come from the attention-mask code path (not checked).

   Something the report missed: every one of the preset's 28 wrong pairs is also wrong in the untrained star decode, and round 1's 31 are all among star's 51. So fine-tuning only removed errors here; it added none.
4. **D3 is over-read.**
   - Zeroing the bias at decode on a model trained with it is an off-distribution ablation. It also zeroes the learned no-geometry offset (b_none = −0.10).
   - The near-half round-1 drop (0.935 → 0.907, 2 new near swaps) fits general damage, not the loss of a far-view cue.
   - The chain "0.634 → 0.701 → 0.742" cannot be split additively into "attention" and "bias" shares.
   - Defensible claim: the trained decoder depends on its bias. "The bias helps" needs a model trained without it (the NG arm).
5. **D4 answers chaining by construction, not by experiment.** Round-1 masks are byte-identical across `none`, `out` and `outs`, so exclusivity never touches the memories. The split of −0.0037 / +0.0053 is noise-sized on 184 correlated pairs and should not be read.
6. **D1 leaves a confound open.** Whether letting the rejected correct far masks through the gate would help was not tested. A far camera's three nearest neighbours are also far cameras, which carry many of the same errors.
7. **The D5 proxy is confounded and slightly enlarges what is known about the dev set.**
   - Fold 2 also leaves out Dog and Frog, and the folds were trained separately.
   - Fold 4 includes CBABasketball yet scores 0.737, below fold 2's 0.742. So there is no evidence that training on the 58° scene helps.
   - Knowing that fold 3 scores best (0.773) on T06 must not influence the fold rule for external scenes. The range matches the deleted probe, so the new information is small.
8. **Statistics.** One scene and one frame give 184 correlated pairs and no confidence intervals. Only differences much larger than about 0.01 J, or clear patterns at the pair level, should be read at all.

## Corrected short conclusion

On the contaminated T06 frame-900 dev set, which only describes and decides nothing:

- **Where the deficit comes from:** about 77% of the gap (0.742 vs 0.941) is far-half pairs where the mask covers the wrong person, spills onto a neighbour, or is a fragment or empty. The rest is poorer masks where the person was found.
- **When it happens:** these errors are set in round 1, which decodes from the reference camera's memory alone. 27 of the 28 final wrong masks were already wrong there.
- **Why round 2 barely changes it:** the reproj gate lets through only 29% of correct far masks, so 61% of far pairs get no neighbour memory for that person. Errors spread through round-2 memory in only one case, a fragment.
- **Fine-tuning:** with the same decode, fine-tuning beats untrained weights (far J 0.608 vs 0.450, wrong masks 28 vs 47), mostly by producing fewer oversized masks. Its remaining failures are a subset of the untrained ones.
- **The 3D bias:** the trained model depends on it (zeroing it at decode drops J to 0.701). Its standalone value is unmeasured.
- **Exclusivity:** it plays no part, since it cannot spread errors across views.
- **D5:** not run, because no all-17-scene checkpoint exists. The other folds give 0.737–0.773 with the same far-half collapse, and nothing here changes the fold-2 rule or any constant.

Files are in `/tmp/claude-1000/-home-sjpark-Documents-SCSegmentation/8f2081e6-71f6-4db0-ab8a-894e91d89bb3/scratchpad/ring_d15/verify/`:
- verify.py
- gate.py
- verify_summary.json
- verify_rows.json
- verify_out.txt