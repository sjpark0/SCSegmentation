# S1-E14 threshold-upsample -- pre-registration (written 2026-10-15 BEFORE any variant was scored)

Scope: host-only post-decode variants of the adopted geoset_rep_x round-2 decode on DEV-A only
(dump: geomem/runs/_e14_logitdump/geoset_rep_x_A; baseline scoring: geomem/runs/_e13_dev/dev_pairs.json, 3 300 pairs).
TEST keyframe and DEV-B are not touched. No tracked file is modified.

Only GT-free information was looked at before this file was written: the logit gradient at the zero crossing
(adopted full-res logits, 12 random views, up to 8 objects each): median |grad L| = 1.23 logit / px (IQR 0.65-1.90),
so a shift of 1 logit moves a median boundary by about 0.8 px. That fixed the tau grid range below.

## Pipeline (every variant)
per eval view: low_res [B,288,288] (all B objects of the view, absent objects = -1024) -> upsampler U -> full-res
logits L [B,h,w] -> pos = L > -tau -> adopted exclusivity (argmax winner, nested-mask cap 0.5) = run.exclusive_masks
applied to (L + tau) -> per scored pair J, F (mvseg2d.metrics.jf_pair) vs the DEV-A GT label map.
tau > 0 grows every mask (lower threshold), tau < 0 shrinks.

## Upsamplers U
- bil2  : ADOPTED. 288 -> 1008 bilinear -> (h,w) bilinear, bit-exact host emulation (e14dump.logits_full_np).
          Must reproduce dev_pairs.json J&F exactly at tau = 0 (pipeline check).
- bil1  : CONTROL only (not eligible): direct 288 -> (h,w) bilinear (cv2 INTER_LINEAR, half-pixel centres).
- bic1  : direct 288 -> (h,w) bicubic (cv2 INTER_CUBIC, A = -0.75, half-pixel centres, replicate border). No constant.
- jbu(ss, sr): joint bilateral upsampling of the logits (Kopf et al. 2007) guided by the target image.
          Output pixel p at low-res coords u = (y+.5)*288/h - .5, v = (x+.5)*288/w - .5; taps = the 4 x 4 cells
          floor(u)-1..floor(u)+2 x floor(v)-1..floor(v)+2 inside the grid; weight = exp(-d_s^2/(2 ss^2)) *
          exp(-||Lab(p) - Lab288(q)||^2/(2 sr^2)); Lab = cv2 float Lab of the image (L 0..100);
          Lab288 = INTER_AREA resize of the Lab image to 288 x 288 (the cell average, the grid spans the whole image).
          L(p) = sum w x_q / sum w.   ss in {0.5, 1.0} (cells), sr in {4, 8, 16, 32, inf} (inf = spatial-only control,
          still eligible as an upsampler). 10 settings.
- gf(r, eps): colour guided filter (He et al. 2013) of the ADOPTED full-res logits (bil2) with the RGB image in [0,1]
          as guide; box radius r px, regulariser eps.   r in {4, 8}, eps in {1e-4, 1e-3, 1e-2}. 6 settings.
          Computed on the crop where the object's bil2 logits > -16, grown by 2r + 2 px (outside: bil2 value).

## tau grid
tau in {-1, -0.5, -0.25, 0, +0.25, +0.5, +1, +1.5, +2, +3, +4}   (11 values)

## Fixed selection criterion
Primary metric = DEV-A all-pair J&F (all 3 300 pairs) aggregated as mvseg2d.dev_compare does (pair -> view "<t>/<cam>"
-> scene means, mean over the 17 scenes), as a paired delta vs the adopted bil2 / tau = 0.
- (a) tau*  = argmax over the tau grid with U = bil2.
- (b) U*    = argmax over {bic1, the 10 jbu settings, the 6 gf settings} at tau = 0  (bil1 is a control, not eligible);
          also the best within each family (bic1, jbu, gf) is reported.
- (c) (U, tau)* = argmax over {bic1, jbu x 10, gf x 6} x tau grid (187 combinations).
- Ties within 1e-4 J&F: prefer the smaller |tau|, then fewer pairs down by > 0.01.
Reported for every grid point: hierarchical delta J&F, pair-mean delta J&F, good-seed / failing-seed delta
(good = adopted pair J&F >= 0.8, as in dev_pairs.json), pairs up / down by > 0.01.
Reported for each pick: scene-cluster paired bootstrap (mvseg2d.paired, 10 000 resamples, seed 0, views "<t>/<cam>")
95 % CI of delta J&F, J, F on all pairs, on good seeds and on failing seeds; risk = pairs with delta J&F < -0.01
(count, by good / failing, worst drops); delta 1-recall and 1-precision on good seeds.
Selection-bias check: leave-one-scene-out selection (pick on 16 scenes by the same rule, score the held-out scene),
averaged over the 17 scenes -- an estimate of the pick's gain that does not reuse the scene it is scored on.
The picks' CIs on DEV-A are optimistic (chosen on the same data); only a DEV-B run (not done here) confirms.
