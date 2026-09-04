## SWEEP: external consumers of `demoSCSam3OneStageNew`

Scope of the proposed action matters, so I split every verdict into **(a) copy the 11 `.py` files only** vs **(b) copy the directory** (`rsync --delete` / `rm -rf OLD && cp -r NEW OLD`). `demoSCSam3MVOpt` is **not** a full copy of the old folder — it holds only the 11 `.py` files + `__pycache__` (24 files, vs 1624 files / 13 GB in `demoSCSam3OneStageNew`).

### 1. `/home/sjpark/Documents/SCSegmentation/SCSam3/runMVSeg.py` — the only importer

**`ALGOS` / `DEFAULT_OUT` / `NEEDS_ALL_VIEWS` (lines 38-46): no code change, but they stop meaning what they say.**
`ALGOS["OneStageNew"] = "demoSCSam3OneStageNew"` and `ALGOS["MVOpt"] = "demoSCSam3MVOpt"` would point at byte-identical code, while `DEFAULT_OUT` still routes them to two different output folders (`SegMaskSam3OneStageNew` vs `SegMaskSam3MVOpt`). `NEEDS_ALL_VIEWS = ("OneStageNew", "MVOpt")` already covers both, so `track_mode`/`track_idx` (lines 304-306) are unchanged. The runner keeps working; what breaks is provenance — the folder name `SegMaskSam3OneStageNew` currently identifies OLD-code output, and after the copy any re-run under that flag produces NEW-code output into the same name.

**`uses_spatial_predictor` (lines 251-252, inside `TrackForward`): branch outcome unchanged, but only because both edits land together.**
- Today, `OneStageNew`: the attribute does not exist, so `getattr` falls back to `getattr(self, "predictor_spatial", None) is not None`. `demoSCSam3OneStageNew/SCSam3Video.py:16` builds `predictor_spatial` and nothing ever retires it → `True` → multi-session request (`request["session_ids"] = self.session_ids; request["spatial_idx"] = m`).
- After the copy: `uses_spatial_predictor = True` (`demoSCSam3MVOpt/SCSam3Video.py:23`) → `True` → **the identical request dict**. No algorithmic change here.
- The coupling is the thing to notice: with `RetireSpatialPredictor` present and the flag *absent*, the fallback would evaluate `False` (it sets `self.predictor_spatial = None`) and `OneStageNew` would silently flip to the single-session `request["session_id"]` branch — a real algorithm change. **Never cherry-pick `RetireSpatialPredictor` into a package without also adding `uses_spatial_predictor`.**
- `demoSCSam3OneStage` is untouched by this: its `__init__` never creates `predictor_spatial` at all, so the fallback stays `False` and it keeps the `session_id` branch.

**`RetireSpatialPredictor` (lines 335-338): this is the one place where the copy really changes what `runMVSeg.py` does.**
- Today, `--algo OneStageNew`: `hasattr(sc, "RetireSpatialPredictor")` is **False** (grep confirms the method exists only in `demoSCSam3MVOpt` and `demoSCSam3OneStage`), so the cross-view model stays resident for the whole temporal pass and **`torch.clear_autocast_cache()` is never called**.
- After the copy: the branch fires — `close_session` on `session_id_statial`, `predictor.model = None`, `gc.collect()`, `torch.cuda.empty_cache()`, then `torch.clear_autocast_cache()`, plus the extra `spatial model  retired` stdout line that the log-scrapers in `runMVOptThree.sh:21` / `runMVSegAll.sh:68` will now see.
- Output-neutral in principle (`masks_spatial` holds detached bool tensors; nothing after line 321 reads `sc.predictor_spatial`) and empirically neutral — see §7.
- `--algo OneStage` is unaffected: `demoSCSam3OneStage/SCSam3Video.py:31` **already** defines `RetireSpatialPredictor`, so that branch already fires today (early-returns, then `clear_autocast_cache()` runs).

Everything else in the runner — `SPATIAL_START_IMPLICIT` (line 37), `pick_reference`, `LoadCameraFolders`, `AddReferenceMask`, `PropagateAcrossViews`, the write loop (lines 342-356) — is unchanged. The `--track-cams` help text (lines 59-63) stays accurate.

### 2. `/home/sjpark/Documents/SCSegmentation/SCSam3/runMVSegForSam2.py`
**No effect, either variant.** `ALGO_DIR` is hardcoded to `demoSCSam3ForSam2New` (line 26) and `DEFAULT_OUT = "SegMaskSam3ForSam2New"`. It never touches `demoSCSam3OneStageNew`.

### 3. Shell scripts
| script | effect of the copy |
|---|---|
| `SCSam3/runMVSegAll.sh` | **Silent relabel.** `ALGOS=(OneStage OneStageNew)` (line 25) and `OUTNAME[OneStageNew]=SegMaskSam3OneStageNew` (line 26) — MVOpt is not registered here at all. After the copy this sweep runs NEW code but writes and scores it as method `SegMaskSam3OneStageNew` into `jf_raw_sam3.json` (line 94) and reports it against the SAM 2 baselines (lines 100-110). It does **not** set `SCSAM3_TRIM_CACHED_OUTPUTS`, so the trim path stays off. |
| `SCSam3/waitAndRunMVSeg.sh` | **Highest-blast-radius caller.** Line 55 runs `--algo OneStageNew --overwrite Frog`, line 63 runs the full sweep `--overwrite`. One invocation overwrites every `Data/MVSeg/*/SegMaskSam3OneStageNew/` with new-code output. It does that today too — but today the overwriting code is the same code that produced them. |
| `SCSam3/runMVOptThree.sh` | Keeps working (`--algo ${ALGO:-MVOpt}`). Note it exports `-e SCSAM3_TRIM_CACHED_OUTPUTS=1` (line 14), so `ALGO=OneStageNew ./runMVOptThree.sh …` would, after the copy, activate the `_trim_cached_frame_outputs` path under the OneStageNew label. |
| `SCSam3/runForSam2Three.sh` | No effect — drives `runMVSegForSam2.py`. |
| `SCSam3/launch_container.sh` | No effect — generic `docker run … -v /:/host`. |
| `SCSam3/demo/*.sh` (`MVSeg.sh`, `IoU*.sh`, `MOTA*.sh`, `MVSeg1.sh`, `MVSeg_ObjSelect.sh`) | No effect — SAM 2 pipeline, no reference to any `demoSCSam3*` package. |

### 4. `SCSam3/Dockerfile` and `.dockerignore`
- **Dockerfile: no effect, and no rebuild needed.** It only does `COPY ./sam3 /opt/sam3` (line 27). The demo packages are never baked into the image; every runner reaches them through the `-v /:/host` bind mount. So the copy takes effect on the very next `docker run` — and conversely, no pinned image preserves the old behavior for you.
- **`.dockerignore`: inert either way.** The only one in the tree is `/home/sjpark/Documents/SCSegmentation/SCSam3/demoSCSam3OneStageNew/.dockerignore` (contents: `*` / `!*.py`). Docker honors a `.dockerignore` only at the build-context root, and the documented context is `SCSam3/` (README line 16). `demoSCSam3MVOpt` has none, so (b) deletes this file; nothing observable changes today.

### 5. `figures/` — the real asset risk
- `figures/make_figure.py:14` — `SRC = ".../SCSam3/demoSCSam3OneStageNew"`, reads `{SRC}/{m}/0.png` for m in 0..31 (lines 132, 160).
- `figures/make_baseline_figure.py:15` — `OURS = ".../SCSam3/demoSCSam3OneStageNew"`, reads `{OURS}/{v}/0.png` (line 119).
- Those PNGs live in the 32 numbered directories `0`…`31` **inside** `demoSCSam3OneStageNew` (13 GB, mtime Mar 11), written by the in-package `sam3_demoVideo.py:59`. `demoSCSam3MVOpt` contains **no** numbered directories.
- **(a) `.py`-only copy: no effect.** The PNGs are untouched; both figure scripts still produce byte-identical output.
- **(b) directory copy with delete: both scripts die** with `FileNotFoundError` on `.../0/0.png`, and `fig_teaser_{ko,en,nocap}.png` / `fig_baseline_{ko,en,nocap}.png` become unreproducible without re-running `sam3_demoVideo.py` over `Data/VideoSample_1`. **These PNGs are not recoverable from git** — `.gitignore:183` is `*.png` and `git ls-files` shows only the 13 `.py` files tracked under that folder.
- `figures/run_independent_baseline.py` imports upstream `sam3` only (line 26) — no effect.

### 6. `Data/MVSeg/`
- `eval_jf.py` and `report_jf.py` contain **no** import of, or path into, either package. Method names are just directory names passed with `--methods`; `DEFAULT_METHODS` (line 32) lists only SAM 2 methods. **No effect from the copy** — they are affected only once someone regenerates `SegMaskSam3OneStageNew/`.
- Result artifacts whose meaning becomes ambiguous, since they record `method: "SegMaskSam3OneStageNew"` and would no longer be tied to a distinguishable code state: `jf_sam3_onestagenew.json`, `jf_sam3_onestagenew_full.json`, `jf_raw_sam3_onestage.STALE-prefix-bug.json`, `jf_recheck.json`, `jf_summary.json`, `jf_report.txt`, `missing.json`.

### 7. Empirical check — the outputs already agree
Both output trees exist for 12 datasets, produced by independent runs ~21 h apart (OneStageNew Sep 3 03:06-12:03; MVOpt Sep 4 06:05-06:41, the latter under `SCSAM3_TRIM_CACHED_OUTPUTS=1` via `runMVOptThree.sh`):

```
AlexaMeadeFacePaint Barn Blocks Breakfast Carpark Dog
Fencing Frog MATF Painter PoznanStreet Welder
```
`diff -rq Data/MVSeg/<ds>/SegMaskSam3OneStageNew Data/MVSeg/<ds>/SegMaskSam3MVOpt` returns **0 differing files** for all 12, with matching PNG counts (315…1890). That is direct evidence the memory edits are output-neutral on every dataset where both were run. The three OOM datasets (`AlexaMeadeExhibit`, `CoffeeMartini`, `FlameSteak`) have MVOpt output but **no** OneStageNew output, so they are untested for equivalence — and note `jf_sam3_onestagenew_full.json` nonetheless carries `SegMaskSam3OneStageNew` entries for exactly those three, which is worth reconciling before trusting that file.

### 8. Notebooks, configs, CI
- `SCSam3/demo/MVSeg.json` — dataset geometry only (`start_frame`, `cam_list`, `prefix`…). No algo or path. No effect.
- `SCSam3/demo/DemoMultiSam2.ipynb`, `SCSam3/demo/SCSamSample.ipynb`, `SCSam/demo/DemoMultiSam.ipynb`, `SCSam3/sam3/examples/*.ipynb` — grep-clean of `demoSCSam3*`. No effect.
- `.github/copilot-instructions.md` — prose only; no reference. No effect. There is no CI workflow in the repo.
- `README.md` — mentions `SCSam3/` and `Data/MVSeg/` generically; no package reference. No effect.

### 9. In-package entry point reached by `cd`-ing in
`demoSCSam3OneStageNew/sam3_demoVideo.py` (the generator of the figure PNGs) would run under the new code. It never calls `RetireSpatialPredictor`, and its `RunNaiveTracking` (`SCSam3Video.py:194-231`) always sends `session_ids` + `spatial_idx` regardless of the new flag, so the only change it sees is the memory patches themselves. `test.py` is a 5-line scratch file. `SCSam3TrackerPredictorNewMem copy.py` is tracked in git but importable by no one (space in the filename) and referenced nowhere — (b) would delete it harmlessly.

### 10. The written plan says the opposite
`docs/analysis/SCHEDULE.md:9` states the policy explicitly: *"원본(`demoSCSam3OneStageNew`, `demoSCSam3OneStage`, 그 외 sibling 폴더)은 절대 건드리지 않는다 — 게시된 J&F 숫자가 그 폴더들의 출력에 묶여 있고, 문제가 생기면 즉시 되돌아갈 기준점이 필요하다."* And `SCHEDULE.md:243` records that the OneStageNew-neutrality argument rests specifically on *"`RetireSpatialPredictor`가 없으므로 `hasattr` 가드에 걸려 은퇴 코드가 실행되지 않는다"* — i.e. on the retire code **not** existing in that package. Copying invalidates that written argument, and `docs/tools/{raw_table,eval_validity,per_frame_analysis,gt_cov}.py` all hardcode `'SegMaskSam3OneStageNew'` as the OLD-code arm of their OneStage-vs-OneStageNew deltas.

### Bottom line
- **(a) `.py`-only copy:** the only genuine runtime change to any external consumer is that `runMVSeg.py --algo OneStageNew` now takes the `RetireSpatialPredictor` + `clear_autocast_cache` branch at lines 335-338. The `uses_spatial_predictor` branch resolves to the same request shape as today. Everything else — figures, eval, Dockerfile, all other scripts — is untouched. Fully reversible: all 13 `.py` files are tracked (`git checkout SCSam3/demoSCSam3OneStageNew`).
- **(b) directory copy:** additionally deletes 13 GB of untracked, git-ignored figure source PNGs in `0/`…`31/` and breaks `figures/make_figure.py` and `figures/make_baseline_figure.py` irreversibly.
- Independent of either: the copy collapses two registry entries onto one implementation while `DEFAULT_OUT` keeps two folder names, and `waitAndRunMVSeg.sh` will `--overwrite` the published `SegMaskSam3OneStageNew` baseline masks the next time it runs.