# Copilot Instructions for SCSegmentation

## 1) Big picture architecture
- Repository is a multi-variant SAM-based segmentation stack (SCSam, SCSam2, SCSam3, SCSamurai) focused on multi-view and video use-cases.
- Core pattern: `LoadImage` + `AddPoint`/mask input + `RunSegmentation` (in `SCSam/demo/SCSam.py`, `SCSam2/demo/SCSam2.py`, `SCSamurai/demo/...`).
- Geometry is built on COLMAP pose data in `*/demo/pose.py` (`load_colmap_data()`, `computecloseInfinity()`, `computeOffset*()`), used to project click coordinates across cameras.
- Model instantiation in `*/demo/build_sam.py` (SAM v2 hydra-based `build_sam2` and video predictors) and `SCSam/demo/SCSam.py` (SAM v1 `sam_model_registry`).

## 2) Essential workflows (run / test / debug)
- Canonical entrypoint: `SCSam2/demo/sam2_demo.py`; run with `python sam2_demo.py` after setting the dataset path in `sc.LoadImage(...)`.
- Batch metric flow in `SCSam2/demo/MVSeg.sh`: convert dataset via `MaskConvertMVSeg.py`, run `sam2_demoVideoNew_maskSingleInputMVSeg.py`, then `ComputeIOU.py` / `ComputeMOTA.py`.
- Dataset directory expected format: `<dataset>/sparse/0/{cameras.bin,images.bin,points3D.bin}` + `<dataset>/images/<frame>.jpg` or `.png`.
- Evaluation helpers in `SCSam2/demo/ComputeIOU*.py` and `ComputeMOTA*.py`.

## 3) Project-specific conventions
- Keep augmentations in the `demo/` folder; core algorithm is built as class wrappers (`SCSam2`, `SCSamurai`) with state fields `images`, `input_points`, `input_labels`, `masks`.
- Use `perms` from COLMAP to align frames/poses. Code depends on the permutation being stable (`pose.load_colmap_data()` returns both `w2c` and `c2w`).
- Visual output debugging uses pyplot overlays (`misc.show_mask`/`show_points`). Maintain this pattern for inspectability.
- For video tracking state, prefer `SAM2VideoPredictorSpatial` mechanics in `SCSam2/demo/SCSam2ImagePredictor.py` and `SCSamurai/demo/SCSamuraiImagePredictor.py`.

## 4) Integration/external dependencies
- Mandatory: `torch`, `numpy`, `opencv-python`, `matplotlib`, `PIL`, `tqdm`, `hydra-core`, `omegaconf`, plus `segment-anything` (v1) and `sam2` (v2) packages.
- Model checkpoints in `*/models/`: `sam_vit_h_4b8939.pth`, `sam2.1_hiera_large.pt`, and the `configs/sam2.1/*.yaml` path.
- COLMAP dependencies: `colmap_read_model.py` reads COLMAP binary files; ensure COLMAP output is generated consistently.

## 5) When editing as an AI
- Focus first on one subfolder: e.g. `SCSam2/demo/` contains the de facto pipeline; apply and test there before cross-copying to `SCSam3` or `SCSamurai`.
- When changing coordinate transform logic, update both `computeOffset` and `computeOffset1`, and add a small numeric smoke test in `SCSam2/demo/pose.py` or in a new script.
- Preserve the data flow: `input_points` → `SAM predictor (point_coords + labels)` → raw masks → `mask projection` via `pose` offsets.
- Avoid large refactors of `build_sam.py` unless targeting SAM2 hydra override behavior; most bugfixes are in the high-level wrapper classes and script-level default dataset names.

## 6) Missing / known friction points (to ask user)
- Confirm exact preferred model version: current scripts point to SAM 2.1 hiera large (__hard-coded path__), while `SCSam/` uses SAM v1.
- Confirm if `Data/MVSeg` and `Result/*` are considered canonical test artifacts for CI, or just local examples.

Ask for feedback: is the target primarily to standardize one pipeline (e.g. `SCSam2/demo`), or support all folders equally?