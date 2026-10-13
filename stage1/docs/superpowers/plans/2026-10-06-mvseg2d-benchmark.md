# MVSeg2D Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A benchmark that scores several methods on one task: given the reference camera's ground-truth label map at the reference frame of each MVSeg scene, produce per-object masks for every other camera, and compare them on the two remaining annotated cameras.

**Architecture:** Host-side pure-numpy scorer and reporter (`mvseg2d/`) that only know the result-folder format. Each method is a standalone `methods/<name>/run.py` that runs inside its own existing Docker image (`scsam3`, `scgeo:pi3`, `scgeo:dino`), reads one label map plus the images, and writes binary PNGs plus a `RUN.json`. Two-stage methods pass intermediate geometry/features through `Data/<scene>/Cache/`.

**Tech Stack:** Python 3.12; host venv made with `uv` (numpy, Pillow, opencv-python-headless, pytest); Docker images already on the machine: `scsam3` (torch 2.10, SAM 3 + `../SCSegmentation/SCSam3/demoSCSam3MVOpt` package), `scgeo:pi3` (Pi3X), `scgeo:dino` (transformers + DINOv2-large). RTX 6000 Ada 48 GB.

**Spec:** `docs/superpowers/specs/2026-10-06-mvseg2d-benchmark-design.md`

## Global Constraints

- Methods read only `Data/<scene>/images/*.jpg`, `Data/<scene>/masks/<ref_cam>.png` and `meta.json` (spec 2.1). Other annotated masks are read only by the scorer and by the `oracle` check method.
- Result format: `Data/<scene>/Result/<method>/<cam>/<id>.png`, 8-bit grey, values exactly 0 or 255, same (h, w) as the image; `Data/<scene>/Result/<method>/RUN.json` (spec 2.2).
- Scorer rejects a prediction whose shape differs from the ground truth; it never resizes (spec 4.4).
- Scored views: the annotated cameras minus the reference camera (2 per scene, 34 total). Object set: ids present in `masks/<ref_cam>.png`. Pairs whose target GT is empty are excluded from averages and counted as hallucinations when the prediction is non-empty (spec 2.3).
- Boundary tolerance: `bound_pix = ceil(0.008 * sqrt(h² + w²))`, DAVIS disk structuring element (as `../SCSegmentation/Data/MVSeg/eval_jf.py`).
- Size bins on target GT area: `<2000`, `2000-10000`, `>=10000` px. Hemisphere (fisheye) scenes: `AlexaMeadeExhibit`, `AlexaMeadeFacePaint`, `Dog`, `Welder` (spec 2.4).
- No threshold may be chosen by looking at the 34 scored views (spec 2.5). Method constants are fixed in code with a comment naming their source.
- `../SCSegmentation` is read-only: code is copied into this repo, never imported from there except the SAM 3 package (`SCSam3/demoSCSam3MVOpt`) and its weights (`SCSam3/hf_cache`), which are used in place via PYTHONPATH/HF_HOME.
- Docker runs use `--rm --gpus all --shm-size=32g --user $(id -u):$(id -g) -v /:/host -w /host$PWD -e HF_HUB_OFFLINE=1`. HF_HOME is `/host/home/sjpark/Documents/SCSegmentation/SCSam3/hf_cache` for `scsam3` and `/opt/hf` for `scgeo:*`.
- Commit after every task with the attribution trailer used in this repo's history.

## Review Focus

1. A prediction PNG with values other than {0, 255} (for example 0/1 or RGB): the reader must not silently score it as all-background. Test in Task 1 (`test_read_pred_rejects_non_binary`, RGB accepted via channel 0).
2. A method that failed on one scene (`RUN.json` has `error`): that (method, scene) must be excluded from averages and listed, not averaged as zeros. Tests in Task 3 and Task 4.
3. A target camera folder missing entirely, or an extra folder for a camera not in `meta.json`: missing counts as empty predictions, extra is ignored with a warning. Test in Task 3.
4. An object id in the label map that is absent from `objects_labels.json`: scored anyway with a warning; the label column shows `unknown`. Test in Task 3.
5. A method that writes the reference camera's folder itself (not via `common`): the scorer never scores the reference camera, so no leak. Test in Task 3 (`test_reference_camera_never_scored`).

---

### Task 1: Host environment and `mvseg2d/io.py`

**Files:**
- Create: `pyproject.toml`, `requirements-host.txt`, `mvseg2d/__init__.py`, `mvseg2d/io.py`
- Test: `tests/test_io.py`
- Modify: `.gitignore` (add `.venv/`)

**Interfaces:**
- Produces:
  - `HEMISPHERE: tuple[str, ...]`, `SIZE_BINS = ("<2000", "2000-10000", ">=10000")`, `size_bin(area_px: int) -> str`
  - `load_meta(data: str, scene: str) -> dict` (the `meta.json` dict)
  - `load_labels(data, scene) -> dict[int, dict]` (`objects_labels.json`, keys as int)
  - `load_labelmap(path: str) -> np.ndarray[uint8, (h, w)]` (single channel; channel 0 if 3-channel)
  - `labelmap_objects(lab) -> list[int]` (sorted non-zero ids)
  - `labelmap_to_mask(lab, obj_id) -> np.ndarray[bool]`
  - `read_pred_mask(path) -> np.ndarray[bool]`; raises `BadMaskValues` if any value not in {0, 255}; 3-channel → channel 0
  - `write_mask(path, mask_bool)` writes 0/255 uint8 PNG, creating parent dirs
  - `result_dir(data, scene, method) -> str`, `cache_dir(data, scene, provider) -> str`
  - `class BadMaskValues(ValueError)`, `class ShapeMismatch(ValueError)`

- [ ] **Step 1: Create the venv and lock the host deps**

```bash
cd /home/sjpark/Documents/SCSegmentation2D
~/.local/bin/uv venv .venv --python 3.12
printf 'numpy>=1.26\nPillow\nopencv-python-headless\npytest\n' > requirements-host.txt
~/.local/bin/uv pip install --python .venv/bin/python -r requirements-host.txt
echo ".venv/" >> .gitignore
```
Expected: `.venv/bin/python -c "import numpy, cv2, PIL, pytest"` exits 0.

- [ ] **Step 2: Write `pyproject.toml`** with `[tool.pytest.ini_options] testpaths = ["tests"]` and `pythonpath = ["."]`.

- [ ] **Step 3: Write the failing tests** in `tests/test_io.py`

```python
def test_size_bin_edges():
    assert size_bin(1999) == "<2000" and size_bin(2000) == "2000-10000" and size_bin(10000) == ">=10000"

def test_labelmap_roundtrip(tmp_path):
    lab = np.zeros((4, 5), np.uint8); lab[1:3, 1:3] = 7; lab[0, 4] = 2
    cv2.imwrite(str(tmp_path / "m.png"), lab)
    got = load_labelmap(str(tmp_path / "m.png"))
    assert labelmap_objects(got) == [2, 7]
    assert labelmap_to_mask(got, 7).sum() == 4

def test_write_then_read_pred_mask(tmp_path):
    m = np.zeros((3, 3), bool); m[1, 1] = True
    write_mask(str(tmp_path / "a" / "1.png"), m)
    back = read_pred_mask(str(tmp_path / "a" / "1.png"))
    assert back.dtype == bool and back.sum() == 1

def test_read_pred_rejects_non_binary(tmp_path):
    cv2.imwrite(str(tmp_path / "x.png"), np.array([[0, 1]], np.uint8))
    with pytest.raises(BadMaskValues): read_pred_mask(str(tmp_path / "x.png"))

def test_read_pred_accepts_rgb_channel0(tmp_path):
    rgb = np.zeros((2, 2, 3), np.uint8); rgb[0, 0] = 255
    cv2.imwrite(str(tmp_path / "x.png"), rgb)
    assert read_pred_mask(str(tmp_path / "x.png")).sum() == 1
```

- [ ] **Step 4: Run tests, verify they fail** — `.venv/bin/pytest tests/test_io.py -v` → ImportError.

- [ ] **Step 5: Implement `mvseg2d/io.py`** with the interfaces above. `HEMISPHERE = ("AlexaMeadeExhibit", "AlexaMeadeFacePaint", "Dog", "Welder")`.

- [ ] **Step 6: Run tests, verify pass** — `.venv/bin/pytest tests/test_io.py -v` → 5 passed.

- [ ] **Step 7: Commit** — `git add -A && git commit -m "mvseg2d: host env and mask/labelmap io"` (+ trailer).

---

### Task 2: `mvseg2d/metrics.py` (J and DAVIS F)

**Files:**
- Create: `mvseg2d/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `db_eval_iou(seg: bool[h,w], ann: bool[h,w]) -> float` (both empty → 1.0)
  - `seg2bmap(seg) -> bool[h,w]`, `disk(radius: int) -> uint8 kernel`
  - `bound_pix(h: int, w: int) -> int` = `ceil(0.008 * hypot(h, w))`
  - `db_eval_boundary(fg: bool, gt: bool, bound_px: int, se) -> float`
  - `crop_box(a, b, margin, shape) -> tuple|None`
  - `jf_pair(gt: bool[h,w], pred: bool[h,w], bp: int, se) -> tuple[float, float]` — computes on the union crop grown by `bp + 2` (identical result to full frame) and returns `(J, F)`.
- Source to copy verbatim: `../SCSegmentation/Data/MVSeg/eval_jf.py` lines 36–107 (`db_eval_iou`, `_seg2bmap`, `_disk`, `db_eval_boundary`, `crop_box`).

- [ ] **Step 1: Write the failing tests**

```python
def _sq(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool); m[y0:y1, x0:x1] = True; return m

def test_bound_pix_1080p(): assert bound_pix(1080, 1920) == 18

def test_identical_masks_are_perfect():
    g = _sq(100, 100, 20, 60, 20, 60); bp = bound_pix(100, 100)
    assert jf_pair(g, g.copy(), bp, disk(bp)) == (1.0, 1.0)

def test_disjoint_masks_are_zero():
    g = _sq(100, 100, 0, 10, 0, 10); p = _sq(100, 100, 50, 60, 50, 60); bp = bound_pix(100, 100)
    assert jf_pair(g, p, bp, disk(bp)) == (0.0, 0.0)

def test_half_overlap_iou():
    g = _sq(50, 50, 0, 10, 0, 20); p = _sq(50, 50, 0, 10, 10, 30)
    assert abs(db_eval_iou(p, g) - 1/3) < 1e-9

def test_one_pixel_shift_within_tolerance_has_F_one():
    g = _sq(200, 200, 50, 100, 50, 100); p = _sq(200, 200, 51, 101, 50, 100); bp = bound_pix(200, 200)
    J, F = jf_pair(g, p, bp, disk(bp)); assert F == 1.0 and 0.9 < J < 1.0

def test_crop_matches_full_frame():
    g = _sq(300, 400, 100, 150, 100, 180); p = _sq(300, 400, 110, 160, 90, 170); bp = bound_pix(300, 400); se = disk(bp)
    assert jf_pair(g, p, bp, se) == (db_eval_iou(p, g), db_eval_boundary(p, g, bp, se))
```

- [ ] **Step 2: Run, verify fail** — `.venv/bin/pytest tests/test_metrics.py -v`.
- [ ] **Step 3: Implement `mvseg2d/metrics.py`** by copying the five functions and adding `bound_pix`, `jf_pair`.
- [ ] **Step 4: Run, verify pass** — 6 passed.
- [ ] **Step 5: Commit** — `"mvseg2d: J and DAVIS boundary F metrics"`.

---

### Task 3: `mvseg2d/evaluate.py`

**Files:**
- Create: `mvseg2d/evaluate.py`
- Test: `tests/test_evaluate.py`, `tests/conftest.py` (fixture `fake_data(tmp_path)` building a 2-scene `Data/` tree)

**Interfaces:**
- Consumes: Task 1 io, Task 2 metrics.
- Produces:
  - `scored_views(meta) -> list[str]` = `[c for c in meta["mask_cams"] if c != meta["ref_cam"]]`
  - `view_dist(meta, cam) -> int` = `abs(meta["cams"].index(cam) - meta["cams"].index(meta["ref_cam"]))`
  - `evaluate_scene(data, scene, method) -> dict` with keys `scene`, `method`, `status` (`"ok"|"error"|"missing"`), `error` (str|None), `rows` (list), `hallucinations` (int), `warnings` (list[str]), `run` (RUN.json dict or None)
  - row dict: `scene, cam, obj, label, type, rig ("pinhole"|"hemisphere"), gt_px, pred_px, view_dist, size_bin, J, F`
  - `evaluate_method(data, method, scenes: list[str]) -> dict` = `{"method", "scenes": {scene: evaluate_scene(...)}}`
  - CLI: `python -m mvseg2d.evaluate --data Data --method <m> [--scenes a,b] --out scores/<m>.json`
- Rules (spec 2.3, 4.4, Review Focus): status `"error"` when `RUN.json` has a non-null `error`; `"missing"` when `Result/<method>` does not exist; otherwise `"ok"`. Missing PNG = empty prediction. `ShapeMismatch` raised (not caught) when a PNG's shape differs from GT. GT-empty pairs are not rows; `hallucinations += 1` if their prediction is non-empty. Extra camera folders → warning. Unknown ids → `label="unknown"` + warning.

- [ ] **Step 1: Write `tests/conftest.py` fixture `fake_data`**: two scenes `S1` (pinhole, cams `v0..v4`, ref `v2`, mask cams `v0,v2,v4`) and `Dog` (hemisphere, cams `camera_0001..0004`, ref `camera_0002`, mask cams `0002,0003,0004`), 40×60 label maps with objects 1 and 2 in every annotated view except object 2 absent from `S1/v4`; `objects_labels.json` only lists id 1; `meta.json` as `prepare_data.py` writes it.

- [ ] **Step 2: Write the failing tests**

```python
def test_oracle_like_results_score_one(fake_data):           # copies GT into Result/oracle
    r = evaluate_method(fake_data, "oracle", ["S1", "Dog"])
    assert all(row["J"] == 1.0 and row["F"] == 1.0 for s in r["scenes"].values() for row in s["rows"])

def test_missing_png_counts_as_empty(fake_data):              # Result/empty with no PNGs, RUN.json ok
    s = evaluate_scene(fake_data, "S1", "empty")
    assert s["status"] == "ok" and all(row["J"] == 0.0 for row in s["rows"]) and s["rows"][0]["pred_px"] == 0

def test_gt_empty_pair_excluded_and_hallucination_counted(fake_data):   # pred for obj 2 on S1/v4
    s = evaluate_scene(fake_data, "S1", "halluc")
    assert all(not (row["cam"] == "v4" and row["obj"] == 2) for row in s["rows"]) and s["hallucinations"] == 1

def test_error_run_json_marks_scene(fake_data):
    s = evaluate_scene(fake_data, "S1", "broken"); assert s["status"] == "error" and s["rows"] == []

def test_shape_mismatch_rejected(fake_data):
    with pytest.raises(ShapeMismatch): evaluate_scene(fake_data, "S1", "wrongsize")

def test_reference_camera_never_scored(fake_data):
    s = evaluate_scene(fake_data, "S1", "oracle"); assert {row["cam"] for row in s["rows"]} == {"v0", "v4"}

def test_unknown_label_and_extra_camera_warn(fake_data):        # Result/oracle/S1 has extra folder "v9"
    s = evaluate_scene(fake_data, "S1", "oracle")
    assert any("v9" in w for w in s["warnings"]) and any(row["label"] == "unknown" for row in s["rows"])

def test_row_fields(fake_data):
    row = evaluate_scene(fake_data, "Dog", "oracle")["rows"][0]
    assert row["rig"] == "hemisphere" and row["size_bin"] in SIZE_BINS and row["view_dist"] >= 1
```

- [ ] **Step 3: Run, verify fail.**
- [ ] **Step 4: Implement `mvseg2d/evaluate.py`** per the interfaces; `main()` writes JSON with `indent=1`.
- [ ] **Step 5: Run, verify pass** — 8 passed.
- [ ] **Step 6: Commit** — `"mvseg2d: per-pair evaluator"`.

---

### Task 4: `mvseg2d/report.py`

**Files:**
- Create: `mvseg2d/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `scores/<method>.json` as written by Task 3.
- Produces:
  - `flatten(score: dict) -> list[dict]` (rows with `method` added; error/missing scenes contribute none)
  - `aggregate(rows, by: tuple[str, ...]) -> dict[tuple, dict]` with `n`, `J`, `F`, `JF` means. Aggregation order for the headline: pair → view mean → scene mean → overall mean (spec 2.3).
  - `headline(rows) -> dict[method, {"J","F","JF","n_scenes"}]`
  - `tables(scores: list[dict]) -> str` Markdown with sections: Headline (+ per-method runtime/GPU from RUN.json, counted scenes, failed scenes footnote, hallucinations), Per scene, Size bins, Rig (pinhole/hemisphere), View distance bins (`1`, `2-3`, `>=4`).
  - CLI: `python -m mvseg2d.report scores/*.json --out scores/report.md`

- [ ] **Step 1: Write the failing tests** (synthetic score dicts, two methods, one error scene)

```python
def test_headline_scene_mean_not_pair_mean():   # scene A: 2 pairs J=1,0 ; scene B: 1 pair J=1  → J = (0.5+1)/2
    assert abs(headline(rows)["m"]["J"] - 0.75) < 1e-9

def test_error_scene_excluded_and_footnoted():
    md = tables([score_with_error]); assert "실패" in md and "S_err" in md

def test_size_and_rig_tables_present():
    md = tables([score]); assert "<2000" in md and "hemisphere" in md and "view_dist" in md.lower() or "시점차" in md
```

- [ ] **Step 2: Run, verify fail.** - [ ] **Step 3: Implement.** - [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `"mvseg2d: markdown report over the spec's axes"`.

---

### Task 5: `methods/common.py`, `oracle`, `empty`, scorer end-to-end on real data

**Files:**
- Create: `methods/__init__.py`, `methods/common.py`, `methods/oracle/run.py`, `methods/empty/run.py`, `README.md`
- Test: `tests/test_common.py`

**Interfaces:**
- Produces (`methods/common.py`, importable from host venv and from every container; no torch import at module level):
  - `REPO = dirname(dirname(abspath(__file__)))`; `SIBLING = join(dirname(REPO), "SCSegmentation")`; `SAM3_PKG = join(SIBLING, "SCSam3", "demoSCSam3MVOpt")`
  - `@dataclass Scene: name, cams: list[str], ref_cam, mask_cams, hw: tuple[int,int], image_paths: dict[str,str], ref_labels: np.uint8[h,w], objects: list[int]`
  - `load_scene(data, scene) -> Scene`
  - `all_scenes(data) -> list[str]` (dirs with `meta.json`, sorted)
  - `parse_args(description, extra: callable|None) -> argparse.Namespace` with `--scene` (name or `all`), `--data` (default `Data`), `--out` (method folder name), `--overwrite`
  - `write_result(data, scene_name, method, cam, obj_id, mask_bool)`
  - `copy_reference(scene: Scene, data, method)` writes the reference camera's GT per object
  - `class RunTimer` context manager → `.seconds`, `.gpu_mb` (via `torch.cuda.max_memory_allocated()` if torch importable and CUDA available, else `None`)
  - `write_run(data, scene_name, method, seconds, gpu_mb, config: dict, error: str|None = None)` → RUN.json with `method, scene, seconds, gpu_mb, config, git` (short hash of this repo via `git rev-parse --short HEAD`, or `None`), `date` (ISO), `error`
  - `run_all(args, fn: callable[[Scene, Namespace], dict])`: for each scene, `copy_reference`, time `fn` (returns config dict), catch `Exception` → `write_run(..., error=traceback)`, print one line per scene, continue
- `oracle/run.py`: for each non-reference annotated cam, copies GT per object (reads `masks/<cam>.png` — allowed for this check only). `empty/run.py`: writes nothing but RUN.json.

- [ ] **Step 1: Write failing tests** for `load_scene` on `fake_data` (objects, hw, ref), `write_run` schema, `run_all` continuing after an exception and writing `error`.
- [ ] **Step 2: Run, verify fail.** - [ ] **Step 3: Implement.** - [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: End-to-end on real data**

```bash
.venv/bin/python methods/oracle/run.py --scene all --out oracle
.venv/bin/python methods/empty/run.py  --scene all --out empty
.venv/bin/python -m mvseg2d.evaluate --method oracle --out scores/oracle.json
.venv/bin/python -m mvseg2d.evaluate --method empty  --out scores/empty.json
.venv/bin/python -m mvseg2d.report scores/oracle.json scores/empty.json --out scores/report.md
```
Expected: headline oracle J=F=1.000 on 17 scenes, empty J=F=0.000; report prints hallucinations 0 for both. Record the number of scored pairs here in the plan's commit message.

- [ ] **Step 6: Write `README.md`** (what the repo is, how to prepare data, run a method, evaluate, report; one paragraph each).
- [ ] **Step 7: Commit** — `"methods: common runner, oracle and empty checks; scorer verified on 17 scenes"`.

---

### Task 6: `sam3_prop` — SAM 3 pseudo-video propagation (baseline)

**Files:**
- Create: `methods/sam3_prop/run.py`, `scripts/docker_run.sh`, `scripts/run_sam3_prop.sh`
- Test: `tests/test_sam3_prop.py` (pure parts only)

**Interfaces:**
- Consumes: `methods/common.py`; SAM 3 package in place: `sys.path.insert(0, common.SAM3_PKG)`; `from build_scsam3 import build_scsam3_video_predictor`; `from io_utils import load_video_frames, AsyncVideoFrameCPUToGPU`. The predictor protocol is the one in `../SCSegmentation/MVSeed/run_seed.py` lines 170–215: `handle_request(start_session{images, orig_height, orig_width})`, `reset_session`, `add_prompt{frame_index, mask: float32 tensor, obj_id}`, `handle_stream_request(propagate_in_video{propagation_direction:"both"})`, outputs `out_binary_masks`, `out_obj_ids`. Never pass `start_frame_index`.
- Produces:
  - `view_order(order: str, n: int, ref_idx: int) -> list[int]` for `index` and `ref_outward` (copy of `run_seed.view_order`, without `reverse`)
  - `build_pseudo_video(scene: Scene, order_seq: list[int], folder: str) -> None`: symlinks `images/<cam>.jpg` as `<folder>/%06d.jpg` in sequence order (load_video_frames sorts by `int(stem)`, so names must be integers)
  - `--order {index,ref_outward}` (default `index`); method folder name is `--out` (use `sam3_prop` and `sam3_prop_refout`)
  - `scripts/docker_run.sh <image> <cmd...>` applies the Global Constraints docker flags and the image-specific HF_HOME, and sets `PYTHONPATH=/host$PWD`
  - `scripts/run_sam3_prop.sh <scene|all> [--order ...]`
- Config recorded in RUN.json: `order`, `sequence`, `prompt_position`, `fill_hole_area: 0`.

- [ ] **Step 1: Write failing tests** for `view_order("ref_outward", 5, 2) == [2, 1, 3, 0, 4]`, `view_order("index", 3, 0) == [0, 1, 2]`, and `build_pseudo_video` producing `000000.jpg..` symlinks that resolve to the right cameras.
- [ ] **Step 2: Run, verify fail.** - [ ] **Step 3: Implement `run.py`** (pure parts + GPU `main` guarded so tests import without torch) and the two shell scripts.
- [ ] **Step 4: Run tests, verify pass.**
- [ ] **Step 5: Smoke on Blocks**

```bash
scripts/run_sam3_prop.sh Blocks
.venv/bin/python -m mvseg2d.evaluate --method sam3_prop --scenes Blocks --out scores/sam3_prop_blocks.json
```
Expected: 10 camera folders under `Data/Blocks/Result/sam3_prop/`, PNGs 1920×1080, J on cam0/cam9 in the 0.8–0.95 range (prior stage-1 mean 0.836).

- [ ] **Step 6: Reproducibility check (spec 2.5)**: run Blocks a second time with `--out sam3_prop_rep`, evaluate, and write the per-pair |ΔJ| max into `docs/results-stage1.md` (create the file with just this note for now).
- [ ] **Step 7: Full run** — `scripts/run_sam3_prop.sh all` then `scripts/run_sam3_prop.sh all --order ref_outward --out sam3_prop_refout`; evaluate both to `scores/`.
- [ ] **Step 8: Commit** — `"methods: sam3_prop baseline (index, ref_outward) on 17 scenes"`.

---

### Task 7: `pi3_reproj` — π³ pointmap lifting and reprojection

**Files:**
- Create: `mvseg2d/geo/__init__.py`, `mvseg2d/geo/warp.py` (copy of `../SCSegmentation/MVSeed/geo/warp.py`, unchanged), `mvseg2d/geo/cameras.py`, `methods/pi3_reproj/run.py`, `methods/pi3_reproj/geometry.py`, `scripts/run_pi3_reproj.sh`
- Test: `tests/test_geo.py`
- Modify: spec section 3 row `pi3_reproj`: replace "변형 `--tiles`로 고해상 타일링" with "변형 `--pixel-limit`로 작업 해상도 상향(255k → 1M px)". Tiling is dropped: each Pi3 tile would have its own coordinate frame, so tiles cannot be warped against one another.

**Interfaces:**
- Produces (`mvseg2d/geo/cameras.py`):
  - `@dataclass Pinhole: fx, fy, cx, cy; has_distortion = False; k1 = k2 = 0.0` (what `warp.py` reads from a camera)
  - `@dataclass Pose: R: np.ndarray[3,3], tvec: np.ndarray[3]` (what `warp.pose_matrices` reads)
  - `from_c2w(K: 3x3, c2w: 4x4) -> tuple[Pinhole, Pose]` with the `+0.5` pixel-centre shift of `e4b_project.cameras_from_pi3`
  - `scale(cam: Pinhole, sx, sy) -> Pinhole`
- Produces (`methods/pi3_reproj/geometry.py`, runs in `scgeo:pi3`): `run_pi3(scene: Scene, pixel_limit: int) -> None` writes `Cache/pi3/<cam>.npz` with `depth (float32, work hw, 0 where conf<=0.1)`, `conf`, `K (3x3 at work res)`, `c2w (4x4)`, and `Cache/pi3/MANIFEST.json` `{work_hw, pixel_limit, model: "yyfz233/Pi3X", cams}`. Reference camera is input index 0. Copy `target_size`, Pi3X call and `recover_intrinsic_from_rays_d` usage from `e4b_project.run_pi3`.
- Produces (`methods/pi3_reproj/run.py`): for each target cam, `backward ∪ forward` warp of each object's reference mask (reference GT downsampled to work res by nearest) at work resolution, upsampled to the original by nearest, written via `common.write_result`. `--pixel-limit` (default `255000`; variant `1000000` with `--out pi3_reproj_hr`). `--skip-geometry` reuses the cache. Config: `pixel_limit, work_hw, occ_tol 0.01, close_kernel 5, conf_min 0.1`.
- `scripts/run_pi3_reproj.sh <scene|all> [--pixel-limit N --out NAME]` runs `run.py` in `scgeo:pi3` (geometry and warp in the same container; warp is numpy).

- [ ] **Step 1: Write failing tests** in `tests/test_geo.py`: (a) `from_c2w` of an identity pose gives `Pose.R == I`, `tvec == 0`, `cx == K[0,2] + 0.5`; (b) synthetic two-view consistency: a fronto-parallel plane at depth 5 seen by two pinhole cameras translated 0.5 along x; a 10×10 square mask in view A backward-warps into view B shifted by `fx*0.5/5` px (compute expected with the same intrinsics) with IoU > 0.9 against the analytic shift.
- [ ] **Step 2: Run, verify fail.** - [ ] **Step 3: Implement `cameras.py`, copy `warp.py`, implement `geometry.py` and `run.py`.**
- [ ] **Step 4: Run tests, verify pass.**
- [ ] **Step 5: Smoke on Blocks** — `scripts/run_pi3_reproj.sh Blocks`; expect `Cache/pi3/` with 10 npz, Result PNGs at 1920×1080, evaluate → J well below sam3_prop on small objects, comparable on large (E4b finding).
- [ ] **Step 6: Full runs** — `all` at default, and `all --pixel-limit 1000000 --out pi3_reproj_hr` (if CUDA OOM on 46-camera scenes, record the failure in RUN.json via `run_all` and move on; do not lower the limit per scene).
- [ ] **Step 7: Update the spec line, commit** — `"methods: pi3_reproj (255k and 1M px) on 17 scenes"`.

---

### Task 8: `pi3_sam3` — SAM 3 refinement of the reprojected masks

**Files:**
- Create: `methods/pi3_sam3/run.py`, `methods/sam3_single.py`, `scripts/run_pi3_sam3.sh`
- Test: `tests/test_prompts.py`

**Interfaces:**
- Produces (`methods/sam3_single.py`, shared by Tasks 8–9, runs in `scsam3`):
  - `load_model()` → `build_scsam3_video_model()` from `SAM3_PKG`
  - `load_frame(path) -> (frame, h, w)`: read with `cv2`, convert BGR→RGB, wrap exactly as `load_video_frames` + `AsyncVideoFrameCPUToGPU(..., offload_video_to_cpu=True)[0]` do for a one-image folder (implementer: either call those two on a one-symlink temp folder, or replicate their tensor layout; the first is simpler and proven)
  - `segment(model, frame, h, w, points_xy: list[tuple[float,float]], labels: list[int]) -> np.ndarray[bool, (h,w)]`: one fresh `model.init_state([frame], h, w)`, one `model.add_tracker_new_points(inference_state, frame_idx=0, obj_id=1, points=rel_tensor, labels=int32_tensor)` with relative coords `((x+0.5)/w, (y+0.5)/h)`; returns `out_binary_masks[0] > 0`, or an all-False mask when no mask is returned; deletes the state. Labels: `1` positive point, `2` box top-left, `3` box bottom-right (the SAM 2 convention `add_new_points_or_box` uses).
- Produces (`methods/pi3_sam3/run.py`):
  - `prompts_from_mask(m: bool[h,w]) -> tuple[list[tuple], list[int]] | None`: `None` if `m` is empty; else the bounding box as two points labelled 2 and 3 plus one positive point at the argmax of `cv2.distanceTransform(m, DIST_L2, 5)` (deepest interior pixel) labelled 1
  - for each target cam and object: read `Result/<--source>/<cam>/<id>.png` (default source `pi3_reproj`), build prompts, `segment`, write. Empty source → write nothing (empty prediction). Config: `source`, prompt rule name `"bbox+dt_center"`.
- `scripts/run_pi3_sam3.sh <scene|all> [--source NAME --out NAME]`.

- [ ] **Step 1: Write failing tests** for `prompts_from_mask`: empty → `None`; a 20×30 rectangle at (10..30, 5..35) → labels `[2, 3, 1]`, box points `(5,10)` and `(34,29)`, positive point inside the rectangle; an L-shape → positive point lies on the mask (not in the concavity).
- [ ] **Step 2: Run, verify fail.** - [ ] **Step 3: Implement.** - [ ] **Step 4: Run tests, verify pass.**
- [ ] **Step 5: Smoke on Blocks** — `scripts/run_pi3_sam3.sh Blocks`; evaluate; expect F to rise versus `pi3_reproj` on the same pairs.
- [ ] **Step 6: Full runs** — `all` from `pi3_reproj`, and `all --source pi3_reproj_hr --out pi3hr_sam3`.
- [ ] **Step 7: Commit** — `"methods: pi3_sam3 refinement on 17 scenes"`.

---

### Task 9: `dino_sam3` — DINOv2 correspondence points, SAM 3 segmentation

**Files:**
- Create: `methods/dino_sam3/prompts.py` (runs in `scgeo:dino`), `methods/dino_sam3/run.py` (runs in `scsam3`), `scripts/run_dino_sam3.sh`
- Test: `tests/test_dino_prompts.py` (pure parts)

**Interfaces:**
- Produces (`prompts.py`): copy `resized_hw`, `patch_tokens`, `prototype`, `peaks` from `../SCSegmentation/MVSeed/corr/dino_prompt.py` with constants `MODEL_ID="facebook/dinov2-large"`, `PATCH=14`, `LONG_SIDE=1036`, `K_CANDIDATES=5`, `NMS_PATCHES=3` (comment: fixed before this benchmark in stage1-E3-prereg). Writes `Cache/dino/prompts.json`: `{cam: {obj: [{"point": [x, y], "cos": c}, ...]}}` in original pixel coordinates, candidates sorted by cosine descending.
- Produces (`run.py`): for each target cam and object, take the first candidate with `cos >= TAU_SIM (0.5)`; `sam3_single.segment` with that single positive point; write the mask (empty if no candidate passes). Config: `tau_sim 0.5, k 5, long_side 1036, model`.
- `scripts/run_dino_sam3.sh <scene|all>` runs `prompts.py` in `scgeo:dino` then `run.py` in `scsam3`.

- [ ] **Step 1: Write failing tests**: `resized_hw(1080, 1920)` sides are multiples of 14 with long side ≤ 1036; `peaks` on a synthetic similarity map with two separated maxima returns both, the higher first, and suppresses a neighbour within the NMS radius.
- [ ] **Step 2: Run, verify fail.** - [ ] **Step 3: Implement.** - [ ] **Step 4: Run tests, verify pass.**
- [ ] **Step 5: Smoke on Blocks**, evaluate.
- [ ] **Step 6: Full run `all`.**
- [ ] **Step 7: Commit** — `"methods: dino_sam3 on 17 scenes"`.

---

### Task 10: Stage-1 report

**Files:**
- Create/Modify: `docs/results-stage1.md`, `scores/report.md` (generated; gitignored), `README.md` (results pointer)

- [ ] **Step 1: Evaluate every method folder present** and generate the report:

```bash
for m in sam3_prop sam3_prop_refout pi3_reproj pi3_reproj_hr pi3_sam3 pi3hr_sam3 dino_sam3; do
  .venv/bin/python -m mvseg2d.evaluate --method $m --out scores/$m.json; done
.venv/bin/python -m mvseg2d.report scores/*.json --out scores/report.md
```

- [ ] **Step 2: Write `docs/results-stage1.md`**: paste the headline, per-rig, size-bin and view-distance tables; list failed (method, scene) pairs with their RUN.json error first lines; the Blocks repeat-run |ΔJ| from Task 6; runtime and GPU per method; three sentences of reading per table, stating only what the tables show. End with the open question for stage 2 (which pretrained multi-view models have public weights) without answering it.
- [ ] **Step 3: Commit** — `"docs: stage-1 results for the five first-round methods"`.
