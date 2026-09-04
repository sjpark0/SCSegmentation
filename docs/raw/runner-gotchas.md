Repo root: `/home/sjpark/Documents/SCSegmentation` (referred to below as **`$R`**). All citations are `$R`-prefixed paths with line numbers.

---

# SCSam3 runners — reference

## 0. Files covered

| File | Role |
|---|---|
| `/home/sjpark/Documents/SCSegmentation/SCSam3/runMVSeg.py` | Runner for the `OneStage` / `OneStageNew` / `MVOpt` demo packages |
| `/home/sjpark/Documents/SCSegmentation/SCSam3/runMVSegForSam2.py` | Runner for the `demoSCSam3ForSam2New` (SAM 2-style API) package |
| `/home/sjpark/Documents/SCSegmentation/SCSam3/runMVSegAll.sh` | Full sweep, both algos × all datasets, then J&F |
| `/home/sjpark/Documents/SCSegmentation/SCSam3/waitAndRunMVSeg.sh` | GPU-wait + smoke test + sweep |
| `/home/sjpark/Documents/SCSegmentation/SCSam3/runMVOptThree.sh` | Rescue runs on the memory-fixed package |
| `/home/sjpark/Documents/SCSegmentation/SCSam3/runForSam2Three.sh` | Rescue runs on the ForSam2New variant |
| `/home/sjpark/Documents/SCSegmentation/SCSam3/launch_container.sh` | Interactive container shell (2 lines, no options) |

---

## (a) CLI surface

### `runMVSeg.py` (`$R/SCSam3/runMVSeg.py:49-68`)

| Arg | Default | Effect |
|---|---|---|
| `dataset` (positional, required) | — | Key looked up in the config JSON; exits listing available keys if absent (`:52`, `:74-75`) |
| `--algo` | `OneStageNew` | Choices are `sorted(ALGOS)` = `MVOpt, OneStage, OneStageNew`. Selects the demo dir put on `sys.path` (`:53`, `:116`) |
| `--out` | `None` → `DEFAULT_OUT[algo]` | Output folder name **under the dataset dir** (`:54-55`, `:265`, `:293`) |
| `--config` | `<runner dir>/demo/MVSeg.json` | Absolute at parse time because `HERE` is absolute (`:35`, `:56`) |
| `--data-root` | `<runner dir>/../Data/MVSeg` | `os.path.abspath()`-ed at `:267`, i.e. **before** the `chdir` |
| `--device` | `None` → `"cuda" if torch.cuda.is_available() else "cpu"` | Passed to `MVSegVideo(device)` (`:58`, `:301`, `:308`) — **but see §5.9: all three ALGOS packages ignore it** |
| `--track-cams` | `None` → `"all"` if `algo in NEEDS_ALL_VIEWS` else `"written"` | Choices `written`/`all`; controls how many temporal sessions get opened (`:59-63`, `:304-306`) |
| `--overwrite` | `False` | Skips the "output dir exists and is non-empty" abort; does **not** delete anything (`:64-65`, `:294-295`) |
| `--dry-run` | `False` | Prints the resolved plan and returns before `build_runner` (`:66-67`, `:296-298`) |

Env read at import time: `SPATIAL_START_IMPLICIT` (default `"1"`, `:37`) and `PYTORCH_CUDA_ALLOC_CONF` set via `setdefault` to `expandable_segments:True` (`:33`).

Config keys consumed (`:71-88`): `folder`, `start_frame`, `num_frame`, `cam_list`, `prefix`, `prefix1`, and `perms` — where `perms` falls back to `range(start_cam, num_cam + start_cam)` when absent (`:77-79`).

### `runMVSegForSam2.py` (`$R/SCSam3/runMVSegForSam2.py:30-40`)

| Arg | Default | Effect |
|---|---|---|
| `dataset` (positional, required) | — | Same lookup, terser error (`:33`, `:46`) |
| `--out` | `"SegMaskSam3ForSam2New"` (the module constant `DEFAULT_OUT`, `:27`) | Output folder under the dataset dir |
| `--config` | `<runner dir>/demo/MVSeg.json` | `:35` |
| `--data-root` | `<runner dir>/../Data/MVSeg` | abspath'ed at `:62`, before the `chdir` at `:101` |
| `--device` | `None` → `cuda`/`cpu` | `:37`, `:171`; **is honoured** — `build_scsam3_video_model_newmem` does `model.to(device=device)` (`$R/SCSam3/demoSCSam3ForSam2New/build_scsam3.py:988`) |
| `--overwrite` | `False` | `:38`, `:91-92` |
| `--dry-run` | `False` | `:39`, `:93-95` |

There is **no `--algo` and no `--track-cams`**: the package is hard-wired (`ALGO_DIR = HERE/demoSCSam3ForSam2New`, `:26`) and every camera in `perms` gets a temporal state (`:117-131`). `SPATIAL_START_IMPLICIT` is not read here.

---

## (b) Registration tables (verbatim, `$R/SCSam3/runMVSeg.py:36-46`)

```python
# let the model infer the cross-view start from the prompted view
SPATIAL_START_IMPLICIT = os.environ.get("SPATIAL_START_IMPLICIT", "1") == "1"
ALGOS = {"OneStage": "demoSCSam3OneStage",
         "OneStageNew": "demoSCSam3OneStageNew",
         "MVOpt": "demoSCSam3MVOpt"}
DEFAULT_OUT = {"OneStage": "SegMaskSam3OneStage",
               "OneStageNew": "SegMaskSam3OneStageNew",
               "MVOpt": "SegMaskSam3MVOpt"}
# MVOpt is OneStageNew with the memory fixes; it shares the cross-view memory
# design, so it needs every view tracked, like OneStageNew.
NEEDS_ALL_VIEWS = ("OneStageNew", "MVOpt")
```

`runMVSegForSam2.py` has no tables — just `$R/SCSam3/runMVSegForSam2.py:25-27`:

```python
HERE = os.path.dirname(os.path.abspath(__file__))
ALGO_DIR = os.path.join(HERE, "demoSCSam3ForSam2New")
DEFAULT_OUT = "SegMaskSam3ForSam2New"
```

---

## (c) Shell scripts

| Script | Drives | Env vars (default) | Container flags |
|---|---|---|---|
| `runMVSegAll.sh` | `python runMVSeg.py <ds> --algo <algo> [EXTRA]`, one container per (algo, dataset) | `IMAGE` (`scsam3`) — the **only** env var it reads (`:19`) | `--rm --gpus all --shm-size=32g -v /:/host -w /host$R/SCSam3` (`:58-59`) |
| `waitAndRunMVSeg.sh` | `./runMVSegAll.sh` twice (smoke test, then sweep) | `NOT_BEFORE` (`date -d 'tomorrow 03:00'` epoch), `GIVE_UP` (`tomorrow 09:00`), `FREE_MIB` (`40000`), `POLL` (`300` s) (`:19-22`) | none directly |
| `runMVOptThree.sh` | `python runMVSeg.py <ds> --algo ${ALGO:-MVOpt} [--out $OUT] --overwrite` (`:16`) | `ALGO` (`MVOpt`), `OUT` (unset → flag omitted); sets `SCSAM3_TRIM_CACHED_OUTPUTS=1` **inside** the container (`:14`) | `--rm --gpus all --shm-size=32g --memory=90g --memory-swap=90g` (`:13`) |
| `runForSam2Three.sh` | `python runMVSegForSam2.py <ds> --overwrite` (`:15`) | none (image name `scsam3` hardcoded, `:14`) | `--rm --gpus all --shm-size=32g --memory=90g --memory-swap=90g` (`:13`) |
| `launch_container.sh` | interactive `scsam3` shell | `DISPLAY` | `--shm-size=64g --gpus all -it --rm`, X11 socket bind, `-v /:/host -w /host$PWD` (`:2`) |

**`runMVSegAll.sh` details** (`$R/SCSam3/runMVSegAll.sh`)

- `ALL_DATASETS` (`:22-24`), 15 entries: `AlexaMeadeExhibit AlexaMeadeFacePaint Barn Blocks Breakfast Carpark CoffeeMartini Dog Fencing FlameSteak Frog MATF Painter PoznanStreet Welder`.
- `ALGOS=(OneStage OneStageNew)` (`:25`) — **`MVOpt` is not in the sweep**, despite being registered in the runner.
- `OUTNAME` map (`:26`): `OneStage→SegMaskSam3OneStage`, `OneStageNew→SegMaskSam3OneStageNew`. Used only for the eval step.
- Options: `--algo X` **replaces** the whole array with one element (`:34`); `--dry-run` / `--overwrite` are appended to `EXTRA` and forwarded to the Python runner (`:35-36`); `--no-eval` sets `RUN_EVAL=0` (`:37`); `-h/--help` prints lines 2-20 of itself with `# ` stripped (`:38`); any other `-*` exits 2 (`:39`); bare words become dataset names (`:40`).
- Logs to `$R/SCSam3/logs/mvseg-<YYYYmmdd-HHMMSS>/<algo>-<ds>.log` (`:45-48`).
- After the loops it `chown -R`s `$R/Data/MVSeg` back to the invoking uid/gid from inside a container (`:75-76`) — the runs write as root.
- Eval step runs only when `RUN_EVAL=1` **and** `--dry-run` is not in `EXTRA` (`:88`): `eval_jf.py --methods <outnames> --out jf_raw_sam3.json` in `$R/Data/MVSeg` (`:93-95`), chowns the JSON (`:96-97`), then `report_jf.py` on the SAM 3 numbers (`:100-101`) and, if `$R/Data/MVSeg/jf_raw.json` exists, a side-by-side report adding `SegMask1 SegMaskNew1 SegMaskNew2 SegMaskNew3` (`:104-110`). Both helper scripts and `jf_raw.json` exist on disk today.

**`waitAndRunMVSeg.sh` details** (`$R/SCSam3/waitAndRunMVSeg.sh`)

- Polls `nvidia-smi --query-gpu=memory.free` every `POLL` seconds; starts when `now ≥ NOT_BEFORE && free ≥ FREE_MIB`; exits **3** at `GIVE_UP` after dumping `--query-compute-apps` (`:31-52`).
- Smoke test: `./runMVSegAll.sh --algo OneStageNew --no-eval --overwrite Frog`; failure ⇒ exit **4** (`:54-59`). Success is judged by `$? == 0` **and** `grep -q "all runs finished"` over the whole cumulative log (`:56`).
- Then `./runMVSegAll.sh --overwrite` (all 15 datasets × the default 2 algos) and exits with its status (`:62-67`).
- Everything appends to `$R/SCSam3/logs/auto-<STAMP>.log` (`:25-27`).

**`runMVOptThree.sh` / `runForSam2Three.sh`** — same skeleton: iterate `"$@"`, one container per dataset, per-dataset log under `logs/mvopt-<STAMP>/` resp. `logs/forsam2-<STAMP>/`, print `ok/FAILED(status)` with elapsed seconds, and on failure `grep -aE` the log tail (`OutOfMemory|Error|Killed` vs `Error|error|Killed`) — then one final `chown -R` container pass. Neither takes flags; both pass `--overwrite` unconditionally.

---

## Non-obvious behaviour

### 1. The `sys.path` surgery in `build_runner` — and why it is mandatory

`$R/SCSam3/runMVSeg.py:123-125` (mirrored at `$R/SCSam3/runMVSegForSam2.py:99-101`):

```python
sys.path[:] = [p for p in sys.path if p and os.path.abspath(p) != HERE]
sys.path.insert(0, algo_dir)
os.chdir(algo_dir)
```

Facts, verified:

- `$R/SCSam3/sam3/` has **no `__init__.py`** (the inner `$R/SCSam3/sam3/sam3/__init__.py` does exist — the checkout is `sam3/sam3/`). So the outer directory is only a PEP 420 namespace *portion*.
- In the `scsam3` image, `sam3` is installed editable from `/opt/sam3`, and setuptools registers `_EditableFinder` **after** `PathFinder` on `sys.meta_path` — observed order: `['DistutilsMetaFinder', 'BuiltinImporter', 'FrozenImporter', 'PathFinder', '_EditableFinder', 'VendorImporter']`. `/opt/sam3` itself is *not* on `sys.path`.
- Consequence: while `$R/SCSam3` is on `sys.path`, `PathFinder` resolves `sam3` to the namespace portion first and `_EditableFinder` is never consulted. Reproduced in the container with cwd `$R/SCSam3`:

  ```
  sam3.__file__ = None
  path = ['/host/home/sjpark/Documents/SCSegmentation/SCSam3/sam3']
  FAIL: TypeError expected str, bytes or os.PathLike object, not NoneType
  ```
  The `TypeError` is from `pkg_resources.resource_filename("sam3", "assets/bpe_simple_vocab_16e6.txt.gz")`, which is exactly what the model builder calls for the BPE tokenizer (`$R/SCSam3/demoSCSam3MVOpt/build_scsam3.py:645-648`, `:734-737`, `:874-877`, feeding `_create_text_encoder`/`SimpleTokenizer` at `:549-551`). This confirms the code comment at `$R/SCSam3/runMVSeg.py:118-122`.
- With cwd `$R/SCSam3/demoSCSam3MVOpt` the same probe returns `sam3.__file__ = /opt/sam3/sam3/__init__.py` and `bpe = /opt/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz`.
- The `if p` in the filter also drops the empty-string entry (cwd) unconditionally, not just `HERE`. That is load-bearing: `python runMVSeg.py` puts the script's own directory at `sys.path[0]`, and `python -c` puts `''` there; both denote `$R/SCSam3`.
- `os.chdir(algo_dir)` is **not** what makes the imports work (`sys.path.insert` does). It matters because the demo entry scripts in those packages use relative paths — e.g. `$R/SCSam3/demoSCSam3MVOpt/sam3_demoVideo.py:20` `sc.LoadVideo_File("../../Data/VideoSample_1", perms)` and its `os.makedirs(f"{m}")` output dirs, which is why `demoSCSam3OneStage/`, `demoSCSam3OneStageNew/` and `demoSCSam3ForSam2New/` all contain numeric folders `0`…`31`. **Both runners abspath their own config/data paths before the chdir** (`runMVSeg.py:35,56,267`; `runMVSegForSam2.py:35,62`), so the chdir cannot corrupt them — but any new relative path added to a runner after that point would break.

### 2. `SPATIAL_START_IMPLICIT` and what an explicit start index breaks

`$R/SCSam3/runMVSeg.py:205-210`:

```python
request = dict(type="propagate_in_video", session_id=..., propagation_direction="both")
if not SPATIAL_START_IMPLICIT:
    request["start_frame_index"] = view_index
```

- The predictor reads `start_frame_index`, not `start_frame_idx` (`$R/SCSam3/demoSCSam3MVOpt/SCSam3VideoPredictor.py:102`). The demo base class passes the *wrong* key (`$R/SCSam3/demoSCSam3MVOpt/SCSam3Video.py:216-221`), so the base class has always run implicit too.
- The **processing order is identical either way**: with `start_frame_idx=None`, `_get_processing_order` derives it as `min(t for t, out in enumerate(previous_stages_out) if out is not None)` (`$R/SCSam3/demoSCSam3MVOpt/SCSam3VideoInference.py:231-234`), and `add_tracker_new_mask` marks exactly the prompted view (`:1911`), so the derived value equals `view_index`.
- The real difference is the **action history**. `propagate_in_video` records `frame_idx=start_frame_idx` (`$R/SCSam3/demoSCSam3MVOpt/SCSam3VideoInference.py:1015-1021`). `"both"` means the forward pass runs first, then the backward pass re-enters `parse_action_history_for_propagation`, which contains (`:1265-1272`):

  ```python
  ) or action_history[-1]["frame_idx"] in [0, inference_state["num_frames"] - 1]:
      return "propagation_fetch", None
  ```

  With `SPATIAL_START_IMPLICIT=1` the recorded `frame_idx` is `None`, the test is false, and the backward pass is a real `propagation_partial`. With an explicit index **and a reference view equal to the last view** (`num_frames-1`, where `num_frames` = number of cameras in the cross-view pseudo-video), the backward pass degrades to `propagation_fetch`, which only reads `inference_state["cached_frame_outputs"].get(frame_idx, {})` (`:1055-1058`) — nothing was ever computed for those views, so `masks_spatial` for every view before the reference comes back empty and `TrackForward` seeds them all with the zero mask.
- This is reachable in practice: 6 of the 15 swept datasets have a `cam_list` entry landing on the last view index — Blocks (idx 9 of 10), Carpark (8/9), Fencing (9/10), MATF (9/10), Painter (15/16), PoznanStreet (8/9) — computed from `$R/SCSam3/demo/MVSeg.json`. (The symmetric `frame_idx == 0` case is harmless: the backward range `range(-1, -1, -1)` is empty anyway.)
- **Stale docstring:** `PropagateAcrossViews`' own docstring (`runMVSeg.py:200-204`) says "so the start view is spelled out here", which describes the `SPATIAL_START_IMPLICIT=0` path, i.e. the opposite of the default. Flagging as a contradiction in the source, not resolving it.

### 3. `track_cams` — `written` vs `all`

`$R/SCSam3/runMVSeg.py:304-306`:

```python
track_mode = args.track_cams or ("all" if args.algo in NEEDS_ALL_VIEWS else "written")
track_idx  = (list(range(len(cam_names))) if track_mode == "all"
              else [i for i, n in enumerate(cam_names) if n in written_cams])
```

- `cam_names` comes from `perms` (every camera), `written_cams` from `cam_list` (the annotated/scored cameras) — `:270-271`.
- `LoadCameraFolders` builds a lazy frame reader for **every** camera but opens a temporal session only for `track_idx` (`:155-174`); the cross-view pseudo-video is then `[images[i][start_frame] for i in range(numImage)]` — one decoded frame per skipped camera (`:178-182`).
- The write loop skips any tracked view not in `written_cams` (`:345-346`), so `all` costs sessions that produce no files.
- **Why `NEEDS_ALL_VIEWS` is not cosmetic:** the NewMem tracker's memory attention pulls the four *preceding views by list index* (`$R/SCSam3/demoSCSam3MVOpt/SCSam3TrackerPredictorNewMem.py:1262-1274`):

  ```python
  for s_pos in range(-4, 0):
      prev_spatial_idx = spatial_idx + s_pos
  ```
  Negative results wrap to the end of the list (plain Python indexing, no bounds guard). With `--track-cams written` the list holds only the 3 scored cameras, so each view attends to the other two, wrapped — a different model, not just a faster one. `OneStage` has no such coupling (`propagate_in_video` takes a single `session_id`), which is why its default is `written`.

### 4. Request-shape selection via `uses_spatial_predictor`

`$R/SCSam3/runMVSeg.py:251-256`:

```python
if getattr(self, "uses_spatial_predictor",
           getattr(self, "predictor_spatial", None) is not None):
    request["session_ids"] = self.session_ids
    request["spatial_idx"] = m
else:
    request["session_id"] = self.session_ids[m]
```

The two predictor classes take **mutually incompatible, non-optional** request keys — `request["session_id"]` (`$R/SCSam3/demoSCSam3MVOpt/SCSam3VideoPredictor.py:100`) vs `request["session_ids"]` + `request["spatial_idx"]` (`$R/SCSam3/demoSCSam3MVOpt/SCSam3VideoPredictorNewMem.py:100-101`) — so the wrong branch is a `KeyError`, not a silent degradation. What each package actually declares:

| Package | `predictor_spatial` in `__init__` | `uses_spatial_predictor` | `RetireSpatialPredictor` | Branch taken |
|---|---|---|---|---|
| `demoSCSam3OneStage` | **no** (only `self.predictor`) — `SCSam3Video.py:16-29` | no | **yes** (`SCSam3Video.py:31`) | single-session (`session_id`) |
| `demoSCSam3OneStageNew` | yes (`SCSam3Video.py:16`) | **no** | **no** | multi-session, via the attribute fallback |
| `demoSCSam3MVOpt` | yes (`SCSam3Video.py:16`) | **yes** (`SCSam3Video.py:23`) | **yes** (`SCSam3Video.py:37`) | multi-session, via the flag |

The fallback exists precisely because MVOpt sets `self.predictor_spatial = None` when retiring (`$R/SCSam3/demoSCSam3MVOpt/SCSam3Video.py:69`) — after retirement the attribute check would flip MVOpt into the wrong branch, hence "test the flag, not the attribute" (`runMVSeg.py:246-250` and the matching note at `demoSCSam3MVOpt/SCSam3Video.py:18-23`).

Related: the `spatial` property (`runMVSeg.py:141-144`) is `getattr(self, "predictor_spatial", self.predictor)` — for `OneStage` the cross-view pass therefore runs on the *same* predictor object as the temporal pass. After MVOpt retires, `sc.spatial` would evaluate to `None`; nothing calls it after `:321`.

### 5. The `RetireSpatialPredictor` call guard

`$R/SCSam3/runMVSeg.py:335-338`:

```python
if hasattr(sc, "RetireSpatialPredictor"):
    sc.RetireSpatialPredictor()
    torch.clear_autocast_cache()
    print("spatial model  retired", flush=True)
```

Two consequences that are not visible from the runner alone:

- **`OneStageNew` never retires anything** — it has no such method, so the ~3.2 GiB cross-view model, its N-view pseudo-video, feature cache and per-view tracker memories stay resident for the whole temporal pass. Only `OneStage` and `MVOpt` define the method.
- **`OneStage` prints `spatial model retired` while retiring nothing.** Its `RetireSpatialPredictor` starts with `predictor = getattr(self, "predictor_spatial", None); if predictor is None: return` (`$R/SCSam3/demoSCSam3OneStage/SCSam3Video.py:46-49`), and `OneStage.__init__` never sets that attribute. The log line is unconditional once the method exists.
- The paired `torch.clear_autocast_cache()` is load-bearing per the in-repo rationale (`runMVSeg.py:325-334` and `demoSCSam3OneStage/SCSam3Video.py:55-59`): the retire path drops `predictor.model` outright rather than `.cpu()`-ing it, and the autocast cache keys bf16 copies on weak refs to the now-dead fp32 parameters. Both files assert this is value-neutral (a deterministic re-`.to(bf16)` of frozen weights).

### 6. Reference-mask seeding and tensor forms

**Cross-view seed, `runMVSeg.py`** — `pick_reference` reads `Mask/<cam>/<start_frame:06d>.png` grayscale and scores cameras by `int(np.max(img))` (`:95-111`), then:

```python
for obj_id in range(1, n_obj + 1):
    sc.AddReferenceMask(ref_index, (ref_gt == obj_id).astype("float32"), obj_id)   # :317-318
```
`AddReferenceMask` wraps it as `torch.tensor(mask, dtype=torch.float32)` and posts one `add_prompt` per object at `frame_index=view_index` (`:188-196`). The shape must be exactly 2-D `(H, W)`: `SCSam3TrackerPredictor.add_new_mask` does `assert mask.dim() == 2` then `mask[None, None].float()` (`$R/SCSam3/demoSCSam3MVOpt/SCSam3TrackerPredictor.py:373-375`). `.dim()` means a **torch tensor** — a numpy array raises `AttributeError`.

**Cross-view harvest → temporal seed.** `PropagateAcrossViews` stores `out["out_binary_masks"][i] > 0.0` per `(view, obj_id)` (`:211-216`). `out_binary_masks` is a **numpy** array of shape `(N, H, W)`, bool (`$R/SCSam3/demoSCSam3MVOpt/SCSam3VideoInference.py:442,463,517`), so each stored element is already 2-D — which is why `TrackForward` re-wraps it directly with no index (`:234`) while the ForSam2 runner needs `mask[0, ...]` (below). Missing objects get a single shared `zero = torch.zeros((video_height, video_width), dtype=torch.float32)` allocated once and reused for every `(view, obj)` gap (`:225-236`); reuse is safe because `add_new_mask` copies via `.float().to(device)`.

**`runMVSegForSam2.py`** uses the SAM 2-style path and a different form (`:182-186`):

```python
seed = torch.from_numpy(((ref_gt == (i + 1)) * 255).astype(np.uint8))
sc.AddMaskSingle(ref_index, seed, i + 1)
```
because `AddMaskSingle` divides by 255 on the way in (`$R/SCSam3/demoSCSam3ForSam2New/SCSam3Video.py:109-115`, `mask=mask // 255`) and that package's `add_new_mask` carries the same `assert mask.dim() == 2` (`$R/SCSam3/demoSCSam3ForSam2New/SCSam3TrackerPredictor.py:372`). There `masks_spatial` holds `out_mask_logits[i] > 0.0`, which is `(1, H, W)` (`SCSam3Video.py:125-130`), hence `mask[0, ...]` in the replacement `RunNaiveTracking` (`runMVSegForSam2.py:159`).

### 7. Other things a reader will trip on

| Behaviour | Evidence |
|---|---|
| The reference rule is **max object id**, not object count, despite both docstrings saying "most objects"; ties go to the first camera in `cam_list` (strict `>`). Prompting then assumes ids are contiguous `1..max` — a gap yields an all-zero prompt mask. | `runMVSeg.py:95-111,317`; `runMVSegForSam2.py:71-80,184` |
| `sc.obj_ids` is **overwritten** by each `AddReferenceMask` and ends as whatever the last `add_prompt` returned; `_postprocess_output` drops zero-area masks (`keep = out_binary_masks.any(dim=(1,2))`), so the printed "prompted N objects" can be < `n_obj`. | `runMVSeg.py:196,319`; `demoSCSam3MVOpt/SCSam3VideoInference.py:464` |
| `--device` is **ignored** by all three `ALGOS` packages: `def __init__(self, device)` never uses `device`, building on `range(torch.cuda.device_count())` instead. The `device …` log line is cosmetic for `runMVSeg.py`. It *is* honoured by `runMVSegForSam2.py`. | `demoSCSam3OneStage/SCSam3Video.py:16-17`, `demoSCSam3OneStageNew/SCSam3Video.py:14-15`, `demoSCSam3MVOpt/SCSam3Video.py:14-15` vs `demoSCSam3ForSam2New/build_scsam3.py:988` |
| `--overwrite` only bypasses the emptiness check; it never deletes. Stale masks from a previous run survive for any camera/frame the new run does not rewrite. | `runMVSeg.py:294-295`; `runMVSegForSam2.py:91-92` |
| `video_height/width` are rebound per camera inside the loop and only the **last** camera's dimensions survive into the object and into the cross-view `start_session`. Mixed-resolution rigs would silently use the last camera's size. | `runMVSeg.py:161,175-176,181-182` |
| The output-writing loop advances all view generators in **lockstep, one frame at a time**, and does exactly `num_frame` rounds. `max_frame_num_to_track=num_frame` actually makes the generator willing to yield `num_frame + 1` frames (`range(start, min(start+max, N-1)+1)`), so each generator is deliberately left one frame short of exhaustion; the extra frame is never computed because the generators are lazy. | `runMVSeg.py:240-243,342-356`; `demoSCSam3MVOpt/SCSam3VideoInferenceNewMem.py:245-251` |
| The write loop's `mask.cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask)` — in practice the numpy branch is taken, since `out_binary_masks` is already numpy. | `runMVSeg.py:353`; `demoSCSam3MVOpt/SCSam3VideoInference.py:517` |
| Output layout is `<ds>/<out>/<cam>/<frame_index>/<obj_id>.png`, `uint8` 0/255, with `frame_index` the **absolute** frame number (`start_frame`-based), directory name unpadded (`f"{frame_idx:d}"`). | `runMVSeg.py:349-355`; `runMVSegForSam2.py:203-208` |
| `SCSAM3_TRIM_CACHED_OUTPUTS` (set by `runMVOptThree.sh:14`) is read **only** by the MVOpt package, at import time. Running that script with `ALGO=OneStageNew` silently drops the memory fix. | `demoSCSam3MVOpt/SCSam3VideoInferenceNewMem.py:38,529-558` |
| `runMVSegForSam2.py` hardcodes `max_frame_num_to_track=240` and tracks **every** camera in `perms`; it also runs the backward cross-view pass first when `ref_index != 0`. | `runMVSegForSam2.py:166-169,190-192` |

### 8. Ambiguities / contradictions worth recording

1. `PropagateAcrossViews`' docstring says the start view "is spelled out here" (`runMVSeg.py:200-204`) while the default `SPATIAL_START_IMPLICIT=1` omits it. The docstring describes the non-default path.
2. `NEEDS_ALL_VIEWS`' comment calls MVOpt "OneStageNew with the memory fixes" (`runMVSeg.py:44-45`), but the two differ in more than memory: MVOpt adds `uses_spatial_predictor` and `RetireSpatialPredictor`, which OneStageNew lacks entirely. The runner's own fallback comment at `:249-250` states this correctly.
3. `MVOpt` is registered in `ALGOS`/`DEFAULT_OUT`/`NEEDS_ALL_VIEWS` but is absent from `runMVSegAll.sh`'s `ALGOS` and `OUTNAME` (`runMVSegAll.sh:25-26`); it is only reachable through `runMVOptThree.sh` or a manual `--algo MVOpt`.
4. `$R/SCSam3/demo/MVSeg.json` holds 23 dataset entries (including `Blocks1`, `Frog1`, … variants); the sweep uses only the 15 named in `runMVSegAll.sh:22-24`.
5. The runner's docstring says "Only step 5 needs a GPU" (`runMVSeg.py:20`) — the cross-view propagation in step 4/5 also runs on the model; the three ALGOS packages have no CPU path at all given point 7 (`device` unused, `torch.cuda.device_count()` drives the build).