SWEEP RESULT — complete inventory of differences outside the six-file .py diff.

## 1. Files present in one tree only

| Path | Side | Git | Notes |
|---|---|---|---|
| `/home/sjpark/Documents/SCSegmentation/SCSam3/demoSCSam3OneStageNew/.dockerignore` | OLD only | untracked | 8 bytes: `*\n!*.py` |
| `/home/sjpark/Documents/SCSegmentation/SCSam3/demoSCSam3OneStageNew/SCSam3TrackerPredictorNewMem copy.py` | OLD only | **tracked** | 97418 B backup |
| `.../demoSCSam3OneStageNew/{0..31}/*.png` — 32 dirs, 1600 files, **13 GB** | OLD only | gitignored (`*.png`) | `root:root` |
| `.../demoSCSam3MVOpt/__pycache__/{sam3_demoVideo,test}.cpython-312.pyc` | NEW only | gitignored | trivial |

`.dockerignore` is **functionally inert**. The Docker build context root is `/home/sjpark/Documents/SCSegmentation/SCSam3/` (where the Dockerfile lives), and that directory has **no** `.dockerignore`. Docker only honors `.dockerignore` at context root; one nested a level down is never read. The Dockerfile's only copy is `COPY ./sam3 /opt/sam3` — the demo packages are never baked into the image, they reach the container through `-v /:/host`. So this file has no effect on anything today.

`SCSam3TrackerPredictorNewMem copy.py` is a **pre-fix backup**, not dead-identical: it differs from the live `SCSam3TrackerPredictorNewMem.py` by exactly one line — it lacks `s_pos_and_prevs.append((s_pos, None))` (live file line 1257). That line *does* survive into MVOpt (line 1268). The file cannot be imported (space in the name) and nothing references it.

The 32 numbered dirs are renders written by `sam3_demoVideo.py` (**byte-identical in both trees**) via `cv2.imwrite(f"{spatial_idx}/{frame_idx}.png")`, relative to cwd. Every sibling demo package (`demoSCSam3OneStage`, `TwoStage`, `TwoStageNew`, `ForSam2`, `ForSam2New`) has the same 32×50 layout. MVOpt has none because it was never run that way.

## 2. Modes, ownership, links

- **No symlinks, no hardlinks (all nlink=1), no ACLs, no xattrs, no exec-bit differences anywhere in either tree.** Only one hidden file exists in either tree (`.dockerignore` above).
- All 12 shared top-level `.py` are `-rw-rw-r-- sjpark:sjpark` on both sides. Identical modes.
- **Ownership split:** `demoSCSam3OneStageNew/__pycache__` is `drwxr-xr-x root:root`; `demoSCSam3MVOpt/__pycache__` is `drwxrwxr-x sjpark:sjpark`. The numbered dirs are `root:root 755`. Verified: `sjpark` **cannot write** `OneStageNew/__pycache__` or `OneStageNew/0`; the package root itself is writable.

## 3. .pyc — no shadowing risk, but a path surprise

All 22 `.pyc` are **timestamp-invalidation mode** (`flags=0`), and every one currently validates (mtime+size match its source). Python re-checks mtime+size on every import, so **a stale `.pyc` cannot shadow a source change here.** There are no hash-unchecked pycs, no `.pyc` lacking a `.py`, no orphan cache dirs.

Six pycs differ *despite byte-identical sources* — that is purely the embedded `co_filename`:

```
OLD misc.cpython-312.pyc  co_filename = /host/home/sjpark/.../demoSCSam3OneStageNew/misc.py
NEW misc.cpython-312.pyc  co_filename = demoSCSam3MVOpt/misc.py
```

The 56-byte size delta is exactly the path-length delta. This confirms OLD's cache was compiled **inside the container as root**, NEW's on the host. Consequence: if the pycs are copied with mtimes preserved (`rsync -a`, `cp -a`), they stay valid and get used, so tracebacks from `demoSCSam3OneStageNew` will name `demoSCSam3MVOpt/...`. Cosmetic, but misleading during debugging.

## 4. Measured impact of the copy (`rsync -an --delete` dry run)

```
1634 deletions, 20 sends
deletions: 1600 .png + 32 numbered dirs + 'SCSam3TrackerPredictorNewMem copy.py' + '.dockerignore'
sends:     6 changed .py + 12 .pyc + 2 dirs
```

## What actually matters, ranked

1. **13 GB of gitignored renders are destroyed by any `--delete` or replace-the-folder copy.** They are not in git and not recoverable except by re-running the demo. Note the trap: as `sjpark` this *fails* with EACCES (the dirs are root-owned), which looks like a safe no-op — but this project runs everything as root in the container (`runMVOptThree.sh`, `launch_container.sh`), and there the deletion succeeds silently.

2. **Provenance collision — the most serious non-obvious consequence.** `/home/sjpark/Documents/SCSegmentation/SCSam3/runMVSeg.py` selects the package by directory and names the output by the *flag*, independently:
   - `ALGOS = {"OneStageNew": "demoSCSam3OneStageNew", "MVOpt": "demoSCSam3MVOpt"}`
   - `DEFAULT_OUT = {"OneStageNew": "SegMaskSam3OneStageNew", "MVOpt": "SegMaskSam3MVOpt"}`
   - `--algo` **defaults to `OneStageNew`**

   After the copy, `python runMVSeg.py <ds>` runs MVOpt code and writes it into `SegMaskSam3OneStageNew`. `Data/MVSeg` already holds 12 `SegMaskSam3OneStageNew` result dirs (the published baseline) and 15 `SegMaskSam3MVOpt`. One `--overwrite` run silently replaces baseline masks with MVOpt-computed ones under the baseline's name — and after the copy **no directory on disk still contains OneStageNew code**, so the baseline becomes unreproducible. Keep `demoSCSam3MVOpt` in place (do not move/delete it) or `--algo MVOpt` and `runMVOptThree.sh` (`ALGO:-MVOpt`) break outright.

3. **An env-var behavior gate that exists only in MVOpt.** `SCSAM3_TRIM_CACHED_OUTPUTS` is read at import in `/home/sjpark/Documents/SCSegmentation/SCSam3/demoSCSam3MVOpt/SCSam3VideoInferenceNewMem.py:38` and gates behavior at line 546; `runMVOptThree.sh:14` sets it to 1. OneStageNew has no such switch. Copying gives the baseline package a code path whose behavior depends on the environment it is launched from — so "same code, same results" is only true once this variable is pinned. Flagging it here because it is a configuration surface, not just a line in the diff.

4. **The root-owned `__pycache__` will break a user-level copy partway.** `cp -r` reports EACCES on the 10 pycs and *keeps going*, leaving a tree whose `.py` are new and whose cache is unwritable. Harmless for correctness (timestamp invalidation forces in-memory recompile) but it produces alarming errors and a confusing half-state.

5. **Low impact:** deleting `copy.py` is a real git deletion of a tracked file (do it deliberately, not as copy fallout); deleting `.dockerignore` loses an untracked file permanently but costs nothing since it is inert.

## Recommended copy

Copy only the six changed `.py`; do not use `--delete`; do not copy `__pycache__`; resolve the naming question in item 2 first.

```
rsync -v --no-perms --no-owner --no-group \
  SCSam3TrackerPredictor.py SCSam3TrackerPredictorNewMem.py SCSam3Video.py \
  SCSam3VideoInference.py SCSam3VideoInferenceNewMem.py io_utils.py \
  /home/sjpark/Documents/SCSegmentation/SCSam3/demoSCSam3MVOpt/ \
  /home/sjpark/Documents/SCSegmentation/SCSam3/demoSCSam3OneStageNew/
```

The other six shared `.py` (`SCSam3VideoPredictor.py`, `SCSam3VideoPredictorNewMem.py`, `build_scsam3.py`, `misc.py`, `sam3_demoVideo.py`, `test.py`) are md5-identical and need no action.