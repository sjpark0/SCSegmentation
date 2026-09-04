## Execution Environment

### (a) Image build and run

**Build context / image contents** — `SCSam3/Dockerfile`

| Item | Value | Source |
|---|---|---|
| Base image | `pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel` | `SCSam3/Dockerfile:2` |
| System deps | git, cmake, ninja, boost, eigen, ceres, qt5, ffmpeg, opencv, imagemagick, python3-tk, … | `SCSam3/Dockerfile:7` |
| COLMAP | cloned from GitHub, `tags/3.11.0`, built with Ninja, `ninja install` | `SCSam3/Dockerfile:10-19` |
| Python env | venv at `/opt/venv`, put first on `PATH` | `SCSam3/Dockerfile:21-23` |
| Pinned pips | `numpy==1.26.4`, `setuptools==69.5.1` | `SCSam3/Dockerfile:26, 32` |
| Only `COPY` in the file | `COPY ./sam3 /opt/sam3` | `SCSam3/Dockerfile:27` |
| SAM 3 install | `pip install -e .`, `-e ".[notebooks]"`, `-e ".[train,dev]"` | `SCSam3/Dockerfile:29-31` |
| Arch list | `ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"` | `SCSam3/Dockerfile:41` |

Note: the experiment code (`runMVSeg.py`, `demoSCSam3*/`, `Data/MVSeg/`) is **not** baked into the image — the only `COPY` is `./sam3`. All demo/eval code reaches the container through the `-v /:/host` bind mount at run time.

**Run invocations (verbatim from the scripts)**

| Script | Invocation | Purpose |
|---|---|---|
| `SCSam3/launch_container.sh:1-2` | `xhost +local:docker` then `docker run --shm-size=64g --gpus all -it --rm --env DISPLAY=$DISPLAY --env QT_X11_NO_MITSHM=1 --volume /tmp/.X11-unix:/tmp/.X11-unix --volume /:/host --workdir /host$PWD scsam3` | interactive shell with X11 forwarding; **no** `--memory` cap |
| `SCSam3/runMVSegAll.sh:58-61` | `docker run --rm --gpus all --shm-size=32g -v /:/host -w "${WORKDIR}" "${IMAGE}" python runMVSeg.py "${ds}" --algo "${algo}" "${EXTRA[@]}"` | full sweep; **no** `--memory` cap |
| `SCSam3/runForSam2Three.sh:13-16` | `docker run --rm --gpus all --shm-size=32g --memory=90g --memory-swap=90g -v /:/host -w "/host${ROOT}/SCSam3" scsam3 python runMVSegForSam2.py "${ds}" --overwrite` | 3 stuck datasets, ForSam2New variant |
| `SCSam3/runMVOptThree.sh:13-17` | `docker run --rm --gpus all --shm-size=32g --memory=90g --memory-swap=90g -e SCSAM3_TRIM_CACHED_OUTPUTS=1 -v /:/host -w "/host${ROOT}/SCSam3" scsam3 python runMVSeg.py "${ds}" --algo "${ALGO:-MVOpt}" ${OUT:+--out "${OUT}"} --overwrite` | 3 stuck datasets, memory-fixed package |
| `runMVSegAll.sh:75-76, 93-97`, `runForSam2Three.sh:23`, `runMVOptThree.sh:24` | `docker run --rm -v /:/host <image> chown -R "$(id -u):$(id -g)" "/host${ROOT}/Data/MVSeg"` (and an `eval_jf.py` container with `-w /host${ROOT}/Data/MVSeg`) | fix ownership of container-written outputs; score J&F |

**Mount pattern:** every script bind-mounts the **host root** `/` at `/host` and sets `-w /host<abs path>`, so container paths mirror host paths under a `/host` prefix. `launch_container.sh` additionally mounts `/tmp/.X11-unix`.

**Why the memory cap.** Stated in the script headers, not inferred:
- `runForSam2Three.sh:3-4` — "`--memory` caps the container so a runaway is OOM-killed on its own instead of taking the host down, which is how the last two attempts ended."
- `runMVOptThree.sh:2-3` — "`--memory` caps the container: host RAM exhaustion is what rebooted the machine twice, and the cap turns that into a contained failure."

`--memory-swap=90g` equal to `--memory=90g` means the container gets **zero swap**, so the cgroup OOM killer fires at 90 GiB RSS rather than spilling into the host's 8 GiB swap. Host cgroup v2 (`cgroup2fs`, `docker info` → Cgroup Version: 2), so the cap is enforceable.

**Isolation rationale** (`runMVSegAll.sh:10-14`): one container per dataset "so that one failure or OOM does not take the rest down, and so GPU memory is fully released in between — the 45-camera scenes hold a session per camera." `waitAndRunMVSeg.sh:9-12` runs a Frog smoke test first (13 cameras, 5 objects) before committing to the overnight sweep, and polls `nvidia-smi --query-gpu=memory.free` until `FREE_MIB` (default 40000) is free (`waitAndRunMVSeg.sh:21, 31-52`).

**Scripts with no docker involvement:** `SCSam3/demo/*.sh` (`MVSeg.sh`, `MVSeg1.sh`, `MVSeg_ObjSelect.sh`, `IoU.sh`, `IoU1.sh`, `IoU_Ojbsel.sh`, `MOTA.sh`, `MOTA1.sh`) are flat `NAME=<dataset>; python <script>.py $NAME` loops over the 15 MVSeg datasets, intended to be run *inside* an already-started container. They invoke SAM 2-named scripts (`sam2_demoVideoNew_maskSingleInputMVSeg.py`, `ComputeIOU*.py`, `ComputeMOTA*.py`), all present in `SCSam3/demo/`.

**`.dockerignore` — and a caveat.** Only two exist in the repo:
- `SCSam3/demoSCSam3OneStageNew/.dockerignore` — contents are exactly two lines: `*` and `!*.py`.
- `SCSam2/sam2/demo/frontend/.dockerignore` — a standard Node/Vite ignore list (logs, `node_modules`, `dist`, `.env`, editor dirs, playwright/coverage output).

**Caveat to flag:** Docker only honours a `.dockerignore` at the **build context root**. The documented build context is `SCSam3/` (`README.md:16`), and `SCSam3/.dockerignore` does not exist (verified: `ls` → No such file). So `SCSam3/demoSCSam3OneStageNew/.dockerignore` has **no effect** on the `scsam3` build. It is also moot in practice, since the Dockerfile's only `COPY` is `./sam3`.

---

### (b) HuggingFace credential handling — BuildKit secret (working copy) vs. build ARG (HEAD)

**These two disagree, and that is the single most important fact here.**

| Version | Mechanism | Lines |
|---|---|---|
| **Working copy** (`SCSam3/Dockerfile`, shown as ` M` by `git status`) | **BuildKit secret mount.** `RUN --mount=type=secret,id=hf_token \` then `HF_TOKEN="$(cat /run/secrets/hf_token)" hf download facebook/sam3 --token "${HF_TOKEN}"` | `SCSam3/Dockerfile:38-39` (comment at `:35-37`) |
| **Committed HEAD** (`git show HEAD:SCSam3/Dockerfile`) | **Build ARG.** `ARG HF_TOKEN` followed by `RUN hf download facebook/sam3 --token ${HF_TOKEN}` | HEAD lines 36-37 |

Machine-checked: HEAD has 1 `^ARG ` line and 0 `mount=type=secret` lines; the working copy has 0 `ARG` and 1 secret mount.

**Exact build command a user must run** (identical in both places, so this is unambiguous):

```bash
export HF_TOKEN=...            # shell only, never written to a file
DOCKER_BUILDKIT=1 docker build --secret id=hf_token,env=HF_TOKEN -t scsam3 SCSam3/
```
— `README.md:14-16`; the Dockerfile's own comment (`SCSam3/Dockerfile:37`) gives the same command with `.` as the context, i.e. run from inside `SCSam3/`.

BuildKit is available: `docker buildx version` → `v0.37.0`, Docker server `29.7.2`.

**Contradictions / open items to resolve, stated rather than resolved:**
1. The **locally present image `scsam3:latest` predates the fix.** It was created `2026-02-25T13:26:46+09:00` (`docker image inspect`), and `docker history --no-trunc scsam3:latest` has **3 layers** matching `HF_TOKEN|hf download` and **0** matching `mount=type=secret` — i.e. the running image was built from the ARG version, and the token is recorded in its layer history. (Values not printed here.) Every `docker run … scsam3` in the scripts uses this stale image; the secret-mount Dockerfile has not been built yet.
2. `docs/analysis/REPORT.md:123` (finding G1) describes the ARG form at `SCSam3/Dockerfile:35-36` plus a token in the working-copy `README.md:3`, and reports a **second, different** token baked into 3 image layers. The working copy has since been changed: current `README.md` contains no token-shaped string (`grep -cE 'hf_[A-Za-z0-9]{20,}' README.md` → 0; the only `hf_` occurrence is the literal secret id `hf_token` on line 16), and the Dockerfile now uses the secret mount. So **the source-level part of G1 appears fixed in the working tree but is not committed, and the image-level part is not fixed at all.** REPORT.md's line references (`:35-36`) match HEAD, not the working copy.
3. Neither the Dockerfile nor any script passes a token at **run** time; the checkpoint is expected to already be in the image's HF cache.

---

### (c) Observed hardware

| Property | Observed value | Command |
|---|---|---|
| GPU count | 1 | `nvidia-smi -L` |
| GPU model | NVIDIA RTX 6000 Ada Generation | `nvidia-smi --query-gpu=name` |
| Total VRAM | 49140 MiB (~48.0 GiB) | `nvidia-smi --query-gpu=memory.total` |
| Free VRAM at time of reading | 47573 MiB (944 MiB in use by Xorg/gnome-shell/firefox, no compute processes) | `nvidia-smi` |
| Compute capability | 8.9 | `nvidia-smi --query-gpu=compute_cap` |
| Driver / CUDA | 580.173.02 / CUDA 13.0 | `nvidia-smi` |
| Host RAM total | 125 GiB (`free -g`), 125.5 GiB per `docker info` | `free -g`, `free -h` |
| Host RAM available at reading | 112 GiB available, 13 GiB used, 45 GiB buff/cache | `free -g` |
| Swap | 8.0 GiB total, ~0 used | `free -h` |
| CPU threads | 28 | `nproc` |
| cgroup | v2 (`cgroup2fs`), systemd driver | `stat -fc %T /sys/fs/cgroup/`, `docker info` |
| GPU passthrough | `nvidia-container-runtime-hook`, `nvidia-ctk`, `nvidia-container-cli` all in `/usr/bin` (Docker's `Default Runtime` is `runc`, so `--gpus all` goes through the hook, not a custom runtime) | `which`, `docker info` |

Consistency notes: `--memory=90g` is ~72% of the 125 GiB host, leaving ~35 GiB headroom — consistent with the "contained failure instead of host reboot" rationale. `--shm-size=32g` (sweep) and `=64g` (interactive) both fit under that cap; `TORCH_CUDA_ARCH_LIST` (`Dockerfile:41`) includes `8.6;8.9`, so the RTX 6000 Ada (cc 8.9) is covered.

---

### (d) `PYTORCH_CUDA_ALLOC_CONF` and related env vars

Every occurrence repo-wide (`grep -rn "PYTORCH_CUDA_ALLOC_CONF\|expandable_segments\|max_split_size_mb\|PYTORCH_NO_CUDA_MEMORY_CACHING"`):

| File:line | Statement | Semantics |
|---|---|---|
| `SCSam3/runMVSeg.py:33` | `os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")` | `setdefault` — an externally exported value wins |
| `SCSam3/runMVSegForSam2.py:23` | `os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")` | same |
| `SCSam3/demo/sam2_demoVideoNew_maskSingleInputMVSeg.py:13` | `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"` | hard assignment — overrides the environment |
| `SCSam2/demo/sam2_demoVideoNew_maskSingleInputMVSeg.py:13` | `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"` | hard assignment |
| `SCSam2/demo/sam2_MVSeg_recheck.py:13` | `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"` | hard assignment |
| `docs/analysis/SCHEDULE.md:280` | prose reference | see contradiction below |

No `max_split_size_mb`, no `PYTORCH_NO_CUDA_MEMORY_CACHING`, and **no shell script or `docker run` sets `PYTORCH_CUDA_ALLOC_CONF`** — it is set only in Python, always to `expandable_segments:True`, always before torch is imported (it is at the top of the import block in each file).

Other env vars that shape execution:

| Var | Where set / read | Effect |
|---|---|---|
| `SCSAM3_TRIM_CACHED_OUTPUTS` | **set** to `1` by `SCSam3/runMVOptThree.sh:14` (`-e`); **read** at `SCSam3/demoSCSam3MVOpt/SCSam3VideoInferenceNewMem.py:38` as `os.environ.get(..., "").strip() not in ("", "0")`, gating `_trim_cached_frame_outputs` (`:546-555`) | drops consumed frames from `cached_frame_outputs` during partial propagation; off by default |
| `SPATIAL_START_IMPLICIT` | `SCSam3/runMVSeg.py:37`, `os.environ.get("SPATIAL_START_IMPLICIT", "1") == "1"` | defaults **on**; no script sets it |
| `ALGO`, `OUT`, `IMAGE`, `NOT_BEFORE`, `GIVE_UP`, `FREE_MIB`, `POLL` | `runMVOptThree.sh:16`, `runMVSegAll.sh:19`, `waitAndRunMVSeg.sh:19-22` | script-level overrides, all with defaults |
| `TOKENIZERS_PARALLELISM=false`, `HYDRA_FULL_ERROR=1`, `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` | `SCSam3/sam3/sam3/model/tokenizer_ve.py:27`, `sam3/train/train.py:23`, `sam3/train/utils/train_utils.py:73` | upstream SAM 3 vendored code, not project scripts |
| `DEBIAN_FRONTEND`, `PATH`, `TORCH_CUDA_ARCH_LIST` | `SCSam3/Dockerfile:5, 23, 41` | image-level |
| `DISPLAY`, `QT_X11_NO_MITSHM=1` | `SCSam3/launch_container.sh:2` | X11 forwarding for the interactive container |

**Two documentation-vs-code contradictions worth recording rather than silently resolving:**
1. `docs/analysis/SCHEDULE.md:280` cites the allocator setting as `runMVSeg.py:32`; the statement is actually on **line 33** of `SCSam3/runMVSeg.py`. Off by one.
2. `docs/analysis/SCHEDULE.md:316` claims the trim flag is read as `int(os.environ.get("SCSAM3_TRIM_CACHED_OUTPUTS", "0"))` and therefore raises `ValueError` on an empty-string export. The code at `SCSam3/demoSCSam3MVOpt/SCSam3VideoInferenceNewMem.py:38` does **not** do that — it uses `.strip() not in ("", "0")`, which handles the empty string. Either the doc describes a version that has since been changed, or it describes a different file; I did not find an `int(...)` form of this flag anywhere in the tree.