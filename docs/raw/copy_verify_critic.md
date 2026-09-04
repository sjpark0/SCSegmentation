## 1. The risk class none of the five looked at: **live process / runtime state**

All five reasoned about files at rest. Nobody ran `ps`. **A job is executing against the copied files right now.**

```
1196056 sjpark 11:11:49  bash -c ... ALGO=OneStageNew OUT=SegMaskSam3OneStageNew_verify ./runMVOptThree.sh Welder AlexaMeadeExhibit CoffeeMartini FlameSteak
1208433 sjpark 11:19:58  docker run --rm --gpus all --memory=90g -e SCSAM3_TRIM_CACHED_OUTPUTS=1 \
                           -v /:/host ... scsam3 python runMVSeg.py AlexaMeadeExhibit --algo OneStageNew \
                           --out SegMaskSam3OneStageNew_verify --overwrite
1208515 root   11:19:59  python runMVSeg.py AlexaMeadeExhibit --algo OneStageNew ...
```

Timing, from `stat -c '%z'`: the six files' **ctime is `2026-09-04 11:11:21`** (mtime is MVOpt's preserved `01:00`/`01:02`). The job launched at `11:11:49` — **28 seconds after the copy**. So the copy was not mid-run, but everything since is running patched code under the name OneStageNew.

**(a) The running job already destroyed the only physical evidence of real OneStageNew behavior.**

`SCSam3/logs/onestagenew-verify-101913.log` (pre-copy run, 10:19–11:05) ends:

```
Welder               FAILED(137)   388s
DIFFER  Welder                   677 PNG  차이 18건
TOTAL 12428 PNG, 차이 18건
!!! 차이 있음 !!!
```

`docs/investigations-closed.md:200` records that same run as `| **OneStageNew 신규 재실행 vs MVOpt 저장본** (2026-09-04) | PNG 12,428장 | **0** |`. The count matches; the verdict does not. The log says 18, the doc says 0. Welder was SIGKILLed (137) after 677 of 941 PNGs, so the 18 are plausibly missing-path lines rather than content differences — but the artifact that would settle it is gone:

```
/Data/MVSeg/Welder/SegMaskSam3OneStageNew_verify  total=941 pre-copy=0 post-copy=941
```

Every one of the 941 PNGs now has mtime > 11:11. The 677-file pre-copy output was overwritten by the running job at ~11:19. `runMVSeg.py` has no `rmtree` — only `os.makedirs(..., exist_ok=True)` at `runMVSeg.py:350` — so `--overwrite` overwrites in place and leaves stragglers, which is how a 677-file dir became a 941-file dir with no trace of the earlier run.

**(b) The running job is a tautology.** `args.algo` is used at only five places: `runMVSeg.py:116` (`algo_dir = ALGOS[algo]`), `:265` (out name), `:284` (print), `:300`, `:304` (`NEEDS_ALL_VIEWS`, which contains both `OneStageNew` and `MVOpt` per `:45`). With all 12 shared `.py` byte-identical, `--algo OneStageNew` and `--algo MVOpt` execute the same bytes. It is burning ~90 GB RAM and a GPU to re-test GPU determinism, not code equivalence — and it cannot answer "does the frozen baseline still reproduce," because no OneStageNew code exists on disk to ask.

**(c) Ownership.** The container process runs as **root** (PID 1208515). Live proof: `AlexaMeadeExhibit/SegMaskSam3OneStageNew_verify` is `root:root`. `find /home/sjpark/Documents/SCSegmentation -xdev \! -user sjpark | wc -l` → **10514**. The chowns at `runMVOptThree.sh:23` and `runMVSegAll.sh:73-74` cover **only `Data/MVSeg`** and run only if the loop completes — a Ctrl-C leaves root-owned output. Uncovered, and confirmed root-owned:

```
drwxr-xr-x 2 root root 4096 Mar  9 12:30 SCSam3/demoSCSam3OneStageNew/0
drwxr-xr-x 2 root root 4096 Mar  9 12:31 SCSam3/demoSCSam3OneStageNew/31
drwxr-xr-x 2 root root 4096 Sep  3 19:33 SCSam3/__pycache__
```

The 13 GB the task says "had to survive" is root-owned. That is accidental protection from `sjpark`, and zero protection from any `docker run -v /:/host` — which is every runner in this repo.

**(d) `__pycache__` — I checked, and it is clean.** All 13 `.pyc` in `demoSCSam3OneStageNew/__pycache__` carry `flags=0` (timestamp invalidation), magic `cb0d0d0a` matching CPython 3.12, and header `(mtime,size)` exactly equal to their source's — `VALID=True` for all 13, including the six copied files (e.g. `SCSam3VideoInferenceNewMem.py pyc(mt=1788451325,sz=99150) src(mt=1788451325,sz=99150)`). No shadowing. Worth knowing *why* it is safe: CPython compares source mtime for **equality**, not "newer than," so `cp -p` moving a destination's mtime backward does not fool it. Tools that do use newer-than (make, `rsync -u`, editor external-change detection) will see these six files as 10 hours old.

One process-hygiene note the code-equivalence verifier reported but did not classify as a risk: its `py_compile` ran inside a container **at 11:14/11:16, while the Welder job was live** (`Welder.log` mtime 11:19). Writing into a running job's package directory. It was safe here only because `.pyc` writes are atomic renames and Welder had already imported.

**(e) Silent drift** — see Q2.

## 2. Mechanism keeping the freeze true: **there is none**

Everything I checked came back empty:

- `ls -la .git/hooks/ | grep -v '\.sample'` → only `.` and `..`. No active hooks.
- `find .github -type f` → `copilot-instructions.md` only. No CI.
- `.pre-commit-config.yaml` → `No such file or directory`.
- `grep -rl 'demoSCSam3OneStageNew' --include='*.sh' --include='*.yml' --include='Makefile' --include='*.toml'` → **no matches**. Not one script, config, or Makefile mentions the folder.
- No manifest, hash file, `FROZEN` marker, or README stub inside `demoSCSam3OneStageNew/`.

The only thing that mentions it is `runMVSeg.py:38-40`, which maps `"OneStageNew": "demoSCSam3OneStageNew"` — i.e. the folder is a **live execution target**, not a backup.

How the freeze fails silently:

1. **No image pin.** `SCSam3/Dockerfile` `COPY ./sam3 /opt/sam3` and nothing else; the demo packages arrive purely through `-v /:/host`. Whatever is on disk at `docker run` time is what runs. There is no baked artifact to fall back to. (`demoSCSam3OneStageNew/.dockerignore` is inert — Docker honors only the context-root `.dockerignore`, and the context is `SCSam3/`.)
2. **`--algo OneStageNew` still resolves and still runs.** Nothing errors, warns, or refuses. The currently-running job is the proof.
3. **The freeze is unenforceable in the other direction too.** Any edit to MVOpt silently makes the two folders diverge, and the only signal is a `diff` nobody is scheduled to run.
4. **Outputs are invisible to git** (`.gitignore:183` `*.png`), so no `git status` will ever show that a mask directory was rewritten. `--overwrite` plus no `rmtree` means a rewrite is also invisible on disk except by mtime.
5. **`3232d3e` is unpushed** — `git reflog` confirms it is `HEAD@{0}`, one commit ahead of `origin/main` at `99c05c7`. It is the only surviving copy of real OneStageNew source. I verified it is genuinely the pre-patch code: `git show 3232d3e:SCSam3/demoSCSam3OneStageNew/SCSam3Video.py | grep -c 'uses_spatial_predictor\|RetireSpatialPredictor'` → **0**. One `.git` loss and the baseline source is gone permanently, since the PNGs it produced were never tracked either.

## 3. Cheapest checks that materially raise confidence

**First — stop the running job.** It is a tautology that is actively overwriting `SegMaskSam3OneStageNew_verify` and writing root-owned files:

```
kill 1196056 1196057 && docker stop d86a0c9b2aeb
```

**Then, the single highest-value command — push the only copy of the baseline (seconds, no GPU):**

```
git -C /home/sjpark/Documents/SCSegmentation push origin main
```

**Then the drift check.** This is the whole freeze contract in one line; it exits 0 today (I ran it):

```
diff -rq --exclude=__pycache__ -x '*copy.py' -x '.dockerignore' -x '[0-9]' -x '[0-9][0-9]' \
  /home/sjpark/Documents/SCSegmentation/SCSam3/demoSCSam3OneStageNew \
  /home/sjpark/Documents/SCSegmentation/SCSam3/demoSCSam3MVOpt
```

All 12 shared `.py` share sha256 (e.g. `SCSam3Video.py` → `9db2a57cfd58215cc954b1f93a058c6fe40b72ddf075de74d82f96b9855ee7bc` in both). Wire it into `.git/hooks/pre-commit`; it is the only thing that will ever notice drift.

**Finally, make the 13 GB checkable** — it is git-invisible, root-owned, and has no integrity record, so silent corruption is currently undetectable:

```
cd /home/sjpark/Documents/SCSegmentation/SCSam3/demoSCSam3OneStageNew && \
  find [0-9] [0-9][0-9] -name '*.png' | sort | xargs sha256sum > MANIFEST.sha256
```

(~1600 files; under a minute. Commit `MANIFEST.sha256` — it is not matched by `.gitignore:183`.)

One documentation fix that costs nothing: `docs/investigations-closed.md:200` says **0** differences for a run whose own log (`SCSam3/logs/onestagenew-verify-101913.log`) says **차이 18건** and `!!! 차이 있음 !!!`. Correct the line or record why the 18 are benign — the output that would prove it is already overwritten.