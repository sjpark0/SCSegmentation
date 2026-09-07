#!/usr/bin/env python3
"""Provenance manifests for MVSeg mask folders.

Data/ is gitignored, so until now the only record of how a mask folder such as
Data/MVSeg/Barn/SegMaskSam3MVOpt came to be was the mtime of its PNGs
(REPORT.md P13, ROADMAP Phase 1).  This module writes one MANIFEST.json per
method folder - a file at <dataset>/<method>/MANIFEST.json, next to the camera
folders, never inside them - holding a content digest of the masks plus a
provenance block (git revision, arguments, env flags, image id, source digest).

    python3 eval/manifest.py write  Data/MVSeg/Barn/SegMaskSam3MVOpt \\
                                    [--method M --dataset D --provenance k=v ... --basis TEXT --live --package DIR]
    python3 eval/manifest.py verify Data/MVSeg/Barn/SegMaskSam3MVOpt     # exit 1 on digest mismatch
    python3 eval/manifest.py show   Data/MVSeg/Barn/SegMaskSam3MVOpt
    python3 eval/manifest.py sweep  --root Data/MVSeg --methods SegMaskSam3MVOpt SegMaskSam3OneStage ... \\
                                    [--datasets ...] [--legacy] [--force] [--summary out.json]

Manifest (schema_version 1)
    schema, schema_version, dataset, method, path, written_at, written_by
    cams                sorted camera folder names
    per_cam             {cam: frame_min, frame_max, n_frames, contiguous, png_count, object_ids}
    png_count           every *.png below the folder
    content_digest      sha256 over the sorted list of (relative path, sha256 of file
                        bytes) of every *.png below the folder - deterministic, blind to
                        mtimes, blind to MANIFEST.json itself and to any non-PNG file
    mtime_min/max       of the PNGs, ISO 8601 with offset
    provenance          git_rev (a bare 40-hex commit hash, or "unknown"; any
                        qualifier goes in git_rev_note), git_dirty, algo, argv,
                        track_cams, env {SPATIAL_START_IMPLICIT, SCSAM3_TRIM_CACHED_OUTPUTS},
                        docker_image_id, source_digest (sha256 over the demo package's
                        *.py files, same construction as content_digest) - each value
                        either known or the string "unknown" - plus recorded_at_run_time
                        and provenance_basis, free text saying how the values were
                        established (recorded live by the runner vs inferred afterwards).
                        Inferred blocks may add source_digest_confidence ("high" when
                        the evidence for the package state is direct, "low" when it is
                        circumstantial, with the reasoning in
                        source_digest_confidence_note and every candidate digest in
                        source_digest_candidates)

SCSam3/runMVSeg.py calls write_run_manifest() after its final 'done -> <out_dir>'
print, so every new folder gets a live record.  `sweep --legacy` fills in the
folders that predate this module from the evidence in docs/raw/run-inventory.md,
SCSam3/logs/*, docs/experiments.md and git history (LEGACY_PROVENANCE below);
every such value is marked inferred and the basis text cites the evidence.

Host python, stdlib only.  Hashing runs in a process pool on the CLI paths and
single-threaded from write_run_manifest (no fork after CUDA is up).
"""
import argparse
import hashlib
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime
from multiprocessing import Pool

SCHEMA = "scsam3-mask-manifest"
SCHEMA_VERSION = 1
MANIFEST_NAME = "MANIFEST.json"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNKNOWN = "unknown"
ENV_FLAGS = ("SPATIAL_START_IMPLICIT", "SCSAM3_TRIM_CACHED_OUTPUTS")
PROVENANCE_KEYS = ("git_rev", "git_rev_note", "git_dirty", "algo", "argv", "track_cams",
                   "env", "docker_image_id", "source_digest")
CONTENT_DIGEST_SPEC = (
    "sha256 over the sorted list of (relative posix path, sha256 hex of file bytes) "
    "of every *.png below the folder, each entry fed as '<path>\\0<hex>\\n'; "
    "MANIFEST.json and every non-PNG file excluded; independent of mtimes")
SOURCE_DIGEST_SPEC = (
    "same construction over the *.py files below the package directory "
    "(__pycache__ excluded), paths relative to the package")


# ------------------------------------------------------------------ hashing
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_rel(job):
    root, rel = job
    return rel, sha256_file(os.path.join(root, rel))


def digest_pairs(pairs):
    """The digest of a list of (relative path, sha256 hex) - order-independent."""
    h = hashlib.sha256()
    for rel, hx in sorted(pairs):
        h.update(f"{rel}\0{hx}\n".encode("utf-8"))
    return h.hexdigest()


def _walk(folder, keep):
    out = []
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for f in filenames:
            if keep(f):
                rel = os.path.relpath(os.path.join(dirpath, f), folder)
                out.append(rel.replace(os.sep, "/"))
    return sorted(out)


def list_pngs(folder):
    return _walk(folder, lambda f: f.lower().endswith(".png"))


def hash_files(folder, rels, jobs=0):
    if not rels:
        return []
    n = jobs if jobs > 0 else min(os.cpu_count() or 1, 16)
    if n > 1 and len(rels) > 64:
        with Pool(n) as pool:
            return pool.map(_hash_rel, [(folder, r) for r in rels], chunksize=64)
    return [_hash_rel((folder, r)) for r in rels]


def content_digest(folder, jobs=0):
    """(digest, pairs) over every *.png below `folder`."""
    pairs = hash_files(folder, list_pngs(folder), jobs)
    return digest_pairs(pairs), pairs


def source_digest(package_dir):
    """sha256 over the *.py files of a demo package (SOURCE_DIGEST_SPEC)."""
    package_dir = os.path.abspath(package_dir)
    if not os.path.isdir(package_dir):
        return UNKNOWN
    rels = _walk(package_dir, lambda f: f.endswith(".py"))
    return digest_pairs(hash_files(package_dir, rels, jobs=1))


# ---------------------------------------------------------------------- git
def _git_dir(repo):
    git = os.path.join(repo, ".git")
    if os.path.isfile(git):                      # worktree / submodule pointer
        with open(git) as fh:
            line = fh.read().strip()
        if line.startswith("gitdir:"):
            git = os.path.normpath(os.path.join(repo, line[len("gitdir:"):].strip()))
    return git if os.path.isdir(git) else None


def _read(path):
    with open(path) as fh:
        return fh.read().strip()


def git_rev_from_dot_git(repo):
    """HEAD commit hash read off .git files - no git binary involved.

    HEAD -> refs/heads/<x> -> loose ref, else packed-refs (also through a
    worktree's commondir); a detached HEAD holds the hash directly.
    """
    try:
        git = _git_dir(repo)
        if git is None:
            return UNKNOWN
        head = _read(os.path.join(git, "HEAD"))
        if re.fullmatch(r"[0-9a-f]{40,64}", head):
            return head
        if not head.startswith("ref:"):
            return UNKNOWN
        ref = head[len("ref:"):].strip()
        roots = [git]
        common = os.path.join(git, "commondir")
        if os.path.isfile(common):
            roots.append(os.path.normpath(os.path.join(git, _read(common))))
        for root in roots:
            loose = os.path.join(root, ref)
            if os.path.isfile(loose):
                sha = _read(loose)
                if re.fullmatch(r"[0-9a-f]{40,64}", sha):
                    return sha
        for root in roots:
            packed = os.path.join(root, "packed-refs")
            if not os.path.isfile(packed):
                continue
            with open(packed) as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line[0] in "#^":
                        continue
                    sha, _, name = line.partition(" ")
                    if name == ref:
                        return sha
        return UNKNOWN
    except Exception:
        return UNKNOWN


def git_dirty(repo):
    """'true' / 'false' if tracked files differ from HEAD, 'unknown' without git.

    Runs `git status` against a *copy* of the index so a root process inside
    the container can never rewrite the user's .git/index; untracked files do
    not count.
    """
    if shutil.which("git") is None:
        return UNKNOWN
    try:
        git = _git_dir(repo)
        if git is None:
            return UNKNOWN
        with tempfile.TemporaryDirectory() as td:
            env = dict(os.environ)
            index = os.path.join(git, "index")
            if os.path.isfile(index):
                tmp_index = os.path.join(td, "index")
                shutil.copyfile(index, tmp_index)
                env["GIT_INDEX_FILE"] = tmp_index
            r = subprocess.run(
                ["git", "-c", "safe.directory=*", "-C", repo, "status",
                 "--porcelain", "--untracked-files=no"],
                capture_output=True, text=True, timeout=120, env=env)
        if r.returncode != 0:
            return UNKNOWN
        return "true" if r.stdout.strip() else "false"
    except Exception:
        return UNKNOWN


def git_resolve(rev, repo=REPO):
    """Full hash for a short hash / tag, or the input unchanged without git."""
    if shutil.which("git") is None:
        return rev
    try:
        r = subprocess.run(["git", "-c", "safe.directory=*", "-C", repo, "rev-parse",
                            "--verify", "--quiet", f"{rev}^{{commit}}"],
                           capture_output=True, text=True, timeout=60)
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else rev
    except Exception:
        return rev


_SOURCE_AT = {}


def git_source_digest(commit, package_relpath, repo=REPO):
    """source_digest() of a package as committed, via git archive into a temp dir.

    Never checks anything out; the working tree is untouched.
    """
    key = (commit, package_relpath)
    if key in _SOURCE_AT:
        return _SOURCE_AT[key]
    result = UNKNOWN
    if shutil.which("git") is None:
        print(f"warning: git not found; source_digest of {package_relpath}@{commit} "
              f"is 'unknown'", file=sys.stderr)
    else:
        try:
            r = subprocess.run(["git", "-c", "safe.directory=*", "-C", repo, "archive",
                                "--format=tar", commit, package_relpath],
                               capture_output=True, timeout=300)
            if r.returncode != 0:
                print(f"warning: git archive {commit} {package_relpath} failed "
                      f"({r.stderr.decode(errors='replace').strip()}); source_digest "
                      f"is 'unknown'", file=sys.stderr)
            else:
                with tempfile.TemporaryDirectory() as td:
                    with tarfile.open(fileobj=io.BytesIO(r.stdout)) as tar:
                        try:
                            tar.extractall(td, filter="data")
                        except TypeError:      # `filter=` needs Python 3.12
                            print("warning: tarfile.extractall(filter=) unsupported "
                                  f"on python {sys.version.split()[0]}; extracting "
                                  "the git archive without a filter", file=sys.stderr)
                            tar.extractall(td)
                    result = source_digest(os.path.join(td, package_relpath))
        except Exception as e:
            print(f"warning: source_digest of {package_relpath}@{commit} failed "
                  f"({type(e).__name__}: {e}); 'unknown'", file=sys.stderr)
            result = UNKNOWN
    _SOURCE_AT[key] = result
    return result


# ----------------------------------------------------------------- manifest
def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _iso(ts):
    return datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


def scan_folder(folder, rels):
    """Camera / frame / object layout of the PNG list, plus the mtime range."""
    per_cam, stray, mtimes = {}, [], []
    for rel in rels:
        mtimes.append(os.stat(os.path.join(folder, rel)).st_mtime)
        parts = rel.split("/")
        stem = parts[-1][:-4]
        if len(parts) == 3 and parts[1].isdigit() and stem.isdigit():
            c = per_cam.setdefault(parts[0], {"frames": set(), "objects": set(), "n": 0})
            c["frames"].add(int(parts[1]))
            c["objects"].add(int(stem))
            c["n"] += 1
        else:
            stray.append(rel)
    out = {}
    for cam in sorted(per_cam):
        fr = sorted(per_cam[cam]["frames"])
        out[cam] = {
            "frame_min": fr[0], "frame_max": fr[-1], "n_frames": len(fr),
            "contiguous": fr == list(range(fr[0], fr[-1] + 1)),
            "png_count": per_cam[cam]["n"],
            "object_ids": sorted(per_cam[cam]["objects"]),
        }
    return {
        "cams": sorted(out), "per_cam": out, "unexpected_layout": stray,
        "mtime_min": _iso(min(mtimes)) if mtimes else None,
        "mtime_max": _iso(max(mtimes)) if mtimes else None,
    }


def default_provenance():
    p = {k: UNKNOWN for k in PROVENANCE_KEYS}
    p["env"] = {k: UNKNOWN for k in ENV_FLAGS}
    p["recorded_at_run_time"] = False
    p["provenance_basis"] = UNKNOWN
    return p


def normalise_provenance(provenance):
    """Every required key present; unknowns spelled 'unknown'."""
    p = default_provenance()
    for k, v in (provenance or {}).items():
        if k == "env" and isinstance(v, dict):
            p["env"].update(v)
        else:
            p[k] = v
    for k in PROVENANCE_KEYS:
        if p[k] is None or p[k] == "":
            p[k] = UNKNOWN
    for k in ENV_FLAGS:
        if p["env"].get(k) in (None, ""):
            p["env"][k] = UNKNOWN
    return p


def build_manifest(folder, dataset=None, method=None, provenance=None, jobs=0):
    folder = os.path.abspath(folder)
    method = method or os.path.basename(folder)
    dataset = dataset or os.path.basename(os.path.dirname(folder))
    rels = list_pngs(folder)
    pairs = hash_files(folder, rels, jobs)
    layout = scan_folder(folder, rels)
    m = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "method": method,
        "path": folder,
        "written_at": now_iso(),
        "written_by": f"eval/manifest.py schema {SCHEMA_VERSION}, python {sys.version.split()[0]}",
        "cams": layout["cams"],
        "per_cam": layout["per_cam"],
        "png_count": len(pairs),
        "unexpected_layout": layout["unexpected_layout"],
        "content_digest": digest_pairs(pairs),
        "content_digest_spec": CONTENT_DIGEST_SPEC,
        "source_digest_spec": SOURCE_DIGEST_SPEC,
        "mtime_min": layout["mtime_min"],
        "mtime_max": layout["mtime_max"],
        "provenance": normalise_provenance(provenance),
    }
    return m


def manifest_path(folder):
    return os.path.join(os.path.abspath(folder), MANIFEST_NAME)


def read_manifest(folder):
    with open(manifest_path(folder)) as fh:
        return json.load(fh)


def write_manifest(folder, dataset=None, method=None, provenance=None, jobs=0):
    """Write <folder>/MANIFEST.json atomically; returns the manifest dict."""
    m = build_manifest(folder, dataset, method, provenance, jobs)
    path = manifest_path(folder)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(m, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, path)
    return m


def verify_manifest(folder, jobs=0):
    """(ok, stored manifest, recomputed {content_digest, png_count})."""
    m = read_manifest(folder)
    digest, pairs = content_digest(folder, jobs)
    fresh = {"content_digest": digest, "png_count": len(pairs)}
    ok = (m.get("content_digest") == digest and m.get("png_count") == len(pairs))
    return ok, m, fresh


# --------------------------------------------------------- live provenance
def live_provenance(repo=REPO, package_dir=None, algo=None, argv=None,
                    track_cams=None, extra=None):
    """Provenance for a run that is happening right now, in this process."""
    p = {
        "recorded_at_run_time": True,
        "git_rev": git_rev_from_dot_git(repo),
        "git_rev_note": "HEAD parsed from the repo's .git files by the process that wrote the masks",
        "git_dirty": git_dirty(repo),
        "algo": algo or UNKNOWN,
        "argv": list(sys.argv if argv is None else argv),
        "track_cams": track_cams or UNKNOWN,
        "env": {k: os.environ.get(k, "unset") for k in ENV_FLAGS},
        "docker_image_id": os.environ.get("SCSAM3_IMAGE_ID") or UNKNOWN,
        "source_digest": source_digest(package_dir) if package_dir else UNKNOWN,
        "source_package": (os.path.relpath(os.path.abspath(package_dir), repo)
                           if package_dir else UNKNOWN),
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "python": sys.version.split()[0],
        "provenance_basis": (
            "recorded at run time in the process that wrote the masks: git_rev "
            "parsed from the repo's .git files (no git call), git_dirty from "
            "`git status --porcelain --untracked-files=no` on a copy of the index "
            "('unknown' when git is unavailable), env flags and SCSAM3_IMAGE_ID "
            "from the process environment, source_digest over the package's *.py "
            "files as they were on disk when the run started"),
    }
    if extra:
        p.update(extra)
    return p


def write_run_manifest(out_dir, dataset, method, algo, package_dir, track_cams,
                       repo=REPO, torch=None, extra=None):
    """What SCSam3/runMVSeg.py calls after 'done -> <out_dir>'.

    Single-threaded hashing (the caller has CUDA up; no fork).  Raises on
    failure - the caller decides that a manifest must never fail a run.
    """
    more = {"out_name": method}
    if torch is not None:
        more["torch"] = getattr(torch, "__version__", UNKNOWN)
        more["cuda"] = getattr(getattr(torch, "version", None), "cuda", None) or UNKNOWN
    if extra:
        more.update(extra)
    prov = live_provenance(repo=repo, package_dir=package_dir, algo=algo,
                           track_cams=track_cams, extra=more)
    write_manifest(out_dir, dataset=dataset, method=method, provenance=prov, jobs=1)
    return manifest_path(out_dir)


# ------------------------------------------------ inferred (legacy) provenance
# Folders written before this module existed.  Everything here was inferred on
# 2026-09-07 from PNG mtimes, SCSam3/logs/*, docs/raw/run-inventory.md,
# docs/experiments.md and git history; each basis string cites its evidence.
# Host clock is KST (+09:00); the container logs print UTC.
COMMIT_SIBLINGS = "99c05c7"      # tag baseline-siblings: HEAD during every 2026-09-03/04 run
COMMIT_MVOPT = "3232d3e"         # tag baseline-onestagenew: first commit of demoSCSam3MVOpt
                                 # and of the memory-patched demoSCSam3OneStage
HEAVY3 = ("AlexaMeadeExhibit", "CoffeeMartini", "FlameSteak")
ONESTAGENEW_104557 = ("AlexaMeadeFacePaint", "Barn", "Blocks", "Breakfast", "Carpark")
ONESTAGENEW_114432 = ("Dog", "Fencing", "Frog", "MATF", "Painter", "PoznanStreet", "Welder")
MAIN15 = HEAVY3 + ONESTAGENEW_104557 + ONESTAGENEW_114432

REBUILT_IMAGE = ("scsam3:latest sha256:9064e3dc8548300a0472417a60a8ac230b4b54766d77a00d0d2f5f26a087b3c1 "
                 "(2026-09-04 14:13)")
IMAGE_NOTE_MVOPT = (
    "docker image: the scsam3 image of 2026-02-25T13:26:46+09:00 (docs/raw/env-docker.md), "
    "deleted 2026-09-07 (ROADMAP Phase 0); its id was never recorded. The rebuilt image "
    f"{REBUILT_IMAGE} reproduced all 15 SegMaskSam3MVOpt folders byte for byte "
    "(SCSam3/logs/newimg-full.log: TOTAL 19815 PNG, 0 differences).")
IMAGE_NOTE_OTHER = (
    "docker image: the scsam3 image of 2026-02-25T13:26:46+09:00 (docs/raw/env-docker.md), "
    "deleted 2026-09-07 (ROADMAP Phase 0); image id never recorded. The MVOpt reproduction "
    f"on the rebuilt image {REBUILT_IMAGE} (SCSam3/logs/newimg-full.log) is evidence for "
    "MVOpt only; nothing was re-run for this folder on that image.")
SPATIAL_NOTE = ("SPATIAL_START_IMPLICIT unset = default 1: no script sets it "
                "(docs/raw/env-docker.md:119).")
GIT_REV_NOTE = (
    f"HEAD at run time, inferred from history topology ({COMMIT_MVOPT}'s parent is "
    f"{COMMIT_SIBLINGS}, single branch): every pre-manifest run (2026-09-03/04) predates "
    f"{COMMIT_MVOPT} (2026-09-04 11:11) and no commit lies between the two, so the checked-out "
    f"commit was {COMMIT_SIBLINGS}; the working tree carried uncommitted changes, see basis.")


def _legacy_common(algo, package, commit, track_cams, log, script, run_date, argv,
                   trim, basis, git_dirty_value=UNKNOWN, confidence="high",
                   confidence_note=None, candidates=None):
    p = {
        "recorded_at_run_time": False,
        "inferred": True,
        "git_rev": git_resolve(COMMIT_SIBLINGS),
        "git_rev_note": GIT_REV_NOTE,
        "git_dirty": git_dirty_value,
        "algo": algo,
        "argv": argv,
        "track_cams": track_cams,
        "env": {"SPATIAL_START_IMPLICIT": "unset (default 1)",
                "SCSAM3_TRIM_CACHED_OUTPUTS": trim},
        "docker_image_id": UNKNOWN,
        "source_package": package,
        "source_commit": git_resolve(commit),
        "source_digest": git_source_digest(git_resolve(commit), package),
        "source_digest_confidence": confidence,
        "script": script,
        "log": log,
        "run_date": run_date,
        "provenance_basis": basis,
    }
    if confidence_note:
        p["source_digest_confidence_note"] = confidence_note
    if candidates:
        p["source_digest_candidates"] = candidates
    return p


def legacy_provenance(dataset, method):
    """Inferred provenance for the pre-manifest folders, or None."""
    ds = dataset
    if method == "SegMaskSam3OneStage":
        pkg = "SCSam3/demoSCSam3OneStage"
        candidates = {c: git_source_digest(git_resolve(c), pkg)
                      for c in (COMMIT_SIBLINGS, COMMIT_MVOPT)}
        cand_txt = ", ".join(f"{c} ({'unpatched' if c == COMMIT_SIBLINGS else 'patched'}) = {d}"
                             for c, d in candidates.items())
        painter = ""
        if ds == "Painter":
            painter = (" Painter only: 14 files v6/47..60/17.png (mtime 09:16) survive from "
                       "the earlier sweep mvseg-20260903-085851 because --overwrite does "
                       "not clear the folder; they are 0-px masks on 0-px GT, score-neutral "
                       "(docs/investigations-closed.md #1) and absent from _recheck.")
        confidence_note = (
            f"low. source_digest is the {COMMIT_SIBLINGS} (unpatched) package, chosen because "
            f"the 09-03 logs lack the 'spatial model  retired' line that runMVSeg.py prints "
            f"when the package has RetireSpatialPredictor. That line is printed by "
            f"runMVSeg.py itself, which was untracked until {COMMIT_MVOPT}, so whether the "
            f"09-03 runMVSeg.py printed it at all is unknown: its absence cannot separate "
            f"'package unpatched' from 'runner did not print it yet'. The 09-03 working tree "
            f"was uncommitted either way. Candidate digests: {cand_txt}.")
        basis = (
            f"inferred 2026-09-07. PNG mtimes 2026-09-03 10:46-11:05 KST match "
            f"SCSam3/logs/mvseg-20260903-104557/OneStage-{ds}.log (banner 'algorithm "
            f"OneStage (demoSCSam3OneStage)', 'output .../SegMaskSam3OneStage', '3 tracked "
            f"[written]', 'done ->'); driver log sweep-full-104557.log = runMVSegAll.sh, "
            f"which at the time passed no --memory cap and no SCSAM3_TRIM_CACHED_OUTPUTS "
            f"(docs/raw/env-docker.md:25). --overwrite inferred: the folder already existed "
            f"from mvseg-20260903-085851 and was rewritten. Code state: demoSCSam3OneStage "
            f"PRESUMABLY before the sibling memory patch - the 09-03 logs lack the 'spatial "
            f"model  retired' line that the 09-04 logs of the patched package "
            f"(RetireSpatialPredictor, first in {COMMIT_MVOPT}) carry; but that line is "
            f"printed by runMVSeg.py, untracked until {COMMIT_MVOPT}, so the absence is weak "
            f"evidence (source_digest_confidence low). Two candidate package states, both "
            f"digested: {cand_txt}; source_digest records the first. Nearest committed "
            f"unpatched state is tag baseline-siblings = {COMMIT_SIBLINGS} (ROADMAP Phase 0); "
            f"the 09-03 working tree was uncommitted, so equality with either commit is "
            f"assumed, not proven. {IMAGE_NOTE_OTHER} {SPATIAL_NOTE}{painter}")
        return _legacy_common(
            "OneStage", pkg, COMMIT_SIBLINGS, "written",
            f"SCSam3/logs/mvseg-20260903-104557/OneStage-{ds}.log (driver sweep-full-104557.log)",
            "SCSam3/runMVSegAll.sh", "2026-09-03 10:46-11:05 KST",
            ["python", "runMVSeg.py", ds, "--algo", "OneStage", "--overwrite"],
            "unset", basis, confidence="low", confidence_note=confidence_note,
            candidates=candidates)

    if method == "SegMaskSam3OneStageNew":
        if ds in ONESTAGENEW_104557:
            log = f"SCSam3/logs/mvseg-20260903-104557/OneStageNew-{ds}.log (driver sweep-full-104557.log)"
            when = "2026-09-03 11:10-11:26 KST"
            argv = ["python", "runMVSeg.py", ds, "--algo", "OneStageNew", "--overwrite"]
            ow = ("--overwrite inferred: the folder already existed from "
                  "mvseg-20260903-085851 and was rewritten (diff -rq against the "
                  "fresh MVOpt folder is empty, so no stragglers).")
        elif ds in ONESTAGENEW_114432:
            log = f"SCSam3/logs/mvseg-20260903-114432/OneStageNew-{ds}.log (driver sweepN-114432.log)"
            when = "2026-09-03 11:46-12:09 KST"
            argv = ["python", "runMVSeg.py", ds, "--algo", "OneStageNew"]
            ow = ("--overwrite not known (no earlier folder existed: both earlier "
                  "sweeps died at CoffeeMartini before reaching this dataset).")
        else:
            return None
        basis = (
            f"inferred 2026-09-07. PNG mtimes ({when}) match {log} (banner 'algorithm "
            f"OneStageNew (demoSCSam3OneStageNew)', 'output .../SegMaskSam3OneStageNew', "
            f"'tracked [all]', 'done ->'); runMVSegAll.sh: no --memory cap, no "
            f"SCSAM3_TRIM_CACHED_OUTPUTS (the original package does not read it anyway). "
            f"{ow} Code state: the ORIGINAL OneStageNew package = tag baseline-onestagenew "
            f"({COMMIT_MVOPT}:SCSam3/demoSCSam3OneStageNew); its *.py are identical to "
            f"{COMMIT_SIBLINGS} (git diff --stat lists only .dockerignore), so both commits "
            f"that bracket the run give the same source_digest, and the 09-03 logs lack "
            f"'spatial model  retired', consistent with a package without "
            f"RetireSpatialPredictor. Since af39315 (2026-09-04 11:57) the folder "
            f"demoSCSam3OneStageNew holds MVOpt's code instead (FROZEN.md); the masks here "
            f"are byte-identical to SegMaskSam3MVOpt (docs/raw/output-inventory.md, "
            f"docs/experiments.md (1)). {IMAGE_NOTE_OTHER} {SPATIAL_NOTE}")
        return _legacy_common(
            "OneStageNew", "SCSam3/demoSCSam3OneStageNew", COMMIT_MVOPT, "all",
            log, "SCSam3/runMVSegAll.sh", when, argv, "unset", basis,
            confidence_note=(f"high: the package's *.py are identical at {COMMIT_SIBLINGS} "
                             f"and {COMMIT_MVOPT}, the two commits bracketing the run, so the "
                             f"digest does not depend on which one was checked out."))

    if method == "SegMaskSam3MVOpt":
        if ds in HEAVY3:
            log = f"SCSam3/logs/mvopt-20260904-010657/{ds}.log (driver mvopt-run-010657.log)"
            when = "2026-09-04 01:09-01:44 KST"
        elif ds in MAIN15:
            log = f"SCSam3/logs/mvopt-20260904-060418/{ds}.log (driver mvopt-rest-060418.log)"
            when = "2026-09-04 06:05-06:46 KST"
        else:
            return None
        basis = (
            f"inferred 2026-09-07. PNG mtimes ({when}) match {log} (banner 'algorithm MVOpt "
            f"(demoSCSam3MVOpt)', 'output .../SegMaskSam3MVOpt', 'tracked [all]', 'spatial "
            f"model  retired', 'done ->'); runMVOptThree.sh = docker run --memory=90g "
            f"--memory-swap=90g -e SCSAM3_TRIM_CACHED_OUTPUTS=1 ... python runMVSeg.py <ds> "
            f"--algo MVOpt --overwrite. Code state: demoSCSam3MVOpt as first committed in "
            f"{COMMIT_MVOPT} (2026-09-04 11:11, after the run; identical to HEAD - git diff "
            f"--stat {COMMIT_MVOPT}..HEAD is empty). Equality with the uncommitted package "
            f"that ran is inferred from SCSam3/logs/newimg-full.log: on 2026-09-04 14:50, "
            f"with the committed code, all 15 datasets were re-run and matched these "
            f"folders byte for byte (19815 PNG, 0 differences), and from {COMMIT_MVOPT}'s "
            f"commit message citing this run's equivalence numbers. {IMAGE_NOTE_MVOPT} "
            f"{SPATIAL_NOTE}")
        return _legacy_common(
            "MVOpt", "SCSam3/demoSCSam3MVOpt", COMMIT_MVOPT, "all", log,
            "SCSam3/runMVOptThree.sh", when,
            ["python", "runMVSeg.py", ds, "--algo", "MVOpt", "--overwrite"], "1", basis,
            confidence_note=(f"high: the committed {COMMIT_MVOPT} package re-run on the rebuilt "
                             f"image reproduced this folder byte for byte "
                             f"(SCSam3/logs/newimg-full.log)."))

    if method == "SegMaskSam3OneStage_recheck":
        if ds not in MAIN15:
            return None
        log = f"SCSam3/logs/mvopt-20260904-083026/{ds}.log (driver onestage-recheck-083026.log)"
        basis = (
            f"inferred 2026-09-07. PNG mtimes (2026-09-04 08:31-08:50 KST) match {log} "
            f"(banner 'algorithm OneStage (demoSCSam3OneStage)', 'output "
            f".../SegMaskSam3OneStage_recheck', '3 tracked [written]', 'spatial model  "
            f"retired', 'done ->'); invoked as ALGO=OneStage OUT=SegMaskSam3OneStage_recheck "
            f"./runMVOptThree.sh (docs/raw/sweep_2.md:86), i.e. --memory=90g "
            f"--memory-swap=90g and SCSAM3_TRIM_CACHED_OUTPUTS=1 (which the OneStage "
            f"package does not read). Purpose and code state: the 15-dataset verification "
            f"of the sibling memory patch named in {COMMIT_MVOPT}'s commit message ('OneStage "
            f"15개 재실행으로 검증: 14개 바이트 동일, Painter는 점수 동일'); the 'spatial "
            f"model  retired' line proves RetireSpatialPredictor was present, which "
            f"{COMMIT_SIBLINGS} lacks and {COMMIT_MVOPT} has, so the package is the PATCHED "
            f"demoSCSam3OneStage = {COMMIT_MVOPT} (identical to HEAD). Never scored "
            f"(docs/experiments.md). {IMAGE_NOTE_OTHER} {SPATIAL_NOTE}")
        return _legacy_common(
            "OneStage", "SCSam3/demoSCSam3OneStage", COMMIT_MVOPT, "written", log,
            "SCSam3/runMVOptThree.sh", "2026-09-04 08:31-08:50 KST",
            ["python", "runMVSeg.py", ds, "--algo", "OneStage", "--out",
             "SegMaskSam3OneStage_recheck", "--overwrite"], "1", basis,
            git_dirty_value="true (tracked demoSCSam3OneStage/*.py were patched in the "
                            "working tree, committed 2.5 h later as " + COMMIT_MVOPT + ")",
            confidence_note=("high: the 'spatial model  retired' line in every log proves "
                             "RetireSpatialPredictor was present, i.e. the patched package "
                             f"committed 2.5 h later as {COMMIT_MVOPT}."))

    if method in ("SegMask", "SegMask1", "SegMaskNew", "SegMaskNew1", "SegMaskNew2",
                  "SegMaskNew3"):
        p = default_provenance()
        p["inferred"] = True
        p["algo"] = "unknown (SAM 2 baseline)"
        p["env"] = {k: "not applicable (SAM 2 scripts read neither)" for k in ENV_FLAGS}
        p["provenance_basis"] = (
            "inferred 2026-09-07, almost nothing is known. SAM 2 result written in 2025 "
            "(PNG mtimes: SegMaskNew1 2025-10-15..11-11, SegMask1 2025-11-07..11-11, "
            "SegMaskNew2 2025-11-21, SegMaskNew3 2025-11-21); no run log exists under "
            "SCSam3/logs or SCSam2. The writer family is SCSam2/demo/"
            "sam2_demoVideoNew_maskSingleInputMVSeg.py (its current copy hard-codes "
            "SegMaskNew3 at :73,80; earlier states of the same script presumably wrote "
            "SegMaskNew1/2), run from a container with the SAM 2 image; the exact "
            "script state, arguments and settings of the cited column SegMaskNew1 are an "
            "open item (ROADMAP Phase 4, docs/experiments.md).")
        return p
    return None


# --------------------------------------------------------------------- CLI
def _parse_kv(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            sys.exit(f"--provenance expects key=value, got {it!r}")
        k, v = it.split("=", 1)
        if v[:1] in "[{":
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                pass
        if k.startswith("env."):
            out.setdefault("env", {})[k[4:]] = v
        else:
            out[k] = v
    return out


def cmd_write(args):
    prov = {}
    if args.live:
        prov = live_provenance(package_dir=args.package, algo=args.algo,
                               track_cams=args.track_cams)
    elif args.package:
        prov["source_digest"] = source_digest(args.package)
        prov["source_package"] = os.path.relpath(os.path.abspath(args.package), REPO)
        if args.algo:
            prov["algo"] = args.algo
        if args.track_cams:
            prov["track_cams"] = args.track_cams
    prov.update(_parse_kv(args.provenance))
    if args.basis:
        prov["provenance_basis"] = args.basis
    m = write_manifest(args.folder, args.dataset, args.method, prov, args.jobs)
    print(f"wrote {manifest_path(args.folder)}: {m['png_count']} PNG, "
          f"{len(m['cams'])} cams, digest {m['content_digest']}")
    return 0


def cmd_verify(args):
    if not os.path.isfile(manifest_path(args.folder)):
        print(f"no {MANIFEST_NAME} in {args.folder}")
        return 2
    ok, m, fresh = verify_manifest(args.folder, args.jobs)
    print(f"{'OK      ' if ok else 'MISMATCH'} {args.folder}")
    print(f"  stored     {m.get('content_digest')}  ({m.get('png_count')} PNG, "
          f"written {m.get('written_at')})")
    print(f"  recomputed {fresh['content_digest']}  ({fresh['png_count']} PNG)")
    return 0 if ok else 1


def cmd_show(args):
    m = read_manifest(args.folder)
    json.dump(m, sys.stdout, indent=2, ensure_ascii=False)
    print()
    return 0


def cmd_sweep(args):
    root = os.path.abspath(args.root)
    datasets = args.datasets or sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d, "Mask")))
    rows, kept = [], 0
    for ds in datasets:
        for method in args.methods:
            folder = os.path.join(root, ds, method)
            if not os.path.isdir(folder):
                continue
            prov = legacy_provenance(ds, method) if args.legacy else None
            if prov is None:
                prov = default_provenance()
                prov["provenance_basis"] = (
                    f"written by `manifest.py sweep --legacy`: no legacy table entry for "
                    f"{method}; nothing inferred" if args.legacy else
                    "written by `manifest.py sweep` without --legacy; nothing inferred")
            existing = None
            if os.path.isfile(manifest_path(folder)):
                try:
                    existing = read_manifest(folder)
                except (OSError, ValueError):
                    existing = None
            m = build_manifest(folder, ds, method, prov, args.jobs)
            note = ""
            if (existing and not args.force
                    and existing.get("provenance", {}).get("recorded_at_run_time")
                    and existing.get("content_digest") == m["content_digest"]):
                m["provenance"] = existing["provenance"]     # keep the live record
                note = "  (kept live provenance)"
                kept += 1
            path = manifest_path(folder)
            tmp = path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(m, fh, indent=2, ensure_ascii=False)
                fh.write("\n")
            os.replace(tmp, path)
            rows.append({"dataset": ds, "method": method, "png_count": m["png_count"],
                         "cams": m["cams"], "content_digest": m["content_digest"],
                         "mtime_min": m["mtime_min"], "mtime_max": m["mtime_max"],
                         "source_digest": m["provenance"].get("source_digest", UNKNOWN),
                         "log": m["provenance"].get("log", UNKNOWN)})
            print(f"{ds:<20s} {method:<28s} {m['png_count']:>6d} PNG  "
                  f"{m['content_digest']}{note}", flush=True)
    print(f"{len(rows)} manifests written" + (f", {kept} live provenance kept" if kept else ""))
    if args.summary:
        with open(args.summary, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"summary -> {args.summary}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("write", help="write <folder>/MANIFEST.json")
    w.add_argument("folder")
    w.add_argument("--method", default=None, help="default: folder basename")
    w.add_argument("--dataset", default=None, help="default: parent folder basename")
    w.add_argument("--provenance", nargs="*", metavar="KEY=VALUE",
                   help="provenance fields; env.X=... for env flags; JSON for lists")
    w.add_argument("--basis", default=None, help="provenance_basis text")
    w.add_argument("--live", action="store_true",
                   help="record git rev/dirty, env flags, image id, host from this process")
    w.add_argument("--package", default=None, help="demo package dir for source_digest")
    w.add_argument("--algo", default=None)
    w.add_argument("--track-cams", default=None)
    w.add_argument("--jobs", type=int, default=0)
    w.set_defaults(fn=cmd_write)

    v = sub.add_parser("verify", help="recompute the content digest; exit 1 on mismatch")
    v.add_argument("folder")
    v.add_argument("--jobs", type=int, default=0)
    v.set_defaults(fn=cmd_verify)

    s = sub.add_parser("show", help="print the manifest")
    s.add_argument("folder")
    s.set_defaults(fn=cmd_show)

    p = sub.add_parser("sweep", help="write manifests for every <dataset>/<method> that exists")
    p.add_argument("--root", default=os.path.join(REPO, "Data", "MVSeg"))
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--datasets", nargs="*", default=None,
                   help="default: every folder under --root with a Mask/ subfolder")
    p.add_argument("--legacy", action="store_true",
                   help="fill provenance from the inferred table for pre-manifest folders")
    p.add_argument("--force", action="store_true",
                   help="replace a live-recorded provenance block too")
    p.add_argument("--summary", default=None, help="write a JSON summary of the sweep here")
    p.add_argument("--jobs", type=int, default=0)
    p.set_defaults(fn=cmd_sweep)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
