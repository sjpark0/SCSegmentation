"""Opt-in helpers for very long stage-2 runs (the full CROOM take: 32 cameras x 9 038 frames at 3840x2160).
Every helper is output-neutral by construction and is only installed when its runMVSeg.py flag is given
(--mem-prune, --io-cache2, --async-png N, --crop-stream); with no flag nothing here runs.

MemPrune     the tracker keeps one memory entry per (session, object, frame) forever (~0.3 GiB of host RAM per frame
             for 32 cameras).  Memory selection (sam3_tracker_base.frame_filter) only ever returns the K most recent
             entries whose eff_iou_score > mf_threshold, plus frame t-1; object pointers use the same list; mode C
             reads a neighbour session only at t and t-1.  After a session yields frame t, every non-conditioning
             entry f < t-1 that is not among those K is therefore unreachable and is dropped.  A guard aborts the run if
             either premise breaks: an entry it kept disappears, or a kept entry's eff_iou_score changes.
install_io_cache2  a 2-entry cache around the lazy frame loaders: the detector prefetches t+1, so the 1-entry caches
             decode every 4K JPEG twice.
AsyncWriter  cv2.imwrite on a thread pool behind a bounded queue; drain() before anything reads the PNGs back.
LazyCrops    the crop pass's frames decoded and cropped on demand (crop_track.read_frames holds every frame).
"""
import os
import threading
from concurrent.futures import ThreadPoolExecutor


class PruneGuardError(RuntimeError):
    pass


class MemPrune:
    def __init__(self, tracker, num_frames):
        self.K = min(int(num_frames), int(tracker.max_obj_ptrs_in_encoder)) - 1
        self.thr = float(tracker.mf_threshold)
        if not bool(tracker.use_memory_selection):
            raise PruneGuardError("--mem-prune assumes memory selection (use_memory_selection) is on")
        self.pruned = 0
        self.checked = 0

    @staticmethod
    def states(predictor, sid):
        return predictor._ALL_INFERENCE_STATES[sid]["state"]["tracker_inference_states"]

    def prune_session(self, predictor, sid, t):
        for ts in self.states(predictor, sid):
            self.prune_state(ts, int(t))

    def prune_state(self, ts, t):
        nc = ts["output_dict"]["non_cond_frame_outputs"]
        g = ts.setdefault("_long_run_prune", {"kept": {}, "last_t": None})
        # guard: since the last prune at t' the old entries f < t'-1 must be exactly what was kept, scores unchanged
        if g["last_t"] is not None:
            for f, score in g["kept"].items():
                e = nc.get(f)
                if e is None:
                    raise PruneGuardError(f"memory entry {f} vanished after the prune at t={g['last_t']} (now t={t})")
                if _score(e) != score:
                    raise PruneGuardError(f"eff_iou_score of entry {f} changed after it was kept (t={t})")
                self.checked += 1
        frames = sorted(nc)
        valid = []
        for f in reversed(frames):
            if f >= t - 1:
                continue
            s = _score(nc[f])
            if s is not None and s > self.thr:
                valid.append(f)
                if len(valid) >= self.K:
                    break
        keep = set(valid)
        for f in frames:
            if f < t - 1 and f not in keep:
                del nc[f]
                for d in ts["output_dict_per_obj"].values():
                    d["non_cond_frame_outputs"].pop(f, None)
                self.pruned += 1
        g["kept"] = {f: _score(nc[f]) for f in nc if f < t - 1}
        g["last_t"] = t

    def record(self):
        return {"K": self.K, "mf_threshold": self.thr, "entries_pruned": self.pruned, "guard_checks": self.checked}


def _score(e):
    s = e.get("eff_iou_score")
    if s is None:
        return None
    return float(s.float().max()) if hasattr(s, "float") else float(s)


def install_io_cache2(io_utils):
    """Patch AsyncVideoFrameLoader / AsyncVideoFrameCPUToGPU.__getitem__ with a 2-entry cache (idempotent)."""
    for cls in (io_utils.AsyncVideoFrameLoader, io_utils.AsyncVideoFrameCPUToGPU):
        if getattr(cls, "_long_run_cache2", False):
            continue
        orig = cls.__getitem__

        def getitem(self, index, _orig=orig):
            c = self.__dict__.setdefault("_c2", {})
            if index in c:
                return c[index]
            r = _orig(self, index)
            c[index] = r
            while len(c) > 2:
                c.pop(next(iter(c)))
            return r
        cls.__getitem__ = getitem
        cls._long_run_cache2 = True


class AsyncWriter:
    """write(path, array) queues cv2.imwrite(path, array) (the caller passes a fresh array); drain() waits for all."""

    def __init__(self, cv2, threads, depth=256):
        self.cv2 = cv2
        self.pool = ThreadPoolExecutor(int(threads))
        self.sem = threading.BoundedSemaphore(depth)
        self.depth = depth
        self.lock = threading.Lock()
        self.errors = []
        self.n = 0

    def _job(self, path, array):
        try:
            if not self.cv2.imwrite(path, array):
                raise IOError(f"could not write {path}")
        except Exception as exc:          # noqa: BLE001 -- re-raised by drain / the next write
            with self.lock:
                self.errors.append(exc)
        finally:
            self.sem.release()

    def write(self, path, array):
        if self.errors:
            raise self.errors[0]
        self.sem.acquire()
        self.pool.submit(self._job, path, array)
        self.n += 1

    def drain(self):
        for _ in range(self.depth):
            self.sem.acquire()
        for _ in range(self.depth):
            self.sem.release()
        if self.errors:
            raise self.errors[0]

    def close(self):
        self.drain()
        self.pool.shutdown(wait=True)


class LazyCrops:
    """crops[k] = video frame start_frame + k (<k:06d>.jpg) cropped to the square window, decoded on demand."""

    def __init__(self, cv2, np, video_dir, start_frame, num_frame, cx0, cy0, side, hw):
        self.cv2, self.np = cv2, np
        self.video_dir, self.start, self.n = video_dir, int(start_frame), int(num_frame)
        self.cx0, self.cy0, self.side, self.hw = int(cx0), int(cy0), int(side), tuple(hw)
        self._c = {}

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        if k in self._c:
            return self._c[k]
        if not 0 <= k < self.n:
            raise IndexError(k)
        p = os.path.join(self.video_dir, f"{self.start + k:06d}.jpg")
        img = self.cv2.imread(p)
        if img is None:
            raise FileNotFoundError(p)
        if img.shape[:2] != self.hw:
            raise ValueError(f"{p}: video frame is {img.shape[1]}x{img.shape[0]}, not {self.hw[1]}x{self.hw[0]}")
        c = self.np.ascontiguousarray(img[self.cy0:self.cy0 + self.side, self.cx0:self.cx0 + self.side])
        self._c[k] = c
        while len(self._c) > 2:
            self._c.pop(next(iter(self._c)))
        return c
