"""Shared paths and loaders for the SCSegmentation CPU test suite (SPEC.md section 4).

Run inside the scsam3 container for the full suite (torch + sam3 on CPU), or on the
host for the torch-free subset; every torch-dependent module skips itself via
pytest.importorskip.
"""
import importlib.util
import os
import sys

sys.dont_write_bytecode = True   # host runs must not leave __pycache__ in the repo

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, "SCSam3", "demoSCSam3MVOpt")
sys.path.insert(0, PKG)              # xview_gather, build_scsam3, SCSam3TrackerPredictorNewMem, io_utils
# never put SCSam3/ itself on sys.path: it shadows the installed `sam3` as a namespace package


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_runmvseg():
    return _load("runMVSeg", os.path.join(REPO, "SCSam3", "runMVSeg.py"))


def load_report_jf():
    return _load("report_jf", os.path.join(REPO, "eval", "report_jf.py"))


@pytest.fixture
def cpu_tensors(monkeypatch):
    """The tracker calls .cuda() / .pin_memory() on stored memories; on CPU both are identity."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self, *a, **k: self)
    return torch
