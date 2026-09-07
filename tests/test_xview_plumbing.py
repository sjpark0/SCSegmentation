"""T3: the two constructor kwargs are threaded through all five hops, the env helpers
parse strictly, and the frozen OneStage SCSam3Video.__init__(self, device) contract the
runner relies on (it passes no kwargs for non-XW runs) is intact."""
import ast
import inspect
import os

import pytest

from conftest import PKG, REPO

NAMES = ("cross_view_window", "cross_view_hygiene")


def _init_args(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "SCSam3Video")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    return init.args


def test_frozen_onestage_signature_ast():
    a = _init_args(os.path.join(REPO, "SCSam3", "demoSCSam3OneStage", "SCSam3Video.py"))
    assert [x.arg for x in a.args] == ["self", "device"]
    assert not a.kwonlyargs and a.vararg is None and a.kwarg is None


def test_mvopt_signature_ast():
    a = _init_args(os.path.join(PKG, "SCSam3Video.py"))
    assert [x.arg for x in a.args] == ["self", "device", *NAMES]
    assert len(a.defaults) == 2 and all(d.value is None for d in a.defaults)


def test_signatures_thread_cross_view():
    pytest.importorskip("torch")
    pytest.importorskip("sam3")
    import build_scsam3
    import SCSam3TrackerPredictorNewMem as T
    import SCSam3Video
    import SCSam3VideoPredictorNewMem as P
    hops = [SCSam3Video.SCSam3Video.__init__, P.SCSam3VideoPredictorNewMem.__init__,
            build_scsam3.build_scsam3_video_model_newmem, build_scsam3.build_tracker_newmem,
            T.SCSam3TrackerPredictorNewMem.__init__]
    for fn in hops:
        sig = inspect.signature(fn)
        for name in NAMES:
            assert name in sig.parameters, (fn, name)
            assert sig.parameters[name].default is None, (fn, name)
    # tracker: bound by name BEFORE **kwargs, so Sam3TrackerBase.__init__ never sees them
    ps = list(inspect.signature(T.SCSam3TrackerPredictorNewMem.__init__).parameters)
    assert ps.index("cross_view_window") < ps.index("kwargs")
    assert inspect.signature(T.SCSam3TrackerPredictorNewMem.__init__).parameters["kwargs"].kind \
        is inspect.Parameter.VAR_KEYWORD
    # the constructor bound (W=7 aliases the cond row) lives in the pure module
    with pytest.raises(ValueError):
        T.resolve_cross_view(7, True, 7)


def test_env_parsing():
    pytest.importorskip("torch")
    pytest.importorskip("sam3")
    from build_scsam3 import xview_env_window, xview_env_hygiene, XVIEW_WINDOW_ENV, XVIEW_HYGIENE_ENV
    assert (XVIEW_WINDOW_ENV, XVIEW_HYGIENE_ENV) == ("SCSAM3_XVIEW_WINDOW", "SCSAM3_XVIEW_HYGIENE")
    assert xview_env_window({}) is None
    assert xview_env_window({XVIEW_WINDOW_ENV: ""}) is None
    assert xview_env_window({XVIEW_WINDOW_ENV: " "}) is None
    assert xview_env_window({XVIEW_WINDOW_ENV: "4"}) == 4
    assert xview_env_window({XVIEW_WINDOW_ENV: " 0 "}) == 0
    with pytest.raises(ValueError):
        xview_env_window({XVIEW_WINDOW_ENV: "x"})
    assert xview_env_hygiene({}) is None
    assert xview_env_hygiene({XVIEW_HYGIENE_ENV: ""}) is None
    for v in ("1", "true", "on", "yes", "TRUE"):
        assert xview_env_hygiene({XVIEW_HYGIENE_ENV: v}) is True
    for v in ("0", "false", "off", "no", "OFF"):
        assert xview_env_hygiene({XVIEW_HYGIENE_ENV: v}) is False
    with pytest.raises(ValueError):
        xview_env_hygiene({XVIEW_HYGIENE_ENV: "maybe"})
