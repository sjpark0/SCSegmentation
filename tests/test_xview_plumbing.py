"""T3: the six cross-view constructor kwargs (Phase 2 window/hygiene, P4 mode, P12 gate/
ptr/tpos-shift) are threaded through all five hops, the env helpers parse strictly, and
the frozen OneStage SCSam3Video.__init__(self, device) contract the runner relies on (it
passes no kwargs for non-XW runs) is intact."""
import ast
import inspect
import os

import pytest

from conftest import PKG, REPO

NAMES = ("cross_view_window", "cross_view_hygiene", "cross_view_mode",
         "cross_view_gate", "cross_view_ptr", "cross_view_tpos_shift")


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
    assert len(a.defaults) == 6 and all(d.value is None for d in a.defaults)


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
    # the mode bound (a letter other than A needs hygiene) lives there too
    with pytest.raises(ValueError):
        T.resolve_cross_view_mode("B", False)
    assert T.resolve_cross_view_mode(None, False) == "A"
    # the P12 knob bounds live in the pure module too (W + s <= num_maskmem - 1)
    with pytest.raises(ValueError):
        T.resolve_cross_view_knobs(None, None, 2, 5, True, 7)
    assert T.resolve_cross_view_knobs(None, None, None, 4, False, 7) == (False, False, 0)
    assert T.new_xview_stats()["calls"] == 0


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


# ------------------------------------------------------------- hop-by-hop forwarding (AST)
HOPS = (  # (file, class or None, function) -> the downstream builder/constructor it calls
    (os.path.join(PKG, "SCSam3Video.py"), "SCSam3Video", "__init__", "build_scsam3_video_predictor_newmem"),
    (os.path.join(PKG, "SCSam3VideoPredictorNewMem.py"), "SCSam3VideoPredictorNewMem", "__init__",
     "build_scsam3_video_model_newmem"),
    (os.path.join(PKG, "build_scsam3.py"), None, "build_scsam3_video_model_newmem", "build_tracker_newmem"),
    (os.path.join(PKG, "build_scsam3.py"), None, "build_tracker_newmem", "SCSam3TrackerPredictorNewMem"),
)


def _function_node(tree, cls_name, fn_name):
    scope = tree.body if cls_name is None else next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls_name).body
    return next(n for n in scope if isinstance(n, ast.FunctionDef) and n.name == fn_name)


def _callee(call):
    f = call.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None


def _check_hop(fn, callee):
    """The body of `fn` calls `callee` exactly once, and that call carries name=name for
    every cross-view kwarg, `name` being a parameter of `fn` itself (no literal, no other
    variable, not hidden in a ** splat)."""
    params = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and _callee(n) == callee]
    assert len(calls) == 1, (fn.name, callee, len(calls))
    forwarded = {k.arg: k.value for k in calls[0].keywords if k.arg is not None}
    for name in NAMES:
        assert name in params, (fn.name, name)
        assert name in forwarded, (fn.name, callee, name)                     # dropped keyword
        val = forwarded[name]
        assert isinstance(val, ast.Name) and val.id == name, (fn.name, callee, name, ast.dump(val))


def test_hops_forward_cross_view_kwargs_ast():
    for path, cls_name, fn_name, callee in HOPS:
        tree = ast.parse(open(path, encoding="utf-8").read())
        _check_hop(_function_node(tree, cls_name, fn_name), callee)
    # the checker is sensitive: a hop that drops a keyword, binds it to a literal or to
    # another parameter, or hides it in a splat, fails
    # (the synthetic source declares and forwards all six names, so each `bad` below
    #  isolates one of the six)
    good = ("def build_tracker_newmem(a, cross_view_window=None, cross_view_hygiene=None,\n"
            "                         cross_view_mode=None, cross_view_gate=None,\n"
            "                         cross_view_ptr=None, cross_view_tpos_shift=None):\n"
            "    return SCSam3TrackerPredictorNewMem(a, cross_view_window=cross_view_window,\n"
            "        cross_view_hygiene=cross_view_hygiene, cross_view_mode=cross_view_mode,\n"
            "        cross_view_gate=cross_view_gate, cross_view_ptr=cross_view_ptr,\n"
            "        cross_view_tpos_shift=cross_view_tpos_shift)\n")
    _check_hop(_function_node(ast.parse(good), None, "build_tracker_newmem"), "SCSam3TrackerPredictorNewMem")
    bads = (good.replace(" cross_view_mode=cross_view_mode,\n", "\n"),              # dropped
            good.replace("cross_view_window=cross_view_window,", "cross_view_window=None,"),
            good.replace("cross_view_hygiene=cross_view_hygiene", "cross_view_hygiene=cross_view_window"),
            good.replace("cross_view_tpos_shift=cross_view_tpos_shift)", "**kw)"),  # splat
            good.replace("cross_view_gate=cross_view_gate, ", ""),                  # dropped
            good.replace("cross_view_ptr=cross_view_ptr,", "cross_view_ptr=cross_view_gate,"))
    assert len(bads) == len(NAMES)                       # one derivation per name
    for bad in bads:
        assert bad != good
        with pytest.raises(AssertionError):
            _check_hop(_function_node(ast.parse(bad), None, "build_tracker_newmem"), "SCSam3TrackerPredictorNewMem")
