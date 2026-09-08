"""CPU harness for the REAL SCSam3TrackerPredictorNewMem._prepare_memory_conditioned_features_multiple.

The real tracker cannot be constructed on CPU (PositionEmbeddingSine allocates on cuda),
so the tests use a bare instance: __new__ + torch.nn.Module.__init__, the attributes the
method reads set by hand, a stub transformer.encoder that records its kwargs, and
maskmem_tpos_enc rows equal to their row index so the temporal row of every memory
token can be read back from `prompt_pos`.  Fake stored outputs carry a constant
maskmem_features / obj_ptr value 1000*view+frame so the token SOURCE can be read back
from `prompt`.  Requires the `cpu_tensors` fixture (Tensor.cuda / pin_memory -> identity).
"""
import hashlib

import pytest

torch = pytest.importorskip("torch")
import SCSam3TrackerPredictorNewMem as M  # noqa: E402  (package module, conftest put PKG on sys.path)

C, MEM = 256, 64          # hidden_dim, mem_dim of the real model
GOLDEN_HW = 72            # 1008 / 14: the real memory grid, 5184 tokens per memory
SMALL_HW = 8              # for the lockstep drives


def bare_tracker(num_maskmem=7, max_cond=4, keep_first=False, use_sel=True, xw=None, xh=None,
                 xm=None, xg=None, xp=None, xs=None, mf=0.01):
    tr = M.SCSam3TrackerPredictorNewMem.__new__(M.SCSam3TrackerPredictorNewMem)
    torch.nn.Module.__init__(tr)
    tr.training = False
    tr.hidden_dim = C
    tr.mem_dim = MEM
    tr.num_maskmem = num_maskmem
    tr.maskmem_tpos_enc = (torch.arange(num_maskmem, dtype=torch.float32)
                           .view(num_maskmem, 1, 1, 1).expand(num_maskmem, 1, 1, MEM).clone())
    tr.max_cond_frames_in_attn = max_cond
    tr.keep_first_cond_frame = keep_first
    tr.memory_temporal_stride_for_eval = 1
    tr.use_memory_selection = use_sel
    tr.max_obj_ptrs_in_encoder = 16
    tr.mf_threshold = mf
    tr.cond_frame_spatial_embedding = None
    tr.cond_frame_obj_ptr_embedding = None
    torch.manual_seed(7)
    tr.obj_ptr_tpos_proj = torch.nn.Linear(C, MEM)   # real model projects hidden_dim -> mem_dim

    class Enc:
        def __init__(s):
            s.calls = []

        def __call__(s, **kw):
            s.calls.append(kw)
            return {"memory": kw["src"][0]}

    class Tr:
        pass

    tr.transformer = Tr()
    tr.transformer.encoder = Enc()
    # what __init__ would have done (the bare instance skipped it)
    tr.cross_view_window, tr.cross_view_hygiene = M.resolve_cross_view(xw, xh, num_maskmem)
    tr.cross_view_mode = M.resolve_cross_view_mode(xm, tr.cross_view_hygiene)
    tr.cross_view_gate, tr.cross_view_ptr, tr.cross_view_tpos_shift = M.resolve_cross_view_knobs(
        xg, xp, xs, tr.cross_view_window, tr.cross_view_hygiene, num_maskmem)
    tr.xview_stats = M.new_xview_stats()
    # record the rel_pos_list handed to _get_tpos_enc (pointer temporal positions)
    tr._tpos_calls = []
    _orig_tpos = tr._get_tpos_enc

    def _rec_tpos(rel_pos_list, device, max_abs_pos=None, dummy=False):
        tr._tpos_calls.append((list(rel_pos_list), max_abs_pos))
        return _orig_tpos(rel_pos_list, device, max_abs_pos=max_abs_pos, dummy=dummy)
    tr._get_tpos_enc = _rec_tpos
    return tr


def mem_out(frame, view, hw=GOLDEN_HW, eff=1.0):
    """One stored output; eff=None omits "eff_iou_score" (memory selection off, or the
    consolidated seed cond entry, which never carries the key)."""
    val = float(1000 * view + frame)
    out = {"maskmem_features": torch.full((1, MEM, hw, hw), val),
           "maskmem_pos_enc": [torch.zeros(1, MEM, hw, hw)],
           "obj_ptr": torch.full((1, C), val),
           "object_score_logits": torch.tensor([[10.0]])}
    if eff is not None:
        out["eff_iou_score"] = torch.tensor(float(eff))      # 0-dim, like cal_mem_score
    return out


def output_dicts_lockstep(N, v, t, start=0, hw=GOLDEN_HW, all_hold_t=False):
    """Session m holds cond[start]; non_cond[start+1..t-1] for all m; non_cond[t] iff m < v
    (runMVSeg.py:346-348 lockstep: lower views have already computed frame t).
    all_hold_t: every session holds non_cond[t] (mode-E pass-2 inputs: pass-1 outputs)."""
    ods = []
    for m in range(N):
        nc = {f: mem_out(f, m, hw) for f in range(start + 1, t)}
        if m < v or all_hold_t:
            nc[t] = mem_out(t, m, hw)
        ods.append({"cond_frame_outputs": {start: mem_out(start, m, hw)},
                    "non_cond_frame_outputs": nc})
    return ods


def call(tr, ods, v, t, hw=GOLDEN_HW, num_frames=22, rev=False, xview_pass=None):
    """One real call; returns the recorded encoder kwargs."""
    seq = hw * hw
    feats = [torch.zeros(seq, 1, C)]
    pos = [torch.zeros(seq, 1, C)]
    tr.transformer.encoder.calls.clear()
    tr._tpos_calls.clear()
    with torch.no_grad():          # obj_ptr_tpos_proj is a Linear: keep the record grad-free
        tr._prepare_memory_conditioned_features_multiple(
            frame_idx=t, spatial_idx=v, is_init_cond_frame=False,
            current_vision_feats=feats, current_vision_pos_embeds=pos, feat_sizes=[(hw, hw)],
            output_dicts=ods, num_frames=num_frames, track_in_reverse=rev,
            xview_pass=xview_pass)
    return tr.transformer.encoder.calls[-1]


def run(tr, ods, v, t, hw=GOLDEN_HW, num_frames=22, xview_pass=None, rev=False):
    """Decode the recorded prompt: memory-token sources, tpos rows, pointer sources."""
    seq = hw * hw
    kw = call(tr, ods, v, t, hw, num_frames, rev=rev, xview_pass=xview_pass)
    prompt, ppos, nptr = kw["prompt"], kw["prompt_pos"], kw["num_obj_ptr_tokens"]
    n_mem = (prompt.shape[0] - nptr) // seq
    per_ptr = C // MEM                                   # each pointer is split into 4 tokens
    mem_src = [int(prompt[i * seq, 0, 0].item()) for i in range(n_mem)]
    tpos_rows = [int(ppos[i * seq, 0, 0].item()) for i in range(n_mem)]
    ptr_src = [int(prompt[n_mem * seq + k * per_ptr, 0, 0].item()) for k in range(nptr // per_ptr)]
    return dict(n_mem=n_mem, mem_src=mem_src, tpos_rows=tpos_rows, n_ptr_tokens=nptr,
                ptr_src=ptr_src, prompt=prompt, ppos=ppos,
                ptr_pos=(tr._tpos_calls[-1][0] if tr._tpos_calls else []),
                max_abs_pos=(tr._tpos_calls[-1][1] if tr._tpos_calls else None))


# ------------------------------------------------------------- lockstep drive
def seed_out(view, hw):
    g = torch.Generator().manual_seed(10_000 + view)
    return {"maskmem_features": torch.randn(1, MEM, hw, hw, generator=g),
            "maskmem_pos_enc": [torch.randn(1, MEM, hw, hw, generator=g)],
            "obj_ptr": torch.randn(1, C, generator=g),
            "object_score_logits": torch.tensor([[10.0]]),
            "eff_iou_score": torch.tensor([1.0])}


def derived_out(prompt, ppos, view, frame, hw):
    """Stored memory as a deterministic function of the encoder inputs, so a difference
    in what view v read at frame t propagates to every later reader of view v."""
    h = hashlib.sha256(prompt.detach().numpy().tobytes() + ppos.detach().numpy().tobytes()).digest()
    g = torch.Generator().manual_seed(int.from_bytes(h[:8], "little") % (1 << 62))
    return {"maskmem_features": torch.randn(1, MEM, hw, hw, generator=g),
            "maskmem_pos_enc": [torch.randn(1, MEM, hw, hw, generator=g)],
            "obj_ptr": torch.randn(1, C, generator=g),
            "object_score_logits": torch.tensor([[10.0]]),
            "eff_iou_score": torch.tensor([float((view * 7 + frame) % 5 != 0)])}  # some frames filtered


def lockstep(tr, N, T=20, start=0, hw=SMALL_HW, two_pass=False, pass2_reverse=False,
             gauss_seidel=False):
    """Runner order (for t: for v), frames start+1..start+T.  {(v, t): (prompt, ppos, nptr)}.

    two_pass (mode E, run_two_pass): per frame, pass 1 stores every session's output at
    once (as :916 does), then pass 2 (xview_pass=2) is computed for every session and
    committed only after the sweep (Jacobi).  `rec` then holds the pass-2 inputs, the
    ones that define the written mask.  pass2_reverse sweeps pass 2 in reversed session
    order; gauss_seidel commits inside the sweep (the order-sensitive variant, for the
    test that shows the Jacobi commit is what makes E order-free)."""
    sessions = [{"cond_frame_outputs": {start: seed_out(m, hw)}, "non_cond_frame_outputs": {}}
                for m in range(N)]
    rec = {}
    num_frames = start + T + 1
    for t in range(start + 1, start + T + 1):
        if not two_pass:
            for v in range(N):
                kw = call(tr, sessions, v, t, hw, num_frames)
                p, pp, n = kw["prompt"].clone(), kw["prompt_pos"].clone(), kw["num_obj_ptr_tokens"]
                rec[(v, t)] = (p, pp, n)
                sessions[v]["non_cond_frame_outputs"][t] = derived_out(p, pp, v, t, hw)
            continue
        for v in range(N):                                     # pass 1 (both_tm1), stored at once
            kw = call(tr, sessions, v, t, hw, num_frames, xview_pass=1)
            sessions[v]["non_cond_frame_outputs"][t] = derived_out(kw["prompt"], kw["prompt_pos"], v, t, hw)
        staged = {}
        order = list(range(N))[::-1] if pass2_reverse else list(range(N))
        for v in order:                                        # pass 2 (all_t), compute
            kw = call(tr, sessions, v, t, hw, num_frames, xview_pass=2)
            p, pp, n = kw["prompt"].clone(), kw["prompt_pos"].clone(), kw["num_obj_ptr_tokens"]
            rec[(v, t)] = (p, pp, n)
            staged[v] = derived_out(p, pp, v, t, hw)
            if gauss_seidel:
                sessions[v]["non_cond_frame_outputs"][t] = staged[v]
        for v in range(N):                                     # Jacobi commit
            sessions[v]["non_cond_frame_outputs"][t] = staged[v]
    return rec


def same_inputs(a, b):
    return a[2] == b[2] and torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
