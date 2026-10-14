"""GeoMem-star: decode every view from the reference camera's memory alone, with a learned 3D
positional embedding added to the memory keys and the target queries (spec 3).

Nothing in the SAM 3 package is modified.  The reference memory is produced by the tracker's own
`track_step` with a mask prompt (exactly what prompting frame 0 of a video does); the target view is
decoded by assembling the memory prompt ourselves (one view + its 4 object-pointer tokens, the
conditioning-frame temporal row) and calling the tracker's memory attention and SAM heads.
"""
from dataclasses import dataclass, field, replace

import torch

from geomem import sam3_io
from geomem.attn_patch import attn_mask, patch_cross_attention, rope_queries, unpatch_cross_attention

PTR_TOKENS = 4          # a 256-d object pointer is split into 4 tokens of mem_dim 64
PTR_MAX_ABS_POS = 16    # tracker default max_obj_ptrs_in_encoder
from geomem.arms import ALIGN_ARMS, CONTINUE_ARMS, DEC_ARMS, FUSE_ARMS, NO_EMBED_ARMS, QUERY_ARMS, SET2_ARMS, base_arm  # noqa: E402,F401  (re-exported)

BIAS_ARMS = ("geobias_ft", "geobias_norope_ft", "geobias_only_ft", "geoset_ft") + SET2_ARMS + FUSE_ARMS + ALIGN_ARMS
FT_ARMS = ("geomem_ft",) + BIAS_ARMS
# S1-E9: decoder parameters that stay frozen in DEC_ARMS.  conv_s0 / conv_s1 are applied to the backbone output when
# the feature cache is written (sam3_io.backbone_features), so training them would change inference only.
DECODER_FROZEN = ("conv_s0", "conv_s1")


def decoder_parameters(tracker):
    """(name, parameter) of the SAM mask decoder that S1-E9 trains: everything except DECODER_FROZEN."""
    return [(n, p) for n, p in tracker.sam_mask_decoder.named_parameters() if not n.startswith(DECODER_FROZEN)]


@dataclass
class MemoryView:
    """One view's memory: tokens, positions (spatial + temporal row + 3D key embedding), pointer tokens, grid."""
    feats: torch.Tensor           # [5184, B, 64]
    pos: torch.Tensor             # [5184, B, 64]
    obj_ptr_tokens: torch.Tensor  # [4, B, 64]
    obj_ptr_pos: torch.Tensor     # [4, B, 64]
    object_score_logits: torch.Tensor  # [B, 1]
    p: torch.Tensor               # [5184, 3]
    valid: torch.Tensor           # [5184] bool
    name: str = "ref"
    conf: torch.Tensor = None     # [B] bool per-object confidence (GeoSet-2 spec 2.1); None = all confident


@dataclass
class MemorySet:
    """Memory views read by the target view.  Pointer tokens of ALL views sit at the end of the prompt so the
    tracker's `num_obj_ptr_tokens` (which excludes the last N keys from RoPE) covers every pointer token."""
    views: list = field(default_factory=list)

    def with_view(self, view):
        return MemorySet(self.views + [view])

    def prompt(self):
        feats = torch.cat([v.feats for v in self.views] + [v.obj_ptr_tokens for v in self.views], dim=0)
        pos = torch.cat([v.pos for v in self.views] + [v.obj_ptr_pos for v in self.views], dim=0)
        return feats, pos, PTR_TOKENS * len(self.views)

    def grids(self):
        return [(v.p, v.valid) for v in self.views]

    def subset(self, idx):
        """The same views restricted to the objects `idx` (LongTensor); grids are shared, conf is sliced."""
        def cut(v):
            i = idx.to(v.feats.device)
            return replace(v, feats=v.feats[:, i], pos=v.pos[:, i], obj_ptr_tokens=v.obj_ptr_tokens[:, i],
                           obj_ptr_pos=v.obj_ptr_pos[:, i], object_score_logits=v.object_score_logits[i],
                           conf=None if v.conf is None else v.conf[i.to(v.conf.device)])
        return MemorySet([cut(v) for v in self.views])

    def _gated(self):
        """(view index, ~conf [B]) for every view with at least one unconfident object; the reference is never gated."""
        ref = self.views[0].conf
        assert ref is None or bool(ref.all()), "the reference view must be confident for every object"
        return [(i, ~v.conf.bool()) for i, v in enumerate(self.views) if v.conf is not None and not bool(v.conf.all())]

    def n_gated(self):
        return int(sum(g.sum().item() for _, g in self._gated()))

    def gate_mask(self, dtype, device):
        """Additive attention mask [B,1,1,ΣL] (spec 2.2): -inf on view v's 5184 feature columns and its 4 pointer
        columns for every object whose conf_v is False; 0 elsewhere.  None when nothing is gated."""
        gated = self._gated()
        if not gated:
            return None
        V, B = len(self.views), self.views[0].feats.shape[1]
        L = self.views[0].feats.shape[0]
        m = torch.zeros(B, 1, 1, V * L + PTR_TOKENS * V, dtype=dtype, device=device)
        for i, g in gated:
            rows = g.to(device)
            m[rows, 0, 0, i * L:(i + 1) * L] = float("-inf")
            m[rows, 0, 0, V * L + PTR_TOKENS * i:V * L + PTR_TOKENS * (i + 1)] = float("-inf")
        return m

    # backwards-compatible single-view accessors
    @property
    def feats(self):
        return self.views[0].feats

    @property
    def object_score_logits(self):
        return self.views[0].object_score_logits


RefMemory = MemorySet


class GeoMemStar(torch.nn.Module):
    def __init__(self, model, geo_embed, geo_bias=None, use_geometry=True, use_bias=False, no_rope=False, fusion=None):
        super().__init__()
        self.model = model
        self.geo_embed = geo_embed
        self.geo_bias = geo_bias
        self.fusion = fusion                 # GeoFuse spec 2.2: ViewFusion applied to the target's memory-attention input
        self.last_fusion_stats = None
        self.use_geometry = use_geometry
        self.use_bias = use_bias
        self.no_rope = no_rope
        self.model.eval()
        if use_bias or no_rope:
            assert geo_bias is not None or not use_bias, "use_bias needs a GeoBias"
            patch_cross_attention(self.tracker, no_rope=no_rope)

    def close(self):
        """Undo the attention patch (tests and arm switches share one model)."""
        unpatch_cross_attention(self.tracker)

    @property
    def tracker(self):
        return self.model.tracker

    # ------------------------------------------------------------------ freezing
    def freeze(self, arm):
        """requires_grad per spec 3.4; returns the trainable parameter names."""
        arm = base_arm(arm)
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.geo_embed.parameters():
            p.requires_grad_(True)
        names = [f"geo_embed.{n}" for n, _ in self.geo_embed.named_parameters()]
        if arm in NO_EMBED_ARMS:                         # no 3D embedding in these arms
            for p in self.geo_embed.parameters():
                p.requires_grad_(False)
            names = []
        if self.geo_bias is not None and arm in BIAS_ARMS:
            for n, p in self.geo_bias.named_parameters():
                p.requires_grad_(True)
                names.append(f"geo_bias.{n}")
        if arm in FT_ARMS:
            for n, p in self.tracker.transformer.encoder.named_parameters():
                p.requires_grad_(True)
                names.append(f"tracker.transformer.encoder.{n}")
        elif arm not in ("geomem", "geomem_ssl"):
            raise ValueError(arm)
        if arm in DEC_ARMS:                              # S1-E9: the mask decoder too (conv_s0 / conv_s1 stay frozen)
            for n, p in decoder_parameters(self.tracker):
                p.requires_grad_(True)
                names.append(f"tracker.sam_mask_decoder.{n}")
        if self.fusion is not None:
            for n, p in self.fusion.named_parameters():
                p.requires_grad_(arm in FUSE_ARMS)
                if arm in FUSE_ARMS:
                    names.append(f"fusion.{n}")
        return names

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ------------------------------------------------------------------ reference memory
    def encode_reference(self, image_1008, backbone_out, masks_bool, p_ref, valid_ref, name="ref"):
        """Reference view + its GT masks -> MemorySet with one view (one batch entry per object)."""
        tr = self.tracker
        B = masks_bool.shape[0]
        vision_feats, vision_pos, feat_sizes = sam3_io.expand_features(tr, backbone_out, B)
        # The reference pass uses frozen modules only (memory encoder, SAM heads) and no memory
        # attention, so its outputs are constants for every arm; the trainable key embedding is added below.
        with torch.no_grad():
            out = tr.track_step(
                frame_idx=0, is_init_cond_frame=True,
                current_vision_feats=vision_feats, current_vision_pos_embeds=vision_pos, feat_sizes=feat_sizes,
                image=image_1008.expand(B, -1, -1, -1), point_inputs=None,
                mask_inputs=sam3_io.mask_inputs(masks_bool, tr.image_size),
                output_dict={"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
                num_frames=2, track_in_reverse=False, run_mem_encoder=True, prev_sam_mask_logits=None)
        return MemorySet([self._pack_view(out["maskmem_features"], out["maskmem_pos_enc"][-1], out["obj_ptr"],
                                          out["object_score_logits"], p_ref, valid_ref, name)])

    def encode_aligned(self, image_ref, backbone_ref, masks_bool, p_ref, valid_ref, W_v, valid_v, name):
        """GeoAlign spec 2.1: a neighbour view's level-72 features aligned to the reference grid (`geomem.align`), holes
        filled with the reference's own, plus the reference GT masks -> one MemoryView ("the reference as seen from v").
        Image, high-res levels and grid are the reference's; `backbone_ref` is not mutated."""
        from geomem.align import fill_aligned
        f2 = backbone_ref["backbone_fpn"][2]                                              # [1,256,72,72]
        F_ref = f2[0].reshape(f2.shape[1], -1).T                                         # [5184,256] row-major like flatten(2)
        filled = fill_aligned(W_v, valid_v, F_ref)
        backbone_v = dict(backbone_ref, backbone_fpn=list(backbone_ref["backbone_fpn"][:2]) + [filled.T.reshape(f2.shape)])
        return self.encode_reference(image_ref, backbone_v, masks_bool, p_ref, valid_ref, name=name).views[0]

    def _pack_view(self, maskmem_features, maskmem_pos_enc, obj_ptr, object_score_logits, p, valid, name):
        tr = self.tracker
        B = maskmem_features.shape[0]
        feats = maskmem_features.flatten(2).permute(2, 0, 1)                      # [5184,B,64]
        pos = maskmem_pos_enc.flatten(2).permute(2, 0, 1)                         # [5184,B,64]
        pos = pos + tr.maskmem_tpos_enc[tr.num_maskmem - 1]                       # conditioning-frame row
        if self.use_geometry:
            key, _ = self.geo_embed(p, valid)
            pos = pos + key.to(pos.dtype)[:, None, :]
        C, M = obj_ptr.shape[1], tr.mem_dim
        ptr_tokens = obj_ptr.reshape(B, C // M, M).permute(1, 0, 2)               # [4,B,64]
        ptr_pos = tr._get_tpos_enc([1], max_abs_pos=PTR_MAX_ABS_POS, device=obj_ptr.device)
        ptr_pos = ptr_pos.unsqueeze(1).expand(-1, B, -1).repeat_interleave(C // M, dim=0)
        return MemoryView(feats=feats, pos=pos, obj_ptr_tokens=ptr_tokens.to(feats.dtype), obj_ptr_pos=ptr_pos.to(pos.dtype),
                          object_score_logits=object_score_logits, p=p, valid=valid, name=name)

    def encode_memory_from_prediction(self, backbone_out, B, logits_1008, presence, obj_ptr, p_v, valid_v, name,
                                      ious=None, tau=None, conf=None):
        """A decoded view's own memory (as tracking does after predicting a frame), for the memory set.
        With `tau` (GeoSet-2 spec 2.1) the view carries conf = (presence > 0) & (ious >= tau); `ious` is then required.
        An explicit `conf` [B] bool (GeoSet-2 §8 gate signals) takes precedence."""
        tr = self.tracker
        if conf is None and tau is not None:
            assert ious is not None, "conf needs the predicted IoUs"
            conf = ((presence > 0) & (ious >= tau)).detach()
        vision_feats, _, feat_sizes = sam3_io.expand_features(tr, backbone_out, B)
        with torch.no_grad():
            maskmem_features, maskmem_pos_enc = tr._encode_new_memory(
                image=None, current_vision_feats=vision_feats, feat_sizes=feat_sizes,
                pred_masks_high_res=logits_1008, object_score_logits=presence[:, None],
                is_mask_from_pts=False, output_dict=None, is_init_cond_frame=False)
        view = self._pack_view(maskmem_features, maskmem_pos_enc[-1], obj_ptr.detach(), presence[:, None].detach(),
                               p_v, valid_v, name)
        return replace(view, conf=conf)

    # ------------------------------------------------------------------ target decoding
    def decode_view(self, backbone_out, B, memory, p_v, valid_v, chunk=None, aligned=None, rope_q=None, **kw):
        """Target view features + memory set -> dict(logits_1008 [B,1,1008,1008], ious [B], presence [B], low_res, obj_ptr [B,256]).
        Extra keyword arguments go to `_heads` (GeoQuery passes `query=`).  `chunk` decodes at most that many objects
        at a time when the set is gated (the per-object mask [B,1,Lq,ΣL] grows with B; GeoSet-2 OOM guard).
        `aligned = (W [N,5184,256], valid [N,5184])` (geomem.align) feeds the fusion module (GeoFuse spec 2.3): the
        memory-attention input `src` becomes fusion(src, W, valid); ignored when the wrapper has no fusion.
        `rope_q` (S1-E10 C2): a query RoPE table (attn_patch.axial_cis_at) used by the memory cross-attention in
        place of the layers' own; None = the layers' table (unchanged path)."""
        tr = self.tracker
        if chunk is not None and B > chunk and memory.gate_mask(torch.float32, p_v.device) is not None:
            outs = [self.decode_view(backbone_out, len(i), memory.subset(i), p_v, valid_v, aligned=aligned, rope_q=rope_q, **kw)
                    for i in torch.arange(B, device=p_v.device).split(chunk)]
            return {k: torch.cat([o[k] for o in outs], dim=0) for k in outs[0]}
        vision_feats, vision_pos, feat_sizes = sam3_io.expand_features(tr, backbone_out, B)
        if self.fusion is not None and aligned is not None:
            src = vision_feats[-1]                                                     # [5184, B, 256], B identical copies
            fused, self.last_fusion_stats = self.fusion(src[:, 0], aligned[0], aligned[1])
            vision_feats = list(vision_feats[:-1]) + [fused[:, None, :].expand(-1, B, -1).to(src.dtype)]
        src_pos = vision_pos[-1]
        if self.use_geometry:
            _, query = self.geo_embed(p_v, valid_v)
            src_pos = src_pos + query.to(src_pos.dtype)[:, None, :]
        prompt, prompt_pos, n_ptr = memory.prompt()
        mask = self.geo_bias(p_v, valid_v, memory.grids()) if self.use_bias else None
        gate = memory.gate_mask(mask.dtype if mask is not None else torch.float32, p_v.device)   # [B,1,1,ΣL] or None
        if gate is not None:
            mask = gate if mask is None else mask + gate                                  # -> [B,1,Lq,ΣL]
        with attn_mask(tr, mask) if (mask is not None or self.no_rope) else _nullctx(), \
                rope_queries(tr, rope_q) if rope_q is not None else _nullctx():
            enc = tr.transformer.encoder(
                src=vision_feats[-1:], src_key_padding_mask=[None], src_pos=[src_pos],
                prompt=prompt, prompt_pos=prompt_pos, prompt_key_padding_mask=None,
                feat_sizes=feat_sizes[-1:], num_obj_ptr_tokens=n_ptr)
        H, W = feat_sizes[-1]
        pix_feat_with_mem = enc["memory"].permute(1, 2, 0).view(B, tr.hidden_dim, H, W)
        high_res = [x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                    for x, s in zip(vision_feats[:-1], feat_sizes[:-1])]
        return self._heads(pix_feat_with_mem, high_res, **kw)

    def _heads(self, pix_feat_with_mem, high_res):
        (_, _, ious, low_res, high_res_masks, obj_ptr, object_score_logits) = self.tracker._forward_sam_heads(
            backbone_features=pix_feat_with_mem, point_inputs=None, mask_inputs=None,
            high_res_features=high_res, multimask_output=False)
        return {"logits_1008": high_res_masks, "ious": ious[:, 0], "presence": object_score_logits[:, 0],
                "low_res": low_res, "obj_ptr": obj_ptr}


class GeoQuery(GeoMemStar):
    """geobias_only path + object-query hub (GeoQuery spec 2): the hub reads every view's tokens once per
    scene and its queries enter every view's SAM mask decoder as one extra sparse prompt token."""

    def __init__(self, model, geo_embed, geo_bias=None, hub=None, inject="prompt", **flags):
        """`inject`: "prompt" adds the query to the decoder's padding prompt token (spec 2.4); "ptr" adds
        `hub.ptr_delta(q)` to the reference view's pointer tokens before the memory attention (diagnosis 2026-10-08:
        the frozen decoder barely reacts to the prompt token, the pointer channel is ~14x more sensitive)."""
        super().__init__(model, geo_embed, geo_bias=geo_bias, **flags)
        assert inject in ("prompt", "ptr"), inject
        self.hub = hub
        self.inject = inject

    def freeze(self, arm):
        names = super().freeze(arm)
        if self.hub is not None:
            for n, p in self.hub.named_parameters():
                p.requires_grad_(True)
                names.append(f"hub.{n}")
        return names

    @staticmethod
    def obj_ptr_of(memory):
        """The reference view's object pointers [B,256] re-assembled from its 4 pointer tokens."""
        t = memory.views[0].obj_ptr_tokens                                   # [4,B,64]
        return t.permute(1, 0, 2).reshape(t.shape[1], -1)

    def hub_tokens(self, backbone_outs):
        """Backbone features + positional encodings of each view (B=1), concatenated -> (tokens [N*5184,256], view_ids [N*5184])."""
        toks = []
        for bb in backbone_outs:
            feats, pos, _ = sam3_io.expand_features(self.tracker, bb, 1)
            toks.append((feats[-1] + pos[-1])[:, 0, :])
        tokens = torch.cat(toks, 0)
        view_ids = torch.arange(len(toks), device=tokens.device).repeat_interleave(toks[0].shape[0])
        return tokens, view_ids

    def run_hub(self, memory, backbone_outs, p_views, valid_views, ref_masks_bool, p_ref_np, valid_ref_np):
        """Object clouds from the reference masks, token distances, one hub pass -> dict(q [B,256], mass [L,B,N], cloud_sizes)."""
        from geomem.objcloud import cloud_distances, object_clouds
        clouds = object_clouds(ref_masks_bool, p_ref_np, valid_ref_np)
        dist = cloud_distances(p_views, valid_views, clouds) if self.hub.obj_bias is not None else None
        tokens, view_ids = self.hub_tokens(backbone_outs)
        q, mass = self.hub(self.obj_ptr_of(memory), tokens, dist, view_ids)
        return {"q": q, "mass": mass, "cloud_sizes": [int(len(c)) for c in clouds]}

    def memory_with_query(self, memory, query):
        """MemorySet whose reference view carries obj_ptr_tokens + ptr_delta(query); other views unchanged."""
        from dataclasses import replace
        v = memory.views[0]
        delta = self.hub.ptr_delta(query).to(v.obj_ptr_tokens.dtype)
        return MemorySet([replace(v, obj_ptr_tokens=v.obj_ptr_tokens + delta)] + list(memory.views[1:]))

    def decode_view(self, backbone_out, B, memory, p_v, valid_v, query=None):
        if query is not None and self.inject == "ptr":
            return super().decode_view(backbone_out, B, self.memory_with_query(memory, query), p_v, valid_v)
        return super().decode_view(backbone_out, B, memory, p_v, valid_v, query=query)

    def _heads(self, pix_feat_with_mem, high_res, query=None):
        if query is None:
            return super()._heads(pix_feat_with_mem, high_res)
        ious, low_res, high_res_masks, obj_ptr, object_score_logits = self._sam_heads_with_query(
            pix_feat_with_mem, high_res, self.hub.prompt_token(query))
        return {"logits_1008": high_res_masks, "ious": ious[:, 0], "presence": object_score_logits[:, 0],
                "low_res": low_res, "obj_ptr": obj_ptr}

    def _sam_heads_with_query(self, backbone_features, high_res_features, tok):
        """`_forward_sam_heads` (sam3_tracker_base.py:262-379) with the query token `tok` [B,256] ADDED to the
        empty-point padding token (so a zero token reproduces the parent path exactly: same token count, same
        attention normalisation); mask decoder in single-mask mode; same tail."""
        from sam3.model.sam3_tracker_base import NO_OBJ_SCORE
        tr = self.tracker
        B, device = backbone_features.shape[0], backbone_features.device
        coords = torch.zeros(B, 1, 2, device=device)
        labels = -torch.ones(B, 1, dtype=torch.int32, device=device)
        sparse, dense = tr.sam_prompt_encoder(points=(coords, labels), boxes=None, masks=None)
        # the empty prompt is [B,2,256]: our label −1 point and SAM's own padding point, both `not_a_point_embed`
        sparse = torch.cat([sparse[:, :1] + tok.to(sparse.dtype)[:, None, :], sparse[:, 1:]], dim=1)
        low_res_multimasks, ious, sam_output_tokens, object_score_logits = tr.sam_mask_decoder(
            image_embeddings=backbone_features, image_pe=tr.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=False, repeat_image=False, high_res_features=high_res_features)
        is_obj_appearing = object_score_logits > 0
        low_res_multimasks = torch.where(is_obj_appearing[:, None, None], low_res_multimasks, NO_OBJ_SCORE).float()
        high_res_multimasks = torch.nn.functional.interpolate(
            low_res_multimasks, size=(tr.image_size, tr.image_size), mode="bilinear", align_corners=False)
        obj_ptr = tr.obj_ptr_proj(sam_output_tokens[:, 0])
        lam = is_obj_appearing.float()
        obj_ptr = lam * obj_ptr + (1 - lam) * tr.no_obj_ptr
        return ious, low_res_multimasks, high_res_multimasks, obj_ptr, object_score_logits


import contextlib as _contextlib


@_contextlib.contextmanager
def _nullctx():
    yield
