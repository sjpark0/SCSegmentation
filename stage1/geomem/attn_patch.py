"""Route a 3D attention bias into the SAM 3 memory cross-attention without editing upstream code.

`patch_cross_attention(tracker)` rebinds `forward` on every layer's `cross_attn_image` (a
`sam3.sam.transformer.RoPEAttention`) to a copy of the upstream body (q/k/v projections, head split,
rotary encoding on q and on the non-pointer keys with `rope_k_repeat`, SDPA, recombine, out_proj)
with two additions: the additive mask stored on `tracker._geo_attn_mask` is passed to
`F.scaled_dot_product_attention` (cast to q's dtype), and with `no_rope=True` rotary encoding is
skipped for both q and k.  `unpatch_cross_attention` restores the original bound methods.
"""
import contextlib
import math
import types

import torch
import torch.nn.functional as F


def _layers(tracker):
    return list(tracker.transformer.encoder.layers)


def _patched_forward(no_rope):
    from sam3.sam.rope import apply_rotary_enc, apply_rotary_enc_real

    def forward(self, q, k, v, num_k_exclude_rope=0):
        tracker = self._geo_tracker
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)
        if not no_rope:
            w = h = math.sqrt(q.shape[-2])
            if self.freqs_cis.shape[0] != q.shape[-2]:
                self.freqs_cis = self.compute_cis(end_x=w, end_y=h, device=q.device)
                self.freqs_cis_real = self.freqs_cis.real
                self.freqs_cis_imag = self.freqs_cis.imag
            if q.shape[-2] != k.shape[-2]:
                assert self.rope_k_repeat
            num_k_rope = k.size(-2) - num_k_exclude_rope
            if self.use_rope_real:
                q, k[:, :, :num_k_rope] = apply_rotary_enc_real(
                    q, k[:, :, :num_k_rope], freqs_cis_real=self.freqs_cis_real,
                    freqs_cis_imag=self.freqs_cis_imag, repeat_freqs_k=self.rope_k_repeat)
            else:
                q, k[:, :, :num_k_rope] = apply_rotary_enc(
                    q, k[:, :, :num_k_rope], self.freqs_cis, repeat_freqs_k=self.rope_k_repeat)
        dropout_p = self.dropout_p if self.training else 0.0
        mask = getattr(tracker, "_geo_attn_mask", None)
        if mask is not None:
            assert mask.shape[-2] == q.shape[-2] and mask.shape[-1] == k.shape[-2], (mask.shape, q.shape, k.shape)
            mask = mask.to(dtype=q.dtype, device=q.device)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)
        out = self._recombine_heads(out)
        return self.out_proj(out)

    return forward


def patch_cross_attention(tracker, no_rope=False):
    """Idempotent.  Re-patching with a different `no_rope` requires `unpatch_cross_attention` first."""
    if getattr(tracker, "_geo_patched", False):
        assert tracker._geo_no_rope == no_rope, "already patched with a different no_rope; unpatch first"
        return
    fwd = _patched_forward(no_rope)
    for layer in _layers(tracker):
        attn = layer.cross_attn_image
        # plain-attribute writes: nn.Module.__setattr__ would register the tracker as a submodule
        # of its own layer and create a module cycle (RecursionError in repr/parameters)
        object.__setattr__(attn, "_orig_forward", attn.forward)
        object.__setattr__(attn, "_geo_tracker", tracker)
        object.__setattr__(attn, "forward", types.MethodType(fwd, attn))
    tracker._geo_attn_mask = None
    tracker._geo_patched = True
    tracker._geo_no_rope = no_rope


def unpatch_cross_attention(tracker):
    if not getattr(tracker, "_geo_patched", False):
        return
    for layer in _layers(tracker):
        attn = layer.cross_attn_image
        object.__setattr__(attn, "forward", attn.__dict__["_orig_forward"])
        del attn.__dict__["_orig_forward"], attn.__dict__["_geo_tracker"]
        del attn.__dict__["forward"]                     # fall back to the class method
    tracker._geo_attn_mask = None
    tracker._geo_patched = False
    del tracker._geo_no_rope


def set_attn_mask(tracker, mask):
    tracker._geo_attn_mask = mask


@contextlib.contextmanager
def attn_mask(tracker, mask):
    prev = getattr(tracker, "_geo_attn_mask", None)
    tracker._geo_attn_mask = mask
    try:
        yield
    finally:
        tracker._geo_attn_mask = prev
