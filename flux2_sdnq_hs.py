"""Reusable FLUX.2-klein SDNQ speed path for PyTorch/MPS.

This is the quality-preserving path from the 2K experiments:

- exact query-chunked SDPA to avoid Metal's full attention buffer crash
- hidden-state compression for single-stream transformer blocks
- compress denoising forwards 0..N-2, leave the final forward exact

The failed KV pooling, JiT cache, and SEGA experiments stay in research notes;
this module keeps only the settings that survived visual testing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Flux2SdnqHsConfig:
    qchunk: int = 1024
    hs_stride: int = 2
    hs_skip_transformer_forwards: int = 0
    hs_max_transformer_forward: int | None = None
    hs_single_start_frac: float = 0.0
    hs_single_end_frac: float = 1.0
    verbose: bool = False

    @classmethod
    def for_steps(cls, steps: int, **overrides) -> "Flux2SdnqHsConfig":
        cfg = cls(**overrides)
        if cfg.hs_max_transformer_forward is None:
            # Exclusive upper bound. For 4 steps this compresses calls 0,1,2
            # and leaves call 3 exact, which was the clean 2K setting.
            cfg.hs_max_transformer_forward = max(0, int(steps) - 1)
        return cfg


def _perfect_square(n: int) -> int | None:
    if n <= 0:
        return None
    side = int(n**0.5)
    return side if side * side == n else None


def _infer_text_and_side(seq_len: int) -> tuple[int | None, int | None]:
    for text_len in (512, 256, 1024, 0):
        side = _perfect_square(seq_len - text_len)
        if side is not None:
            return text_len, side

    for text_len in range(0, min(2048, seq_len) + 1):
        side = _perfect_square(seq_len - text_len)
        if side is not None:
            return text_len, side

    return None, None


def _in_frac_window(index: int, total: int, start_frac: float, end_frac: float) -> bool:
    start = int(total * start_frac)
    end = int(total * end_frac)
    return start <= index < max(start + 1, end)


def _downsample_image_tokens(img: torch.Tensor, side: int, stride: int) -> torch.Tensor:
    bsz, _, dim = img.shape
    low_side = side // stride
    grid = img.reshape(bsz, side, side, dim)
    low = grid.reshape(bsz, low_side, stride, low_side, stride, dim).mean(dim=(2, 4))
    return low.reshape(bsz, low_side * low_side, dim)


def _upsample_image_tokens(img_low: torch.Tensor, side: int, stride: int) -> torch.Tensor:
    bsz, _, dim = img_low.shape
    low_side = side // stride
    grid = img_low.reshape(bsz, low_side, low_side, dim)
    up = grid.repeat_interleave(stride, dim=1).repeat_interleave(stride, dim=2)
    return up.reshape(bsz, side * side, dim)


def _reduced_rotary_emb(image_rotary_emb, text_len: int, side: int, stride: int):
    if image_rotary_emb is None:
        return None

    low_side = side // stride
    yy = torch.arange(low_side, device=image_rotary_emb[0].device) * stride + stride // 2
    xx = torch.arange(low_side, device=image_rotary_emb[0].device) * stride + stride // 2
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    image_idx = (grid_y * side + grid_x).flatten()

    reduced = []
    for emb in image_rotary_emb:
        text_emb = emb[:text_len]
        img_emb = emb[text_len:].index_select(0, image_idx)
        reduced.append(torch.cat([text_emb, img_emb], dim=0))
    return tuple(reduced)


def patch_exact_chunked_mps_attention(qchunk: int = 1024):
    """Install exact query-chunked SDPA and disable the failed KV pooling path."""
    import mps_chunked_attn

    mps_chunked_attn.QCHUNK = int(qchunk)
    mps_chunked_attn.KV_STRIDE = 1
    mps_chunked_attn.KV_KEEP_TILE_FRAC = 0.0
    mps_chunked_attn.KV_VALUE_MODE = "mean"
    mps_chunked_attn.KV_LOCAL_RADIUS = -1
    mps_chunked_attn.KV_LOCAL_TILE = 0
    return mps_chunked_attn.patch()


def reset_flux2_sdnq_hs_state(pipe) -> None:
    state = getattr(pipe, "_flux2_sdnq_hs_state", None)
    if state is not None:
        state["transformer_call"] = 0
        state["active_call"] = None


def install_flux2_sdnq_hs_optimizations(pipe, config: Flux2SdnqHsConfig):
    """Patch a loaded Flux2KleinPipeline in-place for the 2K PyTorch SDNQ path."""
    patch_cfg = patch_exact_chunked_mps_attention(config.qchunk)

    if getattr(pipe, "_flux2_sdnq_hs_installed", False):
        pipe._flux2_sdnq_hs_config = config
        reset_flux2_sdnq_hs_state(pipe)
        return patch_cfg

    state = {"transformer_call": 0, "active_call": None}
    pipe._flux2_sdnq_hs_state = state
    pipe._flux2_sdnq_hs_config = config

    orig_transformer_forward = pipe.transformer.forward

    def transformer_forward_with_call_index(*args, **kwargs):
        cfg = pipe._flux2_sdnq_hs_config
        call_index = state["transformer_call"]
        state["active_call"] = call_index
        if cfg.verbose:
            compress = (
                cfg.hs_stride > 1
                and cfg.hs_skip_transformer_forwards <= call_index < cfg.hs_max_transformer_forward
            )
            print(f"FLUX2_SDNQ_HS transformer_call={call_index} compress_single={compress}")
        try:
            return orig_transformer_forward(*args, **kwargs)
        finally:
            state["transformer_call"] += 1
            state["active_call"] = None

    pipe.transformer.forward = transformer_forward_with_call_index

    def hs_active(index: int, total: int) -> bool:
        cfg = pipe._flux2_sdnq_hs_config
        call_index = state["active_call"]
        if cfg.hs_stride <= 1:
            return False
        if call_index is None:
            return False
        if call_index < cfg.hs_skip_transformer_forwards:
            return False
        if call_index >= cfg.hs_max_transformer_forward:
            return False
        return _in_frac_window(index, total, cfg.hs_single_start_frac, cfg.hs_single_end_frac)

    def run_hidden_compressed_single_block(orig, args, kwargs, index: int, total: int):
        cfg = pipe._flux2_sdnq_hs_config
        if not hs_active(index, total):
            return orig(*args, **kwargs)
        if args or kwargs.get("encoder_hidden_states", None) is not None:
            return orig(*args, **kwargs)

        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None:
            return orig(*args, **kwargs)

        text_len, side = _infer_text_and_side(hidden_states.shape[1])
        if text_len is None or side % cfg.hs_stride != 0:
            return orig(*args, **kwargs)

        text, img = hidden_states[:, :text_len], hidden_states[:, text_len:]
        img_low = _downsample_image_tokens(img, side, cfg.hs_stride)
        reduced_hidden = torch.cat([text, img_low], dim=1)

        reduced_kwargs = dict(kwargs)
        reduced_kwargs["hidden_states"] = reduced_hidden
        reduced_kwargs["image_rotary_emb"] = _reduced_rotary_emb(
            kwargs.get("image_rotary_emb"), text_len, side, cfg.hs_stride
        )

        out = orig(**reduced_kwargs)
        out_text, out_img_low = out[:, :text_len], out[:, text_len:]

        img_delta_low = out_img_low - img_low
        restored_img = img + _upsample_image_tokens(img_delta_low, side, cfg.hs_stride)
        return torch.cat([out_text, restored_img], dim=1)

    blocks = pipe.transformer.single_transformer_blocks
    total = len(blocks)
    for index, block in enumerate(blocks):
        orig = block.forward

        def wrapped_single_block(*args, _orig=orig, _index=index, _total=total, **kwargs):
            return run_hidden_compressed_single_block(_orig, args, kwargs, _index, _total)

        block.forward = wrapped_single_block

    pipe._flux2_sdnq_hs_installed = True
    reset_flux2_sdnq_hs_state(pipe)
    return patch_cfg
