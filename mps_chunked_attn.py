"""Chunked scaled-dot-product attention for MPS.

The MPS "math" SDPA path materializes the full Lq x Lk score matrix, which hard-
aborts (Metal buffer limit) once the token grid is large (e.g. klein @ 2048**2 ->
~16.9k tokens). We monkeypatch F.scaled_dot_product_attention to compute attention
in query chunks. Each query still attends to all keys, so exact mode is
mathematically identical to full attention, but never allocates the whole matrix.

Optional image-KV compression attacks time: keep all query tokens, keep text K/V
tokens exact, and pool only the image K/V tail on the latent grid.

Tune via env:
  QCHUNK=1024
  KV_STRIDE=2
  KV_KEEP_TILE_FRAC=0.0
  KV_VALUE_MODE=mean|center|detail
  KV_LOCAL_RADIUS=-1       exact local image rows per query chunk; -1 disables
  KV_LOCAL_TILE=0          use square image-query tiles when >0
  KV_MIN_SEQ=4096
"""

from collections import Counter
import os
import time

import torch
import torch.nn.functional as F

_orig_sdpa = F.scaled_dot_product_attention
QCHUNK = int(os.environ.get("QCHUNK", "1024"))
KV_STRIDE = int(os.environ.get("KV_STRIDE", "1"))
KV_KEEP_TILE_FRAC = float(os.environ.get("KV_KEEP_TILE_FRAC", "0.0"))
KV_VALUE_MODE = os.environ.get("KV_VALUE_MODE", "mean")
KV_LOCAL_RADIUS = int(os.environ.get("KV_LOCAL_RADIUS", "-1"))
KV_LOCAL_TILE = int(os.environ.get("KV_LOCAL_TILE", "0"))
KV_MIN_SEQ = int(os.environ.get("KV_MIN_SEQ", "4096"))
ATTN_PROFILE = os.environ.get("ATTN_PROFILE", "0") not in ("", "0", "false", "False")
_kv_active = True

_stats = Counter()


def _sync_mps():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def _perfect_square(n):
    if n <= 0:
        return None
    side = int(n**0.5)
    return side if side * side == n else None


def _infer_text_and_side(seq_len):
    # FLUX.2-klein pads/truncates text to 512 tokens in the observed pipeline.
    for text_len in (512, 256, 1024, 0):
        side = _perfect_square(seq_len - text_len)
        if side is not None:
            return text_len, side

    for text_len in range(0, min(2048, seq_len) + 1):
        side = _perfect_square(seq_len - text_len)
        if side is not None:
            return text_len, side

    return None, None


def _tile_grid(x, side, stride):
    bsz, heads, _, dim = x.shape
    grid = x.reshape(bsz, heads, side, side, dim)
    grid = grid.reshape(bsz, heads, side // stride, stride, side // stride, stride, dim)
    return grid.permute(0, 1, 2, 4, 3, 5, 6).reshape(
        bsz, heads, -1, stride * stride, dim
    )


def _pool_grid(x, side, stride):
    return _tile_grid(x, side, stride).mean(dim=3)


def _representative_values(img_k, img_v, side, stride):
    if KV_VALUE_MODE == "mean":
        return _pool_grid(img_v, side, stride)

    k_tiles = _tile_grid(img_k, side, stride)
    v_tiles = _tile_grid(img_v, side, stride)

    if KV_VALUE_MODE == "center":
        center = (stride * stride) // 2
        return v_tiles[:, :, :, center, :]

    if KV_VALUE_MODE == "detail":
        score = (k_tiles - k_tiles.mean(dim=3, keepdim=True)).float().square().mean(dim=-1)
        pick = score.argmax(dim=3, keepdim=True).unsqueeze(-1).expand(-1, -1, -1, 1, img_v.shape[-1])
        return v_tiles.gather(3, pick).squeeze(3)

    raise ValueError(f"unknown KV_VALUE_MODE={KV_VALUE_MODE!r}; use mean, center, or detail")


def _adaptive_pool_image_kv(key, value):
    if not _kv_active or KV_STRIDE <= 1 or key.shape[-2] < KV_MIN_SEQ:
        return key, value

    seq_len = key.shape[-2]
    text_len, side = _infer_text_and_side(seq_len)
    if side is None or side % KV_STRIDE != 0:
        _stats["kv_skip_shape"] += 1
        return key, value

    text_k, img_k = key[..., :text_len, :], key[..., text_len:, :]
    text_v, img_v = value[..., :text_len, :], value[..., text_len:, :]

    pooled_k = _pool_grid(img_k, side, KV_STRIDE)
    pooled_v = _representative_values(img_k, img_v, side, KV_STRIDE)

    if KV_KEEP_TILE_FRAC <= 0:
        merged_key = torch.cat([text_k, pooled_k], dim=-2)
        merged_value = torch.cat([text_v, pooled_v], dim=-2)
    else:
        bsz, heads, _, dim = img_k.shape
        tiles_per_side = side // KV_STRIDE
        tokens_per_tile = KV_STRIDE * KV_STRIDE
        tiles = _tile_grid(img_k, side, KV_STRIDE)
        detail = (tiles - tiles.mean(dim=3, keepdim=True)).float().square().mean(dim=(1, 3, 4))
        keep_n = int(round(detail.shape[-1] * KV_KEEP_TILE_FRAC))
        keep_n = max(0, min(keep_n, detail.shape[-1]))

        if bsz != 1 or keep_n == 0:
            merged_key = torch.cat([text_k, pooled_k], dim=-2)
            merged_value = torch.cat([text_v, pooled_v], dim=-2)
        else:
            keep = torch.topk(detail[0], keep_n, sorted=False).indices
            keep_mask = torch.zeros(detail.shape[-1], device=key.device, dtype=torch.bool)
            keep_mask[keep] = True

            k_tiles = _tile_grid(img_k, side, KV_STRIDE)
            v_tiles = _tile_grid(img_v, side, KV_STRIDE)

            keep_k = k_tiles[:, :, keep_mask].reshape(bsz, heads, -1, dim)
            keep_v = v_tiles[:, :, keep_mask].reshape(bsz, heads, -1, dim)
            pooled_k = pooled_k.reshape(bsz, heads, -1, dim)[:, :, ~keep_mask]
            pooled_v = pooled_v.reshape(bsz, heads, -1, dim)[:, :, ~keep_mask]
            merged_key = torch.cat([text_k, keep_k, pooled_k], dim=-2)
            merged_value = torch.cat([text_v, keep_v, pooled_v], dim=-2)

    _stats["kv_calls"] += 1
    _stats[f"kv_{seq_len}_to_{merged_key.shape[-2]}"] += 1
    _stats["kv_tokens_before"] += seq_len
    _stats["kv_tokens_after"] += merged_key.shape[-2]
    return merged_key, merged_value


def _local_exact_global_pooled_attention(query, key, value, scale=None, **kw):
    seq_len = key.shape[-2]
    text_len, side = _infer_text_and_side(seq_len)
    if (
        text_len is None
        or query.shape[-2] != seq_len
        or side % KV_STRIDE != 0
        or KV_LOCAL_RADIUS < 0
        or KV_STRIDE <= 1
        or seq_len < KV_MIN_SEQ
    ):
        return None

    text_k, img_k = key[..., :text_len, :], key[..., text_len:, :]
    text_v, img_v = value[..., :text_len, :], value[..., text_len:, :]
    pooled_k = _pool_grid(img_k, side, KV_STRIDE)
    pooled_v = _representative_values(img_k, img_v, side, KV_STRIDE)

    outs = []

    # Text queries are cheap enough to keep exact, and preserving prompt state
    # avoids text/image drift in later single-stream blocks.
    if text_len:
        outs.append(
            _orig_sdpa(
                query[..., :text_len, :],
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
                **kw,
            )
        )

    if KV_LOCAL_TILE > 0:
        out_img = torch.empty_like(query[..., text_len:, :])
        pooled_side = side // KV_STRIDE
        tile = KV_LOCAL_TILE
        for y in range(0, side, tile):
            y2 = min(y + tile, side)
            for x in range(0, side, tile):
                x2 = min(x + tile, side)

                yy = torch.arange(y, y2, device=query.device)
                xx = torch.arange(x, x2, device=query.device)
                grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
                q_idx = (grid_y * side + grid_x).flatten()

                ly0 = max(0, y - KV_LOCAL_RADIUS)
                ly1 = min(side, y2 + KV_LOCAL_RADIUS)
                lx0 = max(0, x - KV_LOCAL_RADIUS)
                lx1 = min(side, x2 + KV_LOCAL_RADIUS)
                local_y = torch.arange(ly0, ly1, device=query.device)
                local_x = torch.arange(lx0, lx1, device=query.device)
                local_grid_y, local_grid_x = torch.meshgrid(local_y, local_x, indexing="ij")
                local_idx = (local_grid_y * side + local_grid_x).flatten()

                local_k = img_k.index_select(-2, local_idx)
                local_v = img_v.index_select(-2, local_idx)

                pool_y0 = ly0 // KV_STRIDE
                pool_y1 = (ly1 + KV_STRIDE - 1) // KV_STRIDE
                pool_x0 = lx0 // KV_STRIDE
                pool_x1 = (lx1 + KV_STRIDE - 1) // KV_STRIDE
                far_mask = torch.ones((pooled_side, pooled_side), device=key.device, dtype=torch.bool)
                far_mask[pool_y0:pool_y1, pool_x0:pool_x1] = False
                far_mask = far_mask.flatten()
                far_k = pooled_k[..., far_mask, :]
                far_v = pooled_v[..., far_mask, :]

                chunk_key = torch.cat([text_k, local_k, far_k], dim=-2)
                chunk_value = torch.cat([text_v, local_v, far_v], dim=-2)
                _stats["kv_local_calls"] += 1
                _stats["kv_local_tokens_before"] += seq_len
                _stats["kv_local_tokens_after"] += chunk_key.shape[-2]
                _stats[f"kv_local_tile{tile}_{seq_len}_calls"] += 1

                out_chunk = _orig_sdpa(
                    query.index_select(-2, text_len + q_idx),
                    chunk_key,
                    chunk_value,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=scale,
                    **kw,
                )
                out_img[..., q_idx, :] = out_chunk

        outs.append(out_img)
        return torch.cat(outs, dim=-2)

    pooled_side = side // KV_STRIDE
    for q_start in range(text_len, seq_len, QCHUNK):
        q_end = min(q_start + QCHUNK, seq_len)
        img_start = q_start - text_len
        img_end = q_end - text_len

        y0 = max(0, img_start // side - KV_LOCAL_RADIUS)
        y1 = min(side, ((img_end - 1) // side + 1) + KV_LOCAL_RADIUS)
        local_start = y0 * side
        local_end = y1 * side

        local_k = img_k[..., local_start:local_end, :]
        local_v = img_v[..., local_start:local_end, :]

        pool_y0 = y0 // KV_STRIDE
        pool_y1 = (y1 + KV_STRIDE - 1) // KV_STRIDE
        far_mask = torch.ones(pooled_side * pooled_side, device=key.device, dtype=torch.bool)
        far_mask[pool_y0 * pooled_side : pool_y1 * pooled_side] = False
        far_k = pooled_k[..., far_mask, :]
        far_v = pooled_v[..., far_mask, :]

        chunk_key = torch.cat([text_k, local_k, far_k], dim=-2)
        chunk_value = torch.cat([text_v, local_v, far_v], dim=-2)
        _stats["kv_local_calls"] += 1
        _stats["kv_local_tokens_before"] += seq_len
        _stats["kv_local_tokens_after"] += chunk_key.shape[-2]
        _stats[f"kv_local_{seq_len}_calls"] += 1

        outs.append(
            _orig_sdpa(
                query[..., q_start:q_end, :],
                chunk_key,
                chunk_value,
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
                **kw,
            )
        )

    return torch.cat(outs, dim=-2)


def _chunked_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
    if ATTN_PROFILE:
        _sync_mps()
        t0 = time.perf_counter()
        out = _chunked_sdpa_impl(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            **kw,
        )
        _sync_mps()
        dt = time.perf_counter() - t0
        _stats["profiled_sdpa_calls"] += 1
        _stats["profiled_sdpa_s"] += dt
        _stats[f"profiled_lq_{query.shape[-2]}_calls"] += 1
        _stats[f"profiled_lq_{query.shape[-2]}_s"] += dt
        return out

    return _chunked_sdpa_impl(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        **kw,
    )


def _chunked_sdpa_impl(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
    lq = query.shape[-2]
    # Only intercept large, unmasked, non-causal attention (the self-attn that crashes).
    if attn_mask is not None or is_causal or lq <= QCHUNK:
        return _orig_sdpa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            **kw,
        )

    _stats["sdpa_calls"] += 1
    if _kv_active and KV_LOCAL_RADIUS >= 0:
        out = _local_exact_global_pooled_attention(query, key, value, scale=scale, **kw)
        if out is not None:
            return out

    key, value = _adaptive_pool_image_kv(key, value)

    outs = []
    for i in range(0, lq, QCHUNK):
        outs.append(
            _orig_sdpa(
                query[..., i : i + QCHUNK, :],
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
                **kw,
            )
        )
    return torch.cat(outs, dim=-2)


def patch():
    F.scaled_dot_product_attention = _chunked_sdpa
    return {
        "qchunk": QCHUNK,
        "kv_stride": KV_STRIDE,
        "kv_keep_tile_frac": KV_KEEP_TILE_FRAC,
        "kv_value_mode": KV_VALUE_MODE,
        "kv_local_radius": KV_LOCAL_RADIUS,
        "kv_local_tile": KV_LOCAL_TILE,
        "attn_profile": ATTN_PROFILE,
    }


def unpatch():
    F.scaled_dot_product_attention = _orig_sdpa


def set_kv_active(active):
    global _kv_active
    old = _kv_active
    _kv_active = active
    return old


def reset_stats():
    _stats.clear()


def get_stats():
    return dict(_stats)
