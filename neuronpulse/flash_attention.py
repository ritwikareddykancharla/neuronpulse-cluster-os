"""Real Online-Softmax Block-Tiled FlashAttention vs. Naive Materialized N x N Attention.

Implements:
  1. Standard Materialized Scaled Dot-Product Attention: `Softmax((Q @ K^T) / sqrt(d)) @ V`,
     which allocates the full `O(N^2)` score and probability matrices in memory.
  2. Milakov-Gimelshein (2018) / Dao et al. (2022) Online-Softmax Block-Tiled FlashAttention
     (`block_size = 16, 32, 64`), streaming `(B_r x B_c)` tiles with running row-max `m_i` and
     rescaled normalizer `l_i` so the `N x N` attention matrix is NEVER materialized.

Executes both kernels on real IEEE-754 float64 `Q, K, V` tensors, verifies exact numerical
equivalence (`max_abs_diff < 1e-9`), and measures real `tracemalloc` peak intermediate memory
bytes (`O(N)` vs `O(N^2)`) and real `time.perf_counter()` wall-clock execution time (`ms`).
"""
from __future__ import annotations

import gc
import math
import random
import time
import tracemalloc
from typing import Any, Dict, List, Tuple

NEG_INF = float("-inf")
_SUMPROD = math.sumprod


def generate_qkv_tensors(
    seq_len: int = 256,
    head_dim: int = 32,
    seed: int = 2026,
) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    """Generate deterministic real float64 Query, Key, and Value matrices of shape `(N, d)`."""
    n = max(16, min(1024, int(seq_len)))
    d = max(8, min(128, int(head_dim)))
    rng = random.Random(seed + n * 17 + d * 3)
    scale = 1.0 / math.sqrt(d)
    q = [[rng.gauss(0.0, scale) for _ in range(d)] for _ in range(n)]
    k = [[rng.gauss(0.0, scale) for _ in range(d)] for _ in range(n)]
    v = [[rng.gauss(0.0, scale) for _ in range(d)] for _ in range(n)]
    return q, k, v


def naive_materialized_attention(
    q: List[List[float]],
    k: List[List[float]],
    v: List[List[float]],
    out: List[List[float]],
    causal: bool = False,
) -> None:
    """Standard two-pass attention materializing full `N x N` score and softmax matrices."""
    n = len(q)
    d = len(q[0])
    scale = 1.0 / math.sqrt(d)
    exp = math.exp
    sp = _SUMPROD

    # Step 1: Materialize full N x N score matrix S = (Q @ K^T) / sqrt(d)
    scores: List[List[float]] = []
    for i in range(n):
        qi = q[i]
        if causal:
            row_s = [
                (sp(qi, k[j]) * scale) if j <= i else NEG_INF
                for j in range(n)
            ]
        else:
            row_s = [sp(qi, k[j]) * scale for j in range(n)]
        scores.append(row_s)

    # Step 2: Materialize full N x N probability matrix P = Softmax(S)
    probs: List[List[float]] = []
    for i in range(n):
        row_s = scores[i]
        m = max(row_s)
        exps = [exp(s - m) if s != NEG_INF else 0.0 for s in row_s]
        inv_denom = 1.0 / sum(exps)
        probs.append([e * inv_denom for e in exps])

    # Step 3: Output O = P @ V into preallocated output buffer
    v_t = list(zip(*v))
    for i in range(n):
        pi = probs[i]
        out_row = out[i]
        for c in range(d):
            out_row[c] = sp(pi, v_t[c])


def tiled_flash_attention(
    q: List[List[float]],
    k: List[List[float]],
    v: List[List[float]],
    out: List[List[float]],
    block_size: int = 32,
    causal: bool = False,
) -> Dict[str, int]:
    """Online-softmax block-tiled FlashAttention that NEVER materializes the `N x N` matrix."""
    n = len(q)
    d = len(q[0])
    br = max(8, min(n, int(block_size)))
    bc = br
    scale = 1.0 / math.sqrt(d)
    exp = math.exp
    sp = _SUMPROD

    tiles_evaluated = 0
    tiles_skipped_causal = 0

    for i0 in range(0, n, br):
        i1 = min(n, i0 + br)
        cur_br = i1 - i0

        # Tile-local running online-softmax statistics: O(B_r * d) working memory only
        m_i = [NEG_INF] * cur_br
        l_i = [0.0] * cur_br
        acc_i = [[0.0] * d for _ in range(cur_br)]

        for j0 in range(0, n, bc):
            if causal and j0 > (i1 - 1):
                tiles_skipped_causal += 1
                continue

            j1 = min(n, j0 + bc)
            cur_bc = j1 - j0
            tiles_evaluated += 1
            v_tile_t = list(zip(*v[j0:j1]))

            # Score buffer for ONLY this (B_r x B_c) tile
            for r in range(cur_br):
                qi = q[i0 + r]
                global_i = i0 + r
                if causal and j1 - 1 > global_i:
                    s_row = [
                        (sp(qi, k[j0 + c_idx]) * scale) if (j0 + c_idx) <= global_i else NEG_INF
                        for c_idx in range(cur_bc)
                    ]
                else:
                    s_row = [sp(qi, k[j0 + c_idx]) * scale for c_idx in range(cur_bc)]

                m_prev = m_i[r]
                m_tile = max(s_row)
                m_new = m_prev if m_prev >= m_tile else m_tile

                alpha = exp(m_prev - m_new) if m_prev != NEG_INF else 0.0
                p_row = [exp(s_val - m_new) if s_val != NEG_INF else 0.0 for s_val in s_row]
                row_sum_p = sum(p_row)

                acc_row = acc_i[r]
                if alpha == 0.0:
                    for col in range(d):
                        acc_row[col] = sp(p_row, v_tile_t[col])
                else:
                    for col in range(d):
                        acc_row[col] = acc_row[col] * alpha + sp(p_row, v_tile_t[col])

                m_i[r] = m_new
                l_i[r] = alpha * l_i[r] + row_sum_p

        # Finalize normalized output rows for this query block
        for r in range(cur_br):
            inv_l = 1.0 / l_i[r]
            acc_row = acc_i[r]
            out_row = out[i0 + r]
            for col in range(d):
                out_row[col] = acc_row[col] * inv_l

    return {
        "tiles_evaluated": tiles_evaluated,
        "tiles_skipped_causal": tiles_skipped_causal,
    }


def _max_abs_diff(a: List[List[float]], b: List[List[float]]) -> float:
    max_err = 0.0
    for ra, rb in zip(a, b):
        for va, vb in zip(ra, rb):
            err = abs(va - vb)
            if err > max_err:
                max_err = err
    return max_err


def benchmark_attention_pair(
    seq_len: int = 256,
    head_dim: int = 32,
    block_size: int = 32,
    causal: bool = False,
) -> Dict[str, Any]:
    """Execute both Naive O(N^2) and Tiled FlashAttention and measure real peak memory & latency."""
    n = max(16, min(768, int(seq_len)))
    d = max(8, min(128, int(head_dim)))
    bs = max(8, min(n, int(block_size)))
    if bs not in (8, 16, 32, 64, 128):
        # Snap to clean tile size if arbitrary input passed
        bs = max(8, min(128, (bs // 8) * 8))

    # Allocate inputs and output buffers OUTSIDE tracemalloc window so peak_bytes
    # strictly measures attention kernel working/intermediate allocations.
    q, k, v = generate_qkv_tensors(seq_len=n, head_dim=d)
    out_naive = [[0.0] * d for _ in range(n)]
    out_flash = [[0.0] * d for _ in range(n)]

    # 1. Measure real peak heap allocation bytes under tracemalloc
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()

    gc.collect()
    tracemalloc.reset_peak()
    naive_materialized_attention(q, k, v, out_naive, causal=causal)
    _, naive_peak_bytes = tracemalloc.get_traced_memory()

    gc.collect()
    tracemalloc.reset_peak()
    tile_stats = tiled_flash_attention(q, k, v, out_flash, block_size=bs, causal=causal)
    _, flash_peak_bytes = tracemalloc.get_traced_memory()

    if not was_tracing:
        tracemalloc.stop()

    # 2. Measure clean uninstrumented wall-clock execution time (ms)
    t0 = time.perf_counter()
    naive_materialized_attention(q, k, v, out_naive, causal=causal)
    naive_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    t1 = time.perf_counter()
    tile_stats = tiled_flash_attention(q, k, v, out_flash, block_size=bs, causal=causal)
    flash_ms = round((time.perf_counter() - t1) * 1000.0, 3)

    max_diff = _max_abs_diff(out_naive, out_flash)
    mem_reduction_pct = round(
        ((naive_peak_bytes - flash_peak_bytes) / max(1, naive_peak_bytes)) * 100.0,
        2,
    )
    mem_compression_x = round(naive_peak_bytes / max(1, flash_peak_bytes), 2)

    return {
        "seq_len": n,
        "head_dim": d,
        "block_size": bs,
        "causal": bool(causal),
        "naive_score_elements": n * n,
        "flash_tile_elements": bs * bs,
        "tiles_evaluated": tile_stats["tiles_evaluated"],
        "tiles_skipped_causal": tile_stats["tiles_skipped_causal"],
        "naive_peak_bytes": int(naive_peak_bytes),
        "flash_peak_bytes": int(flash_peak_bytes),
        "memory_saved_bytes": int(naive_peak_bytes - flash_peak_bytes),
        "memory_reduction_pct": mem_reduction_pct,
        "memory_compression_x": mem_compression_x,
        "naive_wall_ms": naive_ms,
        "flash_wall_ms": flash_ms,
        "max_abs_diff": max_diff,
        "numerically_exact": bool(max_diff < 1e-9),
        "sample_output_row0": [round(x, 6) for x in out_flash[0][:6]],
    }


def run_attention_scaling_sweep(
    seq_lengths: Tuple[int, ...] = (128, 256, 512),
    head_dim: int = 32,
    block_size: int = 32,
    causal: bool = False,
) -> Dict[str, Any]:
    """Run real scaling sweep across N = 128, 256, 512 proving O(N) vs O(N^2) peak memory."""
    rows = [
        benchmark_attention_pair(
            seq_len=n,
            head_dim=head_dim,
            block_size=block_size,
            causal=causal,
        )
        for n in seq_lengths
    ]
    largest = rows[-1]
    smallest = rows[0]
    naive_growth_x = round(largest["naive_peak_bytes"] / max(1, smallest["naive_peak_bytes"]), 2)
    flash_growth_x = round(largest["flash_peak_bytes"] / max(1, smallest["flash_peak_bytes"]), 2)

    return {
        "head_dim": head_dim,
        "block_size": block_size,
        "causal": causal,
        "sweep": rows,
        "n512_summary": largest,
        "scaling_proof": {
            "seq_len_ratio": f"{smallest['seq_len']} -> {largest['seq_len']} ({largest['seq_len'] // smallest['seq_len']}x)",
            "naive_memory_growth_x": naive_growth_x,
            "flash_memory_growth_x": flash_growth_x,
            "max_abs_diff_across_sweep": max(r["max_abs_diff"] for r in rows),
        },
    }
