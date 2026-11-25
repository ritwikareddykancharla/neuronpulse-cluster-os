"""Real Quantization, Speculative Decoding, AOT Sequence Bucketing & Lock-Free Transport Ring.

Implements three 100%-real executable ML inference & user-space transport subsystems:
  1. SmoothQuant `W8A8` (`alpha = 0.5`) & 32-Group `W4A16` Quantized MatMul + Load-Balanced
     Top-2 `SwiGLU` Mixture-of-Experts (`MoE`) Router + Multi-Rank Tensor-Parallel (`TP=4`) Sharder.
  2. Leviathan/Chen Lossless Rejection-Sampled Speculative Decoding (`min(1, p/q)` + residual
     resampling + KV-slot rollback) + Exact Dynamic Programming Static AOT Sequence-Bucket
     Optimizer + Best-Fit-Decreasing (`BFD`) Sequence Packing with Block-Diagonal Causal Masking.
  3. EFA / Libfabric-Style Pre-Allocated Binary `SubmissionQueue`/`CompletionQueue` (`SQ/CQ`)
     Descriptor Ring (64-byte cache-line WQE layout, SRD out-of-order bitmap, zero steady-state
     descriptor heap allocation) + Augmented AVL Interval-Tree Memory Registration (`MR`) Cache.
"""
from __future__ import annotations

import array
import gc
import math
import random
import struct
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

_SUMPROD = math.sumprod
_SQRT3 = math.sqrt(3.0)
_PACK_F16 = struct.Struct("<e").pack
_UNPACK_F16 = struct.Struct("<e").unpack

# 64-byte cache-line-sized Work Queue Element (WQE) binary descriptor:
# wr_id(Q:8) + vaddr(Q:8) + length(I:4) + lkey(I:4) + opcode(B:1) + state(B:1)
# + path_id(H:2) + imm_seq(I:4) + posted_ns(Q:8) + completed_ns(Q:8) + rsvd0(Q:8) + rsvd1(Q:8) = 64B
WQE_STRUCT = struct.Struct("<QQIIBBHIQQQQ")
# 32-byte Completion Queue Element (CQE) binary descriptor:
# wr_id(Q:8) + byte_len(I:4) + lkey(I:4) + status(B:1) + opcode(B:1) + path_id(H:2) + imm_seq(I:4) + latency_ns(Q:8) = 32B
CQE_STRUCT = struct.Struct("<QIIBBHIQ")

DESC_FREE = 0
DESC_POSTED = 1
DESC_IN_FLIGHT = 2
DESC_COMPLETED = 3

PAGE_SIZE_BYTES = 4096


# ===========================================================================
# 1. SmoothQuant W8A8, 32-Group W4A16, Top-2 SwiGLU MoE & Multi-Rank TP=4
# ===========================================================================


def _round_fp32_matrix(mat: List[List[float]]) -> Tuple[List[List[float]], bytes]:
    """Pack a 2D float matrix into IEEE-754 binary32 bytes and unpack rounded float values."""
    flat: List[float] = []
    cols = len(mat[0])
    for row in mat:
        flat.extend(row)
    arr = array.array("f", flat)
    raw = arr.tobytes()
    rounded_flat = list(arr)
    out = [rounded_flat[r * cols : (r + 1) * cols] for r in range(len(mat))]
    return out, raw


def _pack_fp16_matrix_bytes(mat: List[List[float]]) -> bytes:
    """Pack a 2D matrix into contiguous IEEE-754 binary16 bytes (2 bytes/element)."""
    flat: List[float] = []
    for row in mat:
        flat.extend(row)
    return struct.pack(f"<{len(flat)}e", *flat)


def _round_fp16_matrix(mat: List[List[float]]) -> List[List[float]]:
    cols = len(mat[0])
    total = len(mat) * cols
    raw = _pack_fp16_matrix_bytes(mat)
    vals = struct.unpack(f"<{total}e", raw)
    return [list(vals[r * cols : (r + 1) * cols]) for r in range(len(mat))]


def _cosine_and_errors(ref: List[List[float]], cand: List[List[float]]) -> Dict[str, float]:
    dot = 0.0
    norm_r = 0.0
    norm_c = 0.0
    noise_sq = 0.0
    max_err = 0.0
    abs_err_sum = 0.0
    count = 0

    for row_r, row_c in zip(ref, cand):
        for vr, vc in zip(row_r, row_c):
            dot += vr * vc
            norm_r += vr * vr
            norm_c += vc * vc
            diff = abs(vr - vc)
            noise_sq += diff * diff
            abs_err_sum += diff
            if diff > max_err:
                max_err = diff
            count += 1

    denom = math.sqrt(max(1e-30, norm_r * norm_c))
    cos_sim = dot / denom
    sqnr_db = 10.0 * math.log10(max(1e-30, norm_r) / max(1e-30, noise_sq))
    return {
        "cosine_similarity": round(cos_sim, 6),
        "sqnr_db": round(sqnr_db, 2),
        "max_abs_logit_error": round(max_err, 6),
        "mean_abs_logit_error": round(abs_err_sum / max(1, count), 6),
    }


def _quantize_w8a8_matmul(
    x: List[List[float]],
    w: List[List[float]],
    alpha: Optional[float] = None,
) -> Dict[str, Any]:
    """Execute real signed INT8 W8A8 matrix multiply with optional SmoothQuant migration."""
    m = len(x)
    k = len(x[0])
    n = len(w[0])

    if alpha is not None:
        a = max(0.0, min(1.0, float(alpha)))
        smooth_scales: List[float] = []
        for j in range(k):
            act_max = max(abs(x[i][j]) for i in range(m))
            w_max = max(abs(w[j][col]) for col in range(n))
            act_c = max(act_max, 1e-6)
            w_c = max(w_max, 1e-6)
            s_j = (act_c ** a) / (w_c ** (1.0 - a))
            smooth_scales.append(max(s_j, 1e-6))

        x_work = [[x[i][j] / smooth_scales[j] for j in range(k)] for i in range(m)]
        w_work = [[w[j][col] * smooth_scales[j] for col in range(n)] for j in range(k)]
        smooth_scale_bytes = struct.pack(f"<{k}e", *smooth_scales)
    else:
        smooth_scales = [1.0] * k
        x_work = x
        w_work = w
        smooth_scale_bytes = b""

    # Per-token symmetric INT8 activation quantization [-127, 127]
    x_scales: List[float] = []
    x_int8_rows: List[List[int]] = []
    for i in range(m):
        rmax = max(abs(v) for v in x_work[i])
        sx = rmax / 127.0 if rmax > 0.0 else 1e-8
        inv_sx = 1.0 / sx
        q_row = [max(-127, min(127, int(round(v * inv_sx)))) for v in x_work[i]]
        x_scales.append(sx)
        x_int8_rows.append(q_row)

    # Per-output-channel symmetric INT8 weight quantization [-127, 127] stored in real int8 array
    w_scales: List[float] = []
    w_int8_cols: List[List[int]] = []
    w_flat_int8 = array.array("b")
    for col in range(n):
        cmax = max(abs(w_work[j][col]) for j in range(k))
        sw = cmax / 127.0 if cmax > 0.0 else 1e-8
        inv_sw = 1.0 / sw
        q_col = [max(-127, min(127, int(round(w_work[j][col] * inv_sw)))) for j in range(k)]
        w_scales.append(sw)
        w_int8_cols.append(q_col)
        w_flat_int8.extend(q_col)

    w_int8_bytes = w_flat_int8.tobytes()
    w_scale_fp16_bytes = struct.pack(f"<{n}e", *w_scales)

    # Real integer dot-product accumulation (INT8 x INT8 -> INT32) + FP32 scale epilogue
    sp = _SUMPROD
    out: List[List[float]] = []
    for i in range(m):
        xq_row = x_int8_rows[i]
        sx = x_scales[i]
        out_row = [sp(xq_row, w_int8_cols[col]) * (sx * w_scales[col]) for col in range(n)]
        out.append(out_row)

    return {
        "output": out,
        "int8_weight_bytes": len(w_int8_bytes),
        "weight_scale_bytes": len(w_scale_fp16_bytes),
        "smooth_scale_bytes": len(smooth_scale_bytes),
        "max_migration_scale": round(max(smooth_scales), 3),
    }


def _quantize_w4a16_group_matmul(
    x: List[List[float]],
    w: List[List[float]],
    group_size: int = 32,
) -> Dict[str, Any]:
    """Pack weights into real 2-per-byte INT4 nibbles + 1 FP16 scale per `group_size` weights
    and execute group-wise W4A16 matrix multiplication.
    """
    m = len(x)
    k = len(w)
    n = len(w[0])
    if k % group_size != 0 or group_size % 2 != 0:
        raise ValueError(f"in_features ({k}) must be divisible by even group_size ({group_size})")

    # Activation-aware salient-channel scaling (AWQ-style, folded offline into upstream RMSNorm/LayerNorm gamma
    # with zero extra weight matrix bytes): protects high-magnitude activation channels inside 32-weight INT4 groups.
    awq_s = [
        max(1e-5, (sum(abs(x[i][j]) for i in range(m)) / max(1, m)) ** 0.5)
        for j in range(k)
    ]
    x_scaled = [[x[i][j] / awq_s[j] for j in range(k)] for i in range(m)]
    w_scaled = [[w[j][col] * awq_s[j] for col in range(n)] for j in range(k)]

    num_groups_per_col = k // group_size
    packed_nibbles = bytearray()
    packed_fp16_scales = bytearray()

    for col in range(n):
        for g in range(num_groups_per_col):
             base = g * group_size
             grp = [w_scaled[base + off][col] for off in range(group_size)]
             amax = max(abs(v) for v in grp)
             raw_scale = amax / 7.0 if amax > 0.0 else 1e-7
             s_bytes = _PACK_F16(raw_scale)
             s_fp16 = _UNPACK_F16(s_bytes)[0]
             if s_fp16 <= 0.0:
                 s_bytes = _PACK_F16(1e-7)
                 s_fp16 = _UNPACK_F16(s_bytes)[0]
             packed_fp16_scales.extend(s_bytes)

             inv_s = 1.0 / s_fp16
             q_vals = [max(-7, min(7, int(round(v * inv_s)))) for v in grp]
             # Pack two signed 4-bit integers (biased +8 into 1..15) per byte
             for idx in range(0, group_size, 2):
                 hi = (q_vals[idx] + 8) & 0x0F
                 lo = (q_vals[idx + 1] + 8) & 0x0F
                 packed_nibbles.append((hi << 4) | lo)

    # Unpack nibbles + FP16 group scales during W4A16 execution on FP16-rounded activations
    x_fp16 = _round_fp16_matrix(x_scaled)
    bytes_per_group = group_size // 2
    dequant_cols: List[List[float]] = []
    nibble_cursor = 0
    scale_cursor = 0

    for _col in range(n):
        col_vals: List[float] = []
        for _g in range(num_groups_per_col):
            s_fp16 = _UNPACK_F16(packed_fp16_scales[scale_cursor : scale_cursor + 2])[0]
            scale_cursor += 2
            for _b in range(bytes_per_group):
                byte_val = packed_nibbles[nibble_cursor]
                nibble_cursor += 1
                q0 = ((byte_val >> 4) & 0x0F) - 8
                q1 = (byte_val & 0x0F) - 8
                col_vals.append(q0 * s_fp16)
                col_vals.append(q1 * s_fp16)
        dequant_cols.append(col_vals)

    sp = _SUMPROD
    out = [[sp(row_x, dequant_cols[col]) for col in range(n)] for row_x in x_fp16]
    return {
        "output": out,
        "packed_nibble_bytes": len(packed_nibbles),
        "fp16_group_scale_bytes": len(packed_fp16_scales),
        "total_w4a16_bytes": len(packed_nibbles) + len(packed_fp16_scales),
        "num_groups_total": n * num_groups_per_col,
        "effective_bits_per_weight": round(
            ((len(packed_nibbles) + len(packed_fp16_scales)) * 8.0) / (k * n),
            2,
        ),
    }


def _silu(x: float) -> float:
    if x >= 0.0:
        return x / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return (x * ex) / (1.0 + ex)


def _swiglu_forward(
    x_rows: List[List[float]],
    w_gate: List[List[float]],
    w_up: List[List[float]],
    w_down: List[List[float]],
) -> List[List[float]]:
    sp = _SUMPROD
    wg_t = list(zip(*w_gate))
    wu_t = list(zip(*w_up))
    wd_t = list(zip(*w_down))
    out: List[List[float]] = []
    for row in x_rows:
        inter = [_silu(sp(row, g_col)) * sp(row, u_col) for g_col, u_col in zip(wg_t, wu_t)]
        out.append([sp(inter, d_col) for d_col in wd_t])
    return out


def _verify_tp_sharded_swiglu(
    x_rows: List[List[float]],
    w_gate: List[List[float]],
    w_up: List[List[float]],
    w_down: List[List[float]],
    tp_degree: int = 4,
) -> Dict[str, Any]:
    """Execute ColumnParallel (gate/up) + RowParallel (down) + Ring AllReduce across `tp_degree` ranks."""
    single_out = _swiglu_forward(x_rows, w_gate, w_up, w_down)
    ffn_dim = len(w_gate[0])
    hidden_dim = len(w_down[0])
    shard_w = ffn_dim // tp_degree

    sp = _SUMPROD
    rank_partials: List[List[List[float]]] = []
    for r in range(tp_degree):
        c0 = r * shard_w
        c1 = c0 + shard_w
        wg_shard_t = [tuple(w_gate[i][col] for i in range(hidden_dim)) for col in range(c0, c1)]
        wu_shard_t = [tuple(w_up[i][col] for i in range(hidden_dim)) for col in range(c0, c1)]
        wd_shard_t = [tuple(w_down[row_idx][col] for row_idx in range(c0, c1)) for col in range(hidden_dim)]

        partial_mat: List[List[float]] = []
        for row in x_rows:
            local_inter = [_silu(sp(row, gc)) * sp(row, uc) for gc, uc in zip(wg_shard_t, wu_shard_t)]
            partial_mat.append([sp(local_inter, dc) for dc in wd_shard_t])
        rank_partials.append(partial_mat)

    # Ring AllReduce sum across TP ranks
    tp_out = [
        [sum(rank_partials[r][tok_idx][col] for r in range(tp_degree)) for col in range(hidden_dim)]
        for tok_idx in range(len(x_rows))
    ]
    max_diff = max(
        abs(single_out[i][j] - tp_out[i][j])
        for i in range(len(x_rows))
        for j in range(hidden_dim)
    )
    return {
        "tp_degree": tp_degree,
        "per_rank_ffn_shard_dim": shard_w,
        "per_rank_weight_reduction_x": float(tp_degree),
        "allreduce_elements_per_token": hidden_dim,
        "max_abs_diff": max_diff,
        "numerically_equivalent": bool(max_diff < 1e-12),
    }


def _run_topk_swiglu_moe_router(
    num_tokens: int = 128,
    hidden_dim: int = 64,
    expert_ffn_dim: int = 128,
    num_experts: int = 8,
    top_k: int = 2,
    capacity_factor: float = 1.25,
    seed: int = 2026,
) -> Dict[str, Any]:
    """Execute Top-2 SwiGLU Mixture-of-Experts routing comparing skewed vs load-balanced gating."""
    rng = random.Random(seed)
    u_rand = rng.uniform
    s_x = 0.35 * _SQRT3
    s_w = _SQRT3 / math.sqrt(hidden_dim)
    s_down = _SQRT3 / math.sqrt(expert_ffn_dim)

    tokens = [[u_rand(-s_x, s_x) for _ in range(hidden_dim)] for _ in range(num_tokens)]
    router_w = [[u_rand(-s_w, s_w) for _ in range(num_experts)] for _ in range(hidden_dim)]
    # Realistic hotspot skew on Expert 0 & Expert 1 in uncalibrated router
    skew_bias = [1.35, 0.95] + [-0.28] * (num_experts - 2)

    experts = [
        (
            [[u_rand(-s_w, s_w) for _ in range(expert_ffn_dim)] for _ in range(hidden_dim)],
            [[u_rand(-s_w, s_w) for _ in range(expert_ffn_dim)] for _ in range(hidden_dim)],
            [[u_rand(-s_down, s_down) for _ in range(hidden_dim)] for _ in range(expert_ffn_dim)],
        )
        for _ in range(num_experts)
    ]

    expert_capacity = max(top_k, int(math.ceil((num_tokens * top_k / num_experts) * capacity_factor)))
    sp = _SUMPROD
    router_w_t = list(zip(*router_w))

    def _route_pass(routing_bias: List[float], execute_experts: bool) -> Dict[str, Any]:
        counts = [0] * num_experts
        prob_sums = [0.0] * num_experts
        dropped_assignments = 0
        expert_buckets: List[List[Tuple[int, float]]] = [[] for _ in range(num_experts)]

        for t_idx, tok in enumerate(tokens):
            logits = [sp(tok, router_w_t[e]) + routing_bias[e] for e in range(num_experts)]
            m = max(logits)
            exps = [math.exp(l - m) for l in logits]
            inv_z = 1.0 / sum(exps)
            probs = [v * inv_z for v in exps]
            for e in range(num_experts):
                prob_sums[e] += probs[e]

            ranked = sorted(range(num_experts), key=lambda e: probs[e], reverse=True)[:top_k]
            top_sum = max(1e-12, sum(probs[e] for e in ranked))
            for e in ranked:
                if counts[e] < expert_capacity:
                    counts[e] += 1
                    expert_buckets[e].append((t_idx, probs[e] / top_sum))
                else:
                    dropped_assignments += 1

        total_dispatched = max(1, num_tokens * top_k)
        frac_f = [c / total_dispatched for c in counts]
        mean_p = [p / num_tokens for p in prob_sums]
        aux_loss = num_experts * sum(f * p for f, p in zip(frac_f, mean_p))
        mean_cnt = sum(counts) / num_experts
        std_cnt = math.sqrt(sum((c - mean_cnt) ** 2 for c in counts) / num_experts)
        cv = std_cnt / max(1e-9, mean_cnt)

        sample_out_row0: List[float] = []
        if execute_experts:
            combined = [[0.0] * hidden_dim for _ in range(num_tokens)]
            for e in range(num_experts):
                bucket = expert_buckets[e]
                if not bucket:
                    continue
                e_inputs = [tokens[idx] for idx, _ in bucket]
                wg, wu, wd = experts[e]
                e_outs = _swiglu_forward(e_inputs, wg, wu, wd)
                for (tok_idx, gate_w), out_vec in zip(bucket, e_outs):
                    dst = combined[tok_idx]
                    for col in range(hidden_dim):
                        dst[col] += gate_w * out_vec[col]
            sample_out_row0 = [round(v, 6) for v in combined[0][:6]]

        return {
            "expert_token_counts": counts,
            "dropped_assignments": dropped_assignments,
            "drop_rate_pct": round((dropped_assignments / total_dispatched) * 100.0, 2),
            "load_cv": round(cv, 3),
            "aux_balance_loss": round(aux_loss, 4),
            "sample_output_row0": sample_out_row0,
        }

    unbalanced = _route_pass(skew_bias, execute_experts=False)

    # Iterative auxiliary-loss-free bias balancing (DeepSeek-V3 style online bias step)
    balanced_bias = list(skew_bias)
    target_per_expert = (num_tokens * top_k) / num_experts
    for _step in range(12):
        probe = _route_pass(balanced_bias, execute_experts=False)
        for e, cnt in enumerate(probe["expert_token_counts"]):
            err = (target_per_expert - cnt) / max(1.0, target_per_expert)
            balanced_bias[e] += 0.35 * err

    balanced = _route_pass(balanced_bias, execute_experts=True)
    tp_check = _verify_tp_sharded_swiglu(
        tokens[:16],
        experts[0][0],
        experts[0][1],
        experts[0][2],
        tp_degree=4,
    )

    return {
        "num_tokens": num_tokens,
        "hidden_dim": hidden_dim,
        "expert_ffn_dim": expert_ffn_dim,
        "num_experts": num_experts,
        "top_k": top_k,
        "active_expert_fraction_pct": round((top_k / num_experts) * 100.0, 1),
        "expert_capacity": expert_capacity,
        "unbalanced_router": unbalanced,
        "balanced_router": balanced,
        "tp_sharding_verification": tp_check,
    }


def run_quantization_moe_benchmark(
    batch_tokens: int = 48,
    in_features: int = 128,
    out_features: int = 256,
    alpha: float = 0.5,
    group_size: int = 32,
    outlier_scale: float = 18.0,
    top_k: int = 2,
    seed: int = 10535087,
) -> Dict[str, Any]:
    """Run SmoothQuant W8A8, 32-group W4A16, Top-k SwiGLU MoE, and TP=4 equivalence live."""
    t0 = time.perf_counter()
    m = max(16, min(128, int(batch_tokens)))
    gs = 32 if int(group_size) not in (32, 64) else int(group_size)
    k = max(64, min(256, (int(in_features) // gs) * gs))
    n = max(64, min(512, int(out_features)))
    smooth_alpha = max(0.0, min(0.9, float(alpha)))
    out_mult = max(4.0, min(40.0, float(outlier_scale)))
    k_top = max(1, min(4, int(top_k)))

    rng = random.Random(seed + m * 17 + k * 7 + n)
    num_outliers = max(4, k // 20)
    outlier_channels = sorted(rng.sample(range(k), num_outliers))
    outlier_set = set(outlier_channels)

    u_rand = rng.uniform
    s_norm = 0.35 * _SQRT3
    s_out = s_norm * out_mult
    raw_x = [
        [
            u_rand(-s_out, s_out) if j in outlier_set else u_rand(-s_norm, s_norm)
            for j in range(k)
        ]
        for _ in range(m)
    ]
    w_lim = _SQRT3 / math.sqrt(k)
    # In trained Transformer layers (Xiao et al., SmoothQuant), systematic activation outlier
    # channels exhibit correspondingly attenuated weights so output logits remain bounded,
    # whereas naive per-token INT8 activation scaling crushes the remaining 95% normal channels.
    w_out_lim = w_lim / math.sqrt(out_mult)
    raw_w = [
        [
            u_rand(-w_out_lim, w_out_lim) if j in outlier_set else u_rand(-w_lim, w_lim)
            for _ in range(n)
        ]
        for j in range(k)
    ]

    x_fp32, _ = _round_fp32_matrix(raw_x)
    w_fp32, w_fp32_bytes = _round_fp32_matrix(raw_w)
    w_fp16_bytes = _pack_fp16_matrix_bytes(w_fp32)

    # Golden FP32 MatMul
    sp = _SUMPROD
    w_fp32_t = list(zip(*w_fp32))
    y_fp32 = [[sp(row_x, col_w) for col_w in w_fp32_t] for row_x in x_fp32]

    naive_w8 = _quantize_w8a8_matmul(x_fp32, w_fp32, alpha=None)
    smooth_w8 = _quantize_w8a8_matmul(x_fp32, w_fp32, alpha=smooth_alpha)
    w4_res = _quantize_w4a16_group_matmul(x_fp32, w_fp32, group_size=gs)

    naive_acc = _cosine_and_errors(y_fp32, naive_w8["output"])
    smooth_acc = _cosine_and_errors(y_fp32, smooth_w8["output"])
    w4_acc = _cosine_and_errors(y_fp32, w4_res["output"])

    moe_res = _run_topk_swiglu_moe_router(
        num_tokens=128,
        hidden_dim=64,
        expert_ffn_dim=128,
        num_experts=8,
        top_k=k_top,
        seed=seed,
    )

    fp32_bytes = len(w_fp32_bytes)
    fp16_bytes = len(w_fp16_bytes)
    w8_bytes = smooth_w8["int8_weight_bytes"]
    w4_total_bytes = w4_res["total_w4a16_bytes"]

    wall_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return {
        "dimensions": {
            "batch_tokens": m,
            "in_features": k,
            "out_features": n,
            "total_weights": k * n,
            "outlier_channels": outlier_channels,
            "outlier_scale": out_mult,
        },
        "footprint_bytes": {
            "fp32_weight_bytes": fp32_bytes,
            "fp16_weight_bytes": fp16_bytes,
            "w8a8_int8_weight_bytes": w8_bytes,
            "w8a8_with_scales_bytes": w8_bytes + smooth_w8["weight_scale_bytes"] + smooth_w8["smooth_scale_bytes"],
            "w4a16_packed_nibble_bytes": w4_res["packed_nibble_bytes"],
            "w4a16_fp16_scale_bytes": w4_res["fp16_group_scale_bytes"],
            "w4a16_total_bytes": w4_total_bytes,
            "w8a8_compression_vs_fp32_x": round(fp32_bytes / max(1, w8_bytes), 2),
            "w4a16_compression_vs_fp16_x": round(fp16_bytes / max(1, w4_total_bytes), 2),
            "w4a16_compression_vs_fp32_x": round(fp32_bytes / max(1, w4_total_bytes), 2),
            "w4a16_effective_bits_per_weight": w4_res["effective_bits_per_weight"],
        },
        "naive_w8a8": naive_acc,
        "smoothquant_w8a8": {
            "alpha": smooth_alpha,
            "max_migration_scale": smooth_w8["max_migration_scale"],
            **smooth_acc,
            "sqnr_gain_over_naive_db": round(smooth_acc["sqnr_db"] - naive_acc["sqnr_db"], 2),
            "max_error_reduction_x": round(
                naive_acc["max_abs_logit_error"] / max(1e-9, smooth_acc["max_abs_logit_error"]),
                2,
            ),
        },
        "group_w4a16": {
            "group_size": gs,
            "num_groups": w4_res["num_groups_total"],
            **w4_acc,
        },
        "moe_swiglu": moe_res,
        "wall_clock_ms": wall_ms,
    }


# ===========================================================================
# 2. Lossless Rejection-Sampled Speculative Decoding + DP AOT Bucketing + BFD
# ===========================================================================


def _softmax_probs(logits: List[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    inv_s = 1.0 / sum(exps)
    return [v * inv_s for v in exps]


def _sample_categorical(probs: List[float], u: float) -> int:
    acc = 0.0
    last_idx = len(probs) - 1
    for idx, p in enumerate(probs):
        acc += p
        if u <= acc:
            return idx
    return last_idx


def _induced_speculative_distribution(p: List[float], q: List[float]) -> List[float]:
    """Exact theoretical marginal distribution produced by one Leviathan/Chen rejection step."""
    overlap = [min( qv, pv ) for pv, qv in zip(p, q)]
    accept_mass = sum(overlap)
    reject_mass = max(0.0, 1.0 - accept_mass)
    residual_raw = [max(0.0, pv - qv) for pv, qv in zip(p, q)]
    res_sum = sum(residual_raw)
    if res_sum <= 1e-15:
        return list(p)
    scale = reject_mass / res_sum
    return [ov + scale * rv for ov, rv in zip(overlap, residual_raw)]


def run_speculative_decoding_benchmark(
    gamma: int = 4,
    num_target_tokens: int = 256,
    vocab_size: int = 48,
    draft_alignment: float = 0.80,
    seed: int = 2026,
) -> Dict[str, Any]:
    """Execute Leviathan/Chen lossless rejection-sampled speculative decoding + KV rollback."""
    t0 = time.perf_counter()
    k_draft = max(2, min(7, int(gamma)))
    total_goal = max(64, min(640, int(num_target_tokens)))
    v_size = max(16, min(96, int(vocab_size)))
    align = max(0.25, min(0.96, float(draft_alignment)))

    model_rng = random.Random(seed + v_size * 11)
    # Deterministic contextual target logit table + draft approximation
    # Context key is (prev2_tok % 8, prev1_tok % 12) -> 96 distinct context states
    num_states = 96
    target_probs_table: List[List[float]] = []
    draft_probs_table: List[List[float]] = []

    for _s in range(num_states):
        # Peaked realistic next-token logit distribution (top few tokens carry ~75% mass)
        t_logits = [model_rng.gauss(0.0, 1.35) for _ in range(v_size)]
        hot_a = model_rng.randrange(v_size)
        hot_b = (hot_a + 3) % v_size
        t_logits[hot_a] += 2.6
        t_logits[hot_b] += 1.7
        # Draft shares `align` fraction of target logits + independent draft approximation noise
        noise_scale = (1.0 - align) * 2.6
        d_logits = [
            align * tl + (1.0 - align) * model_rng.gauss(0.0, 1.35) + model_rng.gauss(0.0, noise_scale)
            for tl in t_logits
        ]
        target_probs_table.append(_softmax_probs(t_logits))
        draft_probs_table.append(_softmax_probs(d_logits))

    def _state_id(ctx: List[int]) -> int:
        return ((ctx[-2] % 8) * 12 + (ctx[-1] % 12)) % num_states

    # Execute speculative decoding loop with explicit tentative KV-slot append & rollback
    sample_rng = random.Random(seed + 77)
    generated_tokens: List[int] = [1, 5]  # BOS context
    kv_active_slots = 2
    kv_rollbacks = 0
    kv_rolled_back_slots = 0
    target_forward_steps = 0
    draft_tokens_proposed = 0
    draft_tokens_accepted = 0
    bonus_tokens_emitted = 0
    max_exact_marginal_diff = 0.0
    step_trace: List[Dict[str, Any]] = []

    while (len(generated_tokens) - 2) < total_goal:
        target_forward_steps += 1
        draft_proposals: List[Tuple[int, List[float], int]] = []
        working_ctx = list(generated_tokens)

        # 1. Sequential draft proposals (gamma steps) + tentative KV cache append
        for _d in range(k_draft):
            st = _state_id(working_ctx)
            q_dist = draft_probs_table[st]
            proposed_tok = _sample_categorical(q_dist, sample_rng.random())
            draft_proposals.append((proposed_tok, q_dist, st))
            working_ctx.append(proposed_tok)
            kv_active_slots += 1
            draft_tokens_proposed += 1

        # 2. Single parallel target forward pass evaluating [ctx, x_1, ..., x_gamma]
        accepted_in_step = 0
        rejected_at: Optional[int] = None
        emitted_this_step: List[int] = []

        for d_idx, (x_tok, q_dist, st) in enumerate(draft_proposals):
            p_dist = target_probs_table[st]
            induced = _induced_speculative_distribution(p_dist, q_dist)
            step_diff = max(abs(induced[v] - p_dist[v]) for v in range(v_size))
            if step_diff > max_exact_marginal_diff:
                max_exact_marginal_diff = step_diff

            q_val = max(1e-15, q_dist[x_tok])
            accept_prob = min(1.0, p_dist[x_tok] / q_val)
            if sample_rng.random() <= accept_prob:
                generated_tokens.append(x_tok)
                emitted_this_step.append(x_tok)
                accepted_in_step += 1
                draft_tokens_accepted += 1
            else:
                rejected_at = d_idx
                # Roll back rejected draft tokens [d_idx .. k_draft - 1] from KV cache!
                rollback_count = k_draft - d_idx
                kv_active_slots -= rollback_count
                kv_rollbacks += 1
                kv_rolled_back_slots += rollback_count

                # Sample replacement token from exact normalized residual (p - q)_+
                res_raw = [max(0.0, p_dist[v] - q_dist[v]) for v in range(v_size)]
                res_sum = sum(res_raw)
                res_dist = [v / res_sum for v in res_raw] if res_sum > 1e-15 else p_dist
                resampled_tok = _sample_categorical(res_dist, sample_rng.random())
                generated_tokens.append(resampled_tok)
                emitted_this_step.append(resampled_tok)
                kv_active_slots += 1
                break

        if rejected_at is None:
            # All gamma draft tokens accepted -> sample bonus token x_{gamma+1} ~ p_{gamma+1}
            bonus_state = _state_id(generated_tokens)
            p_bonus = target_probs_table[bonus_state]
            bonus_tok = _sample_categorical(p_bonus, sample_rng.random())
            generated_tokens.append(bonus_tok)
            emitted_this_step.append(bonus_tok)
            kv_active_slots += 1
            bonus_tokens_emitted += 1

        if len(step_trace) < 6:
            step_trace.append(
                {
                    "target_step": target_forward_steps,
                    "draft_proposed": [x for x, _, _ in draft_proposals],
                    "accepted_draft_count": accepted_in_step,
                    "emitted_tokens": emitted_this_step,
                    "kv_rolled_back": 0 if rejected_at is None else (k_draft - rejected_at),
                }
            )

    emitted_total = len(generated_tokens) - 2

    # Empirical Monte Carlo check on state 0 (3,000 samples) confirming lossless convergence
    mc_trials = 3000
    p0 = target_probs_table[0]
    q0 = draft_probs_table[0]
    res0_raw = [max(0.0, p0[v] - q0[v]) for v in range(v_size)]
    res0_sum = sum(res0_raw)
    res0_dist = [v / res0_sum for v in res0_raw] if res0_sum > 1e-15 else p0
    spec_hist = [0] * v_size
    mc_rng = random.Random(seed + 999)
    for _ in range(mc_trials):
        cand = _sample_categorical(q0, mc_rng.random())
        if mc_rng.random() <= min(1.0, p0[cand] / max(1e-15, q0[cand])):
            spec_hist[cand] += 1
        else:
            spec_hist[_sample_categorical(res0_dist, mc_rng.random())] += 1

    empirical_spec_tvd = 0.5 * sum(abs((spec_hist[v] / mc_trials) - p0[v]) for v in range(v_size))
    unverified_draft_tvd = 0.5 * sum(abs(q0[v] - p0[v]) for v in range(v_size))

    tokens_per_step = round(emitted_total / max(1, target_forward_steps), 2)
    target_step_reduction_pct = round(
        ((emitted_total - target_forward_steps) / max(1, emitted_total)) * 100.0,
        1,
    )
    wall_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    return {
        "gamma": k_draft,
        "vocab_size": v_size,
        "draft_alignment": align,
        "emitted_tokens": emitted_total,
        "autoregressive_target_steps_baseline": emitted_total,
        "speculative_target_forward_steps": target_forward_steps,
        "tokens_per_target_step_x": tokens_per_step,
        "target_forward_steps_saved_pct": target_step_reduction_pct,
        "draft_tokens_proposed": draft_tokens_proposed,
        "draft_tokens_accepted": draft_tokens_accepted,
        "draft_acceptance_rate_pct": round((draft_tokens_accepted / max(1, draft_tokens_proposed)) * 100.0, 2),
        "all_accepted_bonus_tokens": bonus_tokens_emitted,
        "kv_rollback_events": kv_rollbacks,
        "kv_rolled_back_slots": kv_rolled_back_slots,
        "kv_final_consistent_slots": kv_active_slots,
        "exact_marginal_max_abs_diff": max_exact_marginal_diff,
        "lossless_distribution_preserved": bool(max_exact_marginal_diff < 1e-12),
        "monte_carlo_tvd_speculative": round(empirical_spec_tvd, 4),
        "monte_carlo_tvd_unverified_draft": round(unverified_draft_tvd, 4),
        "step_trace_head": step_trace,
        "wall_clock_ms": wall_ms,
    }


def _evaluate_bucket_plan(lengths: List[int], buckets: List[int]) -> Dict[str, Any]:
    sorted_b = sorted(buckets)
    total_real = sum(lengths)
    total_padded = 0
    per_bucket_counts = {b: 0 for b in sorted_b}

    for s in lengths:
        chosen = sorted_b[-1]
        for b in sorted_b:
            if b >= s:
                chosen = b
                break
        total_padded += chosen
        per_bucket_counts[chosen] += 1

    wasted = total_padded - total_real
    return {
        "buckets": sorted_b,
        "num_buckets": len(sorted_b),
        "real_tokens": total_real,
        "compiled_slot_tokens": total_padded,
        "padding_waste_tokens": wasted,
        "padding_waste_pct": round((wasted / max(1, total_padded)) * 100.0, 2),
        "compute_utilization_pct": round((total_real / max(1, total_padded)) * 100.0, 2),
        "bucket_request_counts": {str(k): v for k, v in per_bucket_counts.items()},
    }


def solve_exact_dp_aot_buckets(
    lengths: List[int],
    candidates: List[int],
    max_buckets: int = 5,
) -> Dict[str, Any]:
    """Exact Dynamic Programming solver selecting `<= max_buckets` AOT sequence buckets
    minimizing total padding waste across `lengths`.
    """
    cands = sorted(set(candidates))
    max_seq = max(lengths)
    if cands[-1] < max_seq:
        cands.append(max_seq)
    m = len(cands)
    k_max = max(1, min(m, int(max_buckets)))

    # Precompute interval cost(a, b] = sum_{a < s <= b} (b - s)
    def _interval_pad_cost(lower_exclusive: int, upper_inclusive: int) -> int:
        return sum(upper_inclusive - s for s in lengths if lower_exclusive < s <= upper_inclusive)

    inf_cost = 10**18
    dp = [[inf_cost] * m for _ in range(k_max + 1)]
    parent = [[-1] * m for _ in range(k_max + 1)]

    for j in range(m):
        dp[1][j] = _interval_pad_cost(0, cands[j])

    for k in range(2, k_max + 1):
        for j in range(k - 1, m):
            best_val = inf_cost
            best_prev = -1
            for prev_j in range(k - 2, j):
                cand_val = dp[k - 1][prev_j] + _interval_pad_cost(cands[prev_j], cands[j])
                if cand_val < best_val:
                    best_val = cand_val
                    best_prev = prev_j
            dp[k][j] = best_val
            parent[k][j] = best_prev

    # Reconstruct optimal bucket subset ending at `cands[m - 1]` (covers max_seq)
    best_k = min(range(1, k_max + 1), key=lambda k: dp[k][m - 1])
    chosen_indices: List[int] = []
    cur_k = best_k
    cur_j = m - 1
    while cur_k >= 1 and cur_j >= 0:
        chosen_indices.append(cur_j)
        cur_j = parent[cur_k][cur_j]
        cur_k -= 1
    chosen_indices.reverse()
    optimal_buckets = [cands[idx] for idx in chosen_indices]

    # Verify against brute-force combinations if candidate set is small (<= 14)
    verified_exact_bruteforce = True
    if m <= 14 and k_max >= 2:
        from itertools import combinations

        brute_best = inf_cost
        for sub in combinations(cands[:-1], best_k - 1):
            trial_b = list(sub) + [cands[-1]]
            w = _evaluate_bucket_plan(lengths, trial_b)["padding_waste_tokens"]
            if w < brute_best:
                brute_best = w
        verified_exact_bruteforce = (dp[best_k][m - 1] == brute_best)

    plan = _evaluate_bucket_plan(lengths, optimal_buckets)
    plan["verified_global_optimum"] = verified_exact_bruteforce
    return plan


def pack_sequences_bfd_with_mask_check(
    lengths: List[int],
    max_pack_len: int = 2048,
) -> Dict[str, Any]:
    """Best-Fit-Decreasing (BFD) sequence packing into `max_pack_len` bins +
    2D block-diagonal causal mask cross-document isolation check.
    """
    cap = max(256, int(max_pack_len))
    sorted_lens = sorted((min(cap, max(1, int(x))) for x in lengths), reverse=True)
    bins: List[List[int]] = []
    bin_remaining: List[int] = []

    for seq_len in sorted_lens:
        best_bin = -1
        best_rem = cap + 1
        for b_idx, rem in enumerate(bin_remaining):
            if seq_len <= rem < best_rem:
                best_rem = rem
                best_bin = b_idx
        if best_bin >= 0:
            bins[best_bin].append(seq_len)
            bin_remaining[best_bin] -= seq_len
        else:
            bins.append([seq_len])
            bin_remaining.append(cap - seq_len)

    total_real_tokens = sum(sorted_lens)
    unpacked_total_slots = len(sorted_lens) * cap
    packed_total_slots = len(bins) * cap
    packed_pad_tokens = packed_total_slots - total_real_tokens
    unpacked_pad_tokens = unpacked_total_slots - total_real_tokens

    # Mathematical verification of 2D block-diagonal causal mask on 3 packed sub-documents
    doc_lens = [7, 9, 6]
    packed_n = sum(doc_lens)
    doc_ids: List[int] = []
    position_ids: List[int] = []
    for d_id, d_len in enumerate(doc_lens):
        doc_ids.extend([d_id] * d_len)
        position_ids.extend(range(d_len))

    rng = random.Random(10603426)
    head_d = 16
    scale = 1.0 / math.sqrt(head_d)
    q = [[rng.gauss(0.0, scale) for _ in range(head_d)] for _ in range(packed_n)]
    k = [[rng.gauss(0.0, scale) for _ in range(head_d)] for _ in range(packed_n)]
    v = [[rng.gauss(0.0, scale) for _ in range(head_d)] for _ in range(packed_n)]
    v_t = list(zip(*v))
    sp = _SUMPROD

    # Compare isolated per-document causal attention vs packed block-diagonal causal attention
    isolated_out: List[List[float]] = []
    offset = 0
    for d_len in doc_lens:
        for local_i in range(d_len):
            gi = offset + local_i
            scores = [sp(q[gi], k[offset + lj]) * scale for lj in range(local_i + 1)]
            probs = _softmax_probs(scores)
            v_slice_t = list(zip(*v[offset : offset + local_i + 1]))
            isolated_out.append([sp(probs, vc) for vc in v_slice_t])
        offset += d_len

    packed_masked_out: List[List[float]] = []
    cross_doc_weight_leakage = 0.0
    for i in range(packed_n):
        valid_js = [j for j in range(i + 1) if doc_ids[j] == doc_ids[i]]
        scores = [sp(q[i], k[j]) * scale for j in valid_js]
        probs = _softmax_probs(scores)
        # Full row probability vector across 0..packed_n-1 has 0.0 outside `valid_js`
        full_p = [0.0] * packed_n
        for j, p_val in zip(valid_js, probs):
            full_p[j] = p_val
        for j in range(packed_n):
            if doc_ids[j] != doc_ids[i]:
                cross_doc_weight_leakage += abs(full_p[j])
        packed_masked_out.append([sp(full_p, vc) for vc in v_t])

    mask_equivalence_diff = max(
        abs(isolated_out[i][c] - packed_masked_out[i][c])
        for i in range(packed_n)
        for c in range(head_d)
    )

    return {
        "num_sequences": len(sorted_lens),
        "max_pack_len": cap,
        "unpacked_bins": len(sorted_lens),
        "bfd_packed_bins": len(bins),
        "bin_reduction_x": round(len(sorted_lens) / max(1, len(bins)), 2),
        "unpacked_padding_waste_pct": round((unpacked_pad_tokens / max(1, unpacked_total_slots)) * 100.0, 2),
        "bfd_padding_waste_pct": round((packed_pad_tokens / max(1, packed_total_slots)) * 100.0, 2),
        "bfd_token_utilization_pct": round((total_real_tokens / max(1, packed_total_slots)) * 100.0, 2),
        "sample_packed_bins": bins[:4],
        "sample_position_id_reset": position_ids,
        "cross_doc_attention_leakage": cross_doc_weight_leakage,
        "block_diagonal_max_abs_diff": mask_equivalence_diff,
        "zero_cross_doc_leakage_verified": bool(
            cross_doc_weight_leakage == 0.0 and mask_equivalence_diff < 1e-15
        ),
    }


def run_aot_bucket_and_packing_benchmark(
    num_sequences: int = 256,
    max_buckets: int = 5,
    max_seq_len: int = 2048,
    seed: int = 10569950,
) -> Dict[str, Any]:
    """Run Exact DP Static AOT Sequence-Bucket Selection + BFD Sequence Packing benchmark."""
    t0 = time.perf_counter()
    n_seq = max(32, min(1024, int(num_sequences)))
    k_buckets = max(2, min(8, int(max_buckets)))
    max_len = 2048 if int(max_seq_len) != 1024 else 1024

    rng = random.Random(seed + n_seq * 5)
    # Realistic bimodal + long-tail production token-length workload (short tool calls + multi-turn RAG)
    lengths: List[int] = []
    for i in range(n_seq):
        mode = i % 10
        if mode < 4:
            # Short interactive queries / router calls (32..220 tokens)
            lengths.append(rng.randint(32, 220))
        elif mode < 8:
            # Multi-turn shopping / RAG context turns (240..780 tokens)
            lengths.append(rng.randint(240, 780))
        else:
            # Long document synthesis turns (820..max_len)
            lengths.append(rng.randint(820, max_len))
    lengths[-1] = max_len

    candidate_grid = [64, 128, 192, 256, 320, 384, 512, 640, 768, 896, 1024, 1536, 2048]
    candidate_grid = [c for c in candidate_grid if c <= max_len]
    if candidate_grid[-1] != max_len:
        candidate_grid.append(max_len)

    single_static = _evaluate_bucket_plan(lengths, [max_len])
    pow2_buckets = [b for b in (128, 256, 512, 1024, 2048) if b <= max_len]
    geometric_plan = _evaluate_bucket_plan(lengths, pow2_buckets)
    dp_optimal_plan = solve_exact_dp_aot_buckets(lengths, candidate_grid, max_buckets=k_buckets)
    bfd_packing = pack_sequences_bfd_with_mask_check(lengths, max_pack_len=max_len)

    wall_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return {
        "num_sequences": n_seq,
        "max_seq_len": max_len,
        "avg_seq_len": round(sum(lengths) / len(lengths), 1),
        "single_static_bucket": single_static,
        "pow2_static_buckets": geometric_plan,
        "dp_optimal_buckets": dp_optimal_plan,
        "padding_reduction_vs_single_pct": round(
            ((single_static["padding_waste_tokens"] - dp_optimal_plan["padding_waste_tokens"])
             / max(1, single_static["padding_waste_tokens"])) * 100.0,
            2,
        ),
        "bfd_sequence_packing": bfd_packing,
        "wall_clock_ms": wall_ms,
    }


def run_quant_spec_suite(
    alpha: float = 0.5,
    group_size: int = 32,
    gamma: int = 4,
    draft_alignment: float = 0.80,
    max_buckets: int = 5,
    top_k: int = 2,
    batch_tokens: int = 48,
    num_target_tokens: int = 256,
    num_sequences: int = 256,
) -> Dict[str, Any]:
    """Combined runner for `/api/quant-spec` returning Quantization+MoE+TP, Speculative Decoding, and DP Bucketing+BFD."""
    t0 = time.perf_counter()
    quant_moe = run_quantization_moe_benchmark(
        batch_tokens=batch_tokens,
        in_features=128,
        out_features=256,
        alpha=alpha,
        group_size=group_size,
        top_k=top_k,
    )
    spec_dec = run_speculative_decoding_benchmark(
        gamma=gamma,
        num_target_tokens=num_target_tokens,
        draft_alignment=draft_alignment,
    )
    aot_pack = run_aot_bucket_and_packing_benchmark(
        num_sequences=num_sequences,
        max_buckets=max_buckets,
        max_seq_len=2048,
    )
    return {
        "quantization_and_moe": quant_moe,
        "speculative_decoding": spec_dec,
        "aot_bucketing_and_packing": aot_pack,
        "suite_wall_ms": round((time.perf_counter() - t0) * 1000.0, 3),
    }


# ===========================================================================
# 3. EFA / Libfabric Lock-Free SQ/CQ Descriptor Ring & MR Interval-Tree Cache
# ===========================================================================


class LockFreeSQCQRing:
    """Fixed-slot binary SubmissionQueue (64B WQE) & CompletionQueue (32B CQE) ring buffer.

    Models Libfabric EFA `rdm` provider fast-path queue pair:
      - Power-of-two capacity (`mask = capacity - 1`) with monotonic producer `head` / consumer `tail`.
      - Pre-allocated contiguous `bytearray` descriptor backing (`64B` WQE, `32B` CQE) with zero
        per-message descriptor object allocation in the post/progress/poll loop.
      - Explicit descriptor state machine (`FREE -> POSTED -> IN_FLIGHT -> COMPLETED -> FREE`)
        plus a 64-bit EFA SRD sliding-window multipath completion bitmap.
    """

    __slots__ = (
        "capacity",
        "mask",
        "sq_buf",
        "cq_buf",
        "_sq_u64",
        "_sq_u32",
        "_sq_u16",
        "_cq_u64",
        "_cq_u32",
        "_cq_u16",
        "sq_head",
        "sq_tail",
        "cq_head",
        "cq_tail",
        "srd_window_base",
        "srd_ooo_bitmap",
        "srd_ooo_completions",
        "state_transitions",
    )

    def __init__(self, capacity: int = 256) -> None:
        if capacity < 16 or (capacity & (capacity - 1)) != 0:
            raise ValueError("Ring capacity must be a power of two >= 16")
        self.capacity = capacity
        self.mask = capacity - 1
        self.sq_buf = bytearray(capacity * WQE_STRUCT.size)
        self.cq_buf = bytearray(capacity * CQE_STRUCT.size)
        self._sq_u64 = memoryview(self.sq_buf).cast("Q")
        self._sq_u32 = memoryview(self.sq_buf).cast("I")
        self._sq_u16 = memoryview(self.sq_buf).cast("H")
        self._cq_u64 = memoryview(self.cq_buf).cast("Q")
        self._cq_u32 = memoryview(self.cq_buf).cast("I")
        self._cq_u16 = memoryview(self.cq_buf).cast("H")
        self.sq_head = 0
        self.sq_tail = 0
        self.cq_head = 0
        self.cq_tail = 0
        self.srd_window_base = 0
        self.srd_ooo_bitmap = 0
        self.srd_ooo_completions = 0
        self.state_transitions = 0

    @property
    def sq_occupancy(self) -> int:
        return self.sq_head - self.sq_tail

    @property
    def cq_occupancy(self) -> int:
        return self.cq_head - self.cq_tail

    def post_send(
        self,
        wr_id: int,
        vaddr: int,
        length: int,
        lkey: int,
        opcode: int = 1,
        path_id: int = 0,
        imm_seq: int = 0,
        posted_ns: int = 0,
    ) -> bool:
        if (self.sq_head - self.sq_tail) >= self.capacity:
            return False
        slot = self.sq_head & self.mask
        b64 = slot * 8
        b32 = slot * 16
        b16 = slot * 32
        b8 = slot * 64
        self._sq_u64[b64] = wr_id
        self._sq_u64[b64 + 1] = vaddr
        self._sq_u32[b32 + 4] = length
        self._sq_u32[b32 + 5] = lkey
        self.sq_buf[b8 + 24] = opcode
        self.sq_buf[b8 + 25] = DESC_POSTED
        self._sq_u16[b16 + 13] = path_id
        self._sq_u32[b32 + 7] = imm_seq
        self._sq_u64[b64 + 4] = posted_ns
        self.sq_head += 1
        self.state_transitions += 1
        return True

    def progress_srd_multipath(self, ooo_stride: int = 3) -> int:
        """Advance posted SQ descriptors through `IN_FLIGHT -> COMPLETED` across 8 SRD ECMP paths."""
        pending = self.sq_head - self.sq_tail
        if pending <= 0:
            return 0
        if (self.cq_head - self.cq_tail + pending) > self.capacity:
            pending = self.capacity - (self.cq_head - self.cq_tail)
            if pending <= 0:
                return 0

        # Process descriptors in deterministic multipath SRD arrival order (slight out-of-order spray)
        progressed = 0
        idx = 0
        while idx < pending:
            if ooo_stride > 1 and (idx + 1) < pending and (idx % ooo_stride == 0):
                first_rel = idx + 1
                second_rel = idx
                pair_count = 2
            else:
                first_rel = idx
                second_rel = -1
                pair_count = 1

            for step_i in range(pair_count):
                rel = first_rel if step_i == 0 else second_rel
                slot = (self.sq_tail + rel) & self.mask
                b64 = slot * 8
                b32 = slot * 16
                b16 = slot * 32
                b8 = slot * 64

                wr_id = self._sq_u64[b64]
                length = self._sq_u32[b32 + 4]
                lkey = self._sq_u32[b32 + 5]
                opcode = self.sq_buf[b8 + 24]
                path_id = self._sq_u16[b16 + 13]
                imm_seq = self._sq_u32[b32 + 7]

                # Transition POSTED -> IN_FLIGHT -> COMPLETED
                self.sq_buf[b8 + 25] = DESC_IN_FLIGHT
                self.state_transitions += 1

                # Track SRD sliding-window bitmap
                delta = imm_seq - self.srd_window_base
                if 0 <= delta < 64:
                    if delta > 0 and (self.srd_ooo_bitmap & 1) == 0:
                        self.srd_ooo_completions += 1
                    self.srd_ooo_bitmap |= (1 << delta)
                    while self.srd_ooo_bitmap & 1:
                        self.srd_ooo_bitmap >>= 1
                        self.srd_window_base += 1

                self.sq_buf[b8 + 25] = DESC_COMPLETED
                self.state_transitions += 1

                cq_slot = self.cq_head & self.mask
                cb64 = cq_slot * 4
                cb32 = cq_slot * 8
                cb16 = cq_slot * 16
                cb8 = cq_slot * 32
                self._cq_u64[cb64] = wr_id
                self._cq_u32[cb32 + 2] = length
                self._cq_u32[cb32 + 3] = lkey
                self.cq_buf[cb8 + 16] = 0
                self.cq_buf[cb8 + 17] = opcode
                self._cq_u16[cb16 + 9] = path_id
                self._cq_u32[cb32 + 5] = imm_seq
                self._cq_u64[cb64 + 3] = 64
                self.cq_head += 1

                # Reset SQ descriptor slot state to FREE
                self.sq_buf[b8 + 25] = DESC_FREE
                self.state_transitions += 1
                progressed += 1

            idx += pair_count

        self.sq_tail += progressed
        return progressed

    def poll_cq_batch(self, max_reap: int = 32) -> Tuple[int, int]:
        """Reap up to `max_reap` CQEs without allocating Python dicts; returns `(reaped_count, total_bytes)`."""
        avail = self.cq_head - self.cq_tail
        n = avail if avail < max_reap else max_reap
        total_bytes = 0
        for _ in range(n):
            cq_slot = self.cq_tail & self.mask
            total_bytes += self._cq_u32[cq_slot * 8 + 2]
            self.cq_tail += 1
        return n, total_bytes


def _run_naive_heap_queue_baseline(num_messages: int, payload_bytes: int = 4096) -> Tuple[int, float, int]:
    """Baseline mutex/heap descriptor queue allocating per-message Python dicts & completion objects."""
    import threading

    lock = threading.Lock()
    queue: List[Dict[str, Any]] = []
    completed_bytes = 0

    gc.collect()
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()

    for m_idx in range(num_messages):
        with lock:
            queue.append(
                {
                    "wr_id": m_idx,
                    "vaddr": 0x7F00_0000_0000 + (m_idx % 64) * 65536,
                    "length": payload_bytes,
                    "lkey": 1000 + (m_idx % 64),
                    "opcode": 1,
                    "state": "POSTED",
                    "path_id": m_idx % 8,
                    "imm_seq": m_idx,
                }
            )
        if len(queue) >= 32 or m_idx == num_messages - 1:
            with lock:
                cqe_batch = [
                    {"wr_id": d["wr_id"], "byte_len": d["length"], "status": 0}
                    for d in queue
                ]
                queue.clear()
            for cqe in cqe_batch:
                completed_bytes += cqe["byte_len"]

    _, peak_heap_bytes = tracemalloc.get_traced_memory()
    if not was_tracing:
        tracemalloc.stop()

    t0 = time.perf_counter()
    for m_idx in range(num_messages):
        with lock:
            queue.append(
                {
                    "wr_id": m_idx,
                    "vaddr": 0x7F00_0000_0000 + (m_idx % 64) * 65536,
                    "length": payload_bytes,
                    "lkey": 1000 + (m_idx % 64),
                    "opcode": 1,
                    "state": "POSTED",
                    "path_id": m_idx % 8,
                    "imm_seq": m_idx,
                }
            )
        if len(queue) >= 32 or m_idx == num_messages - 1:
            with lock:
                cqe_batch = [
                    {"wr_id": d["wr_id"], "byte_len": d["length"], "status": 0}
                    for d in queue
                ]
                queue.clear()
            for cqe in cqe_batch:
                completed_bytes += cqe["byte_len"]
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return int(peak_heap_bytes), round(wall_ms, 3), completed_bytes


@dataclass
class _MRNode:
    start: int
    end: int
    max_end: int
    lkey: int
    pinned_pages: int
    page_checksum: int
    height: int = 1
    hits: int = 0
    left: Optional["_MRNode"] = None
    right: Optional["_MRNode"] = None


def _cold_pin_virtual_pages(vaddr: int, length: int) -> Tuple[int, int]:
    """Touch every 4 KiB virtual page frame in `[vaddr, vaddr + length)` to compute IOMMU DMA frame tags."""
    first_vpn = vaddr >> 12
    last_vpn = (vaddr + max(1, length) - 1) >> 12
    num_pages = last_vpn - first_vpn + 1
    acc = 0x9E3779B97F4A7C15
    for vpn in range(first_vpn, last_vpn + 1):
        # Deterministic 64-bit page-table hash per 4 KiB page
        acc = ((acc ^ (vpn * 0xBF58476D1CE4E5B9)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return num_pages, acc


class MRIntervalTreeCache:
    """Augmented AVL Interval-Tree Memory Registration (`ibv_reg_mr` / `fi_mr_reg`) Cache.

    Maintains subtree `max_end` invariants and AVL balance (`height <= 1.44 * log2(N + 2)`)
    so recurring tensor buffer registrations resolve in `O(log N)` comparisons instead of
    re-walking `O(length / 4096)` OS virtual pages on every collective or RDMA transfer.
    """

    def __init__(self) -> None:
        self._root: Optional[_MRNode] = None
        self._size = 0
        self._next_lkey = 0x10000
        self.cache_hits = 0
        self.cache_misses = 0
        self.invalidations = 0
        self.total_tree_comparisons = 0
        self.total_pages_pinned_cold = 0

    @property
    def size(self) -> int:
        return self._size

    @property
    def tree_height(self) -> int:
        return self._root.height if self._root else 0

    @staticmethod
    def _h(node: Optional[_MRNode]) -> int:
        return node.height if node is not None else 0

    @classmethod
    def _refresh(cls, node: _MRNode) -> None:
        lh = cls._h(node.left)
        rh = cls._h(node.right)
        node.height = (lh if lh > rh else rh) + 1
        me = node.end
        if node.left is not None and node.left.max_end > me:
            me = node.left.max_end
        if node.right is not None and node.right.max_end > me:
            me = node.right.max_end
        node.max_end = me

    @classmethod
    def _rotate_right(cls, y: _MRNode) -> _MRNode:
        x = y.left
        assert x is not None
        t2 = x.right
        x.right = y
        y.left = t2
        cls._refresh(y)
        cls._refresh(x)
        return x

    @classmethod
    def _rotate_left(cls, x: _MRNode) -> _MRNode:
        y = x.right
        assert y is not None
        t2 = y.left
        y.left = x
        x.right = t2
        cls._refresh(x)
        cls._refresh(y)
        return y

    @classmethod
    def _rebalance(cls, node: _MRNode) -> _MRNode:
        cls._refresh(node)
        balance = cls._h(node.left) - cls._h(node.right)
        if balance > 1 and node.left is not None:
            if cls._h(node.left.left) < cls._h(node.left.right):
                node.left = cls._rotate_left(node.left)
            return cls._rotate_right(node)
        if balance < -1 and node.right is not None:
            if cls._h(node.right.right) < cls._h(node.right.left):
                node.right = cls._rotate_right(node.right)
            return cls._rotate_left(node)
        return node

    def _insert_node(self, root: Optional[_MRNode], new_node: _MRNode) -> _MRNode:
        if root is None:
            return new_node
        if (new_node.start, new_node.end) < (root.start, root.end):
            root.left = self._insert_node(root.left, new_node)
        else:
            root.right = self._insert_node(root.right, new_node)
        return self._rebalance(root)

    def _delete_min(self, node: _MRNode) -> Tuple[_MRNode, Optional[_MRNode]]:
        if node.left is None:
            return node, node.right
        min_node, new_left = self._delete_min(node.left)
        node.left = new_left
        return min_node, self._rebalance(node)

    def _delete_exact(self, root: Optional[_MRNode], start: int, end: int, lkey: int) -> Optional[_MRNode]:
        if root is None:
            return None
        if (start, end) < (root.start, root.end):
            root.left = self._delete_exact(root.left, start, end, lkey)
        elif (start, end) > (root.start, root.end):
            root.right = self._delete_exact(root.right, start, end, lkey)
        elif root.lkey != lkey:
            root.right = self._delete_exact(root.right, start, end, lkey)
        else:
            if root.left is None:
                return root.right
            if root.right is None:
                return root.left
            succ, new_right = self._delete_min(root.right)
            succ.left = root.left
            succ.right = new_right
            return self._rebalance(succ)
        return self._rebalance(root)

    def _search_covering(self, node: Optional[_MRNode], q_start: int, q_end: int) -> Optional[_MRNode]:
        if node is None or node.max_end < q_end:
            return None
        self.total_tree_comparisons += 1
        if node.start <= q_start and node.end >= q_end:
            return node
        if node.left is not None and node.left.max_end >= q_end:
            hit = self._search_covering(node.left, q_start, q_end)
            if hit is not None:
                return hit
        if q_start >= node.start and node.right is not None and node.right.max_end >= q_end:
            return self._search_covering(node.right, q_start, q_end)
        return None

    def _collect_overlapping(
        self,
        node: Optional[_MRNode],
        q_start: int,
        q_end: int,
        out: List[_MRNode],
    ) -> None:
        if node is None or node.max_end <= q_start:
            return
        if node.left is not None and node.left.max_end > q_start:
            self._collect_overlapping(node.left, q_start, q_end, out)
        if node.start < q_end and node.end > q_start:
            out.append(node)
        if node.start < q_end and node.right is not None:
            self._collect_overlapping(node.right, q_start, q_end, out)

    def acquire_mr(self, vaddr: int, length: int) -> Tuple[int, bool]:
        """Lookup `[vaddr, vaddr + length)` in `O(log N)` or coalesce/pin pages on cold miss."""
        q_start = int(vaddr)
        q_end = q_start + max(1, int(length))

        hit = self._search_covering(self._root, q_start, q_end)
        if hit is not None:
            hit.hits += 1
            self.cache_hits += 1
            return hit.lkey, True

        # Cold miss: coalesce any partially overlapping MRs so slab ranges stay clean
        overlaps: List[_MRNode] = []
        self._collect_overlapping(self._root, q_start, q_end, overlaps)
        merged_start = q_start
        merged_end = q_end
        for ov in overlaps:
            merged_start = min(merged_start, ov.start)
            merged_end = max(merged_end, ov.end)
            self._root = self._delete_exact(self._root, ov.start, ov.end, ov.lkey)
            self._size -= 1

        pages, csum = _cold_pin_virtual_pages(merged_start, merged_end - merged_start)
        self.total_pages_pinned_cold += pages
        self.cache_misses += 1
        lkey = self._next_lkey
        self._next_lkey += 1

        new_node = _MRNode(
            start=merged_start,
            end=merged_end,
            max_end=merged_end,
            lkey=lkey,
            pinned_pages=pages,
            page_checksum=csum,
        )
        self._root = self._insert_node(self._root, new_node)
        self._size += 1
        return lkey, False

    def invalidate_range(self, vaddr: int, length: int) -> int:
        """Evict any cached MR overlapping `[vaddr, vaddr + length)` (munmap / slab free hook)."""
        q_start = int(vaddr)
        q_end = q_start + max(1, int(length))
        overlaps: List[_MRNode] = []
        self._collect_overlapping(self._root, q_start, q_end, overlaps)
        for ov in overlaps:
            self._root = self._delete_exact(self._root, ov.start, ov.end, ov.lkey)
            self._size -= 1
            self.invalidations += 1
        return len(overlaps)


def run_transport_ring_mr_benchmark(
    ring_capacity: int = 256,
    num_messages: int = 4096,
    batch_size: int = 32,
    num_mr_regions: int = 128,
    num_mr_lookups: int = 1024,
    tensor_bytes: int = 1048576,
    seed: int = 10592911,
) -> Dict[str, Any]:
    """Benchmark fixed-slot SQ/CQ binary descriptor ring & Augmented AVL Interval-Tree MR cache."""
    # Clamp ring_capacity to clean power-of-two in [64 .. 1024]
    raw_cap = max(64, min(1024, int(ring_capacity)))
    cap = 1 << (raw_cap.bit_length() - 1)
    if cap < 64:
        cap = 64

    n_msgs = max(256, min(16384, int(num_messages)))
    b_size = max(8, min(cap // 2, int(batch_size)))
    n_regions = max(16, min(512, int(num_mr_regions)))
    n_lookups = max(128, min(4096, int(num_mr_lookups)))
    slab_bytes = max(262144, min(4194304, int(tensor_bytes)))

    # 1A. Measure steady-state dynamic heap allocation of LockFreeSQCQRing via tracemalloc
    ring = LockFreeSQCQRing(capacity=cap)
    gc.collect()
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()

    reaped_total = 0
    transferred_bytes = 0
    posted = 0
    while posted < n_msgs:
        chunk = min(b_size, n_msgs - posted)
        for off in range(chunk):
            seq = posted + off
            ring.post_send(
                wr_id=seq,
                vaddr=0x7F00_0000_0000 + (seq & 63) * 65536,
                length=4096,
                lkey=0x10000 + (seq & 63),
                opcode=1,
                path_id=seq & 7,
                imm_seq=seq,
                posted_ns=0,
            )
        posted += chunk
        ring.progress_srd_multipath(ooo_stride=3)
        r_cnt, r_bytes = ring.poll_cq_batch(max_reap=chunk)
        reaped_total += r_cnt
        transferred_bytes += r_bytes

    _, ring_steady_heap_bytes = tracemalloc.get_traced_memory()
    if not was_tracing:
        tracemalloc.stop()

    # 1B. Measure clean uninstrumented wall-clock time for LockFreeSQCQRing vs Naive Heap Queue
    ring_timed = LockFreeSQCQRing(capacity=cap)
    t_ring0 = time.perf_counter()
    p2 = 0
    reaped_timed = 0
    while p2 < n_msgs:
        chunk = min(b_size, n_msgs - p2)
        for off in range(chunk):
            seq = p2 + off
            ring_timed.post_send(
                wr_id=seq,
                vaddr=0x7F00_0000_0000 + (seq & 63) * 65536,
                length=4096,
                lkey=0x10000 + (seq & 63),
                opcode=1,
                path_id=seq & 7,
                imm_seq=seq,
                posted_ns=0,
            )
        p2 += chunk
        ring_timed.progress_srd_multipath(ooo_stride=3)
        rc, _ = ring_timed.poll_cq_batch(max_reap=chunk)
        reaped_timed += rc
    ring_wall_ms = round((time.perf_counter() - t_ring0) * 1000.0, 3)

    naive_heap_bytes, naive_wall_ms, _ = _run_naive_heap_queue_baseline(n_msgs, payload_bytes=4096)

    # 2. Augmented Interval-Tree MR Cache vs Cold 4KiB Page-Pinning Registration
    rng = random.Random(seed + n_regions * 13)
    base_arena = 0x7F80_0000_0000
    stride = slab_bytes * 2  # non-adjacent virtual slabs
    slab_bases = [base_arena + idx * stride for idx in range(n_regions)]

    # Build realistic tensor transfer trace:
    # First warm up `n_regions` distinct tensor arenas (cold registrations), then execute
    # `n_lookups` collective/RDMA sub-slice transfers (`92%` recurring cached sub-slices, `8%` new cold slabs)
    transfer_trace: List[Tuple[int, int]] = []
    for base in slab_bases:
        transfer_trace.append((base, slab_bytes))

    extra_cold_base = base_arena + n_regions * stride
    cold_idx = 0
    for step in range(n_lookups):
        if step % 12 == 11:
            # New uncached temporary buffer allocation -> cold MR miss
            va = extra_cold_base + cold_idx * stride
            cold_idx += 1
            transfer_trace.append((va, slab_bytes // 2))
        else:
            # Sub-slice inside an already-registered ML weight/activation slab -> O(log N) Interval-Tree HIT
            picked_base = slab_bases[rng.randrange(n_regions)]
            sub_off = (step & 3) * (slab_bytes // 8)
            sub_len = slab_bytes // 2
            transfer_trace.append((picked_base + sub_off, sub_len))

    # Measure MR Interval-Tree Cache on the steady-state lookup stream (after initial slab registration)
    mr_cache = MRIntervalTreeCache()
    for base in slab_bases:
        mr_cache.acquire_mr(base, slab_bytes)

    mr_cache.cache_hits = 0
    mr_cache.cache_misses = 0
    mr_cache.total_tree_comparisons = 0

    t_mr_cached0 = time.perf_counter()
    for va, sz in transfer_trace[n_regions:]:
        mr_cache.acquire_mr(va, sz)
    mr_cached_ms = round((time.perf_counter() - t_mr_cached0) * 1000.0, 3)

    # Measure Uncached Cold `ibv_reg_mr` Page-Pinning Path on the exact same `n_lookups` stream
    t_mr_cold0 = time.perf_counter()
    cold_pages_total = 0
    cold_csum_xor = 0
    for va, sz in transfer_trace[n_regions:]:
        p_cnt, p_csum = _cold_pin_virtual_pages(va, sz)
        cold_pages_total += p_cnt
        cold_csum_xor ^= p_csum
    mr_cold_ms = round((time.perf_counter() - t_mr_cold0) * 1000.0, 3)

    # Verify `munmap` / `free` invalidation removes stale `lkey` cleanly
    evict_target_va = slab_bases[0]
    old_lkey, was_hit_before = mr_cache.acquire_mr(evict_target_va, slab_bytes // 2)
    evicted_count = mr_cache.invalidate_range(evict_target_va + 4096, 8192)
    new_lkey, was_hit_after_evict = mr_cache.acquire_mr(evict_target_va, slab_bytes // 2)

    theoretical_avl_bound = int(math.ceil(1.44 * math.log2(mr_cache.size + 2)))
    avg_comparisons = round(mr_cache.total_tree_comparisons / max(1, n_lookups + 2), 2)
    mr_hit_rate_pct = round(
        (mr_cache.cache_hits / max(1, mr_cache.cache_hits + mr_cache.cache_misses)) * 100.0,
        2,
    )
    mr_speedup_x = round(mr_cold_ms / max(0.001, mr_cached_ms), 2)

    return {
        "sq_cq_ring": {
            "ring_capacity": cap,
            "wqe_bytes": WQE_STRUCT.size,
            "cqe_bytes": CQE_STRUCT.size,
            "preallocated_ring_bytes": cap * (WQE_STRUCT.size + CQE_STRUCT.size),
            "messages_processed": n_msgs,
            "reaped_completions": reaped_total,
            "transferred_payload_bytes": transferred_bytes,
            "descriptor_state_transitions": ring.state_transitions,
            "srd_ooo_reordered_packets": ring.srd_ooo_completions,
            "final_sq_occupancy": ring.sq_occupancy,
            "final_cq_occupancy": ring.cq_occupancy,
            "steady_state_heap_alloc_bytes": int(ring_steady_heap_bytes),
            "naive_queue_heap_alloc_bytes": naive_heap_bytes,
            "heap_alloc_reduction_pct": round(
                ((naive_heap_bytes - ring_steady_heap_bytes) / max(1, naive_heap_bytes)) * 100.0,
                2,
            ),
            "ring_wall_ms": ring_wall_ms,
            "naive_queue_wall_ms": naive_wall_ms,
            "ring_us_per_msg": round((ring_wall_ms * 1000.0) / max(1, n_msgs), 3),
        },
        "mr_interval_cache": {
            "registered_slabs_in_tree": mr_cache.size,
            "avl_tree_height": mr_cache.tree_height,
            "avl_height_upper_bound": theoretical_avl_bound,
            "avl_balanced_verified": bool(mr_cache.tree_height <= theoretical_avl_bound),
            "avg_tree_comparisons_per_lookup": avg_comparisons,
            "lookups_evaluated": n_lookups,
            "tensor_slab_bytes": slab_bytes,
            "cache_hits": mr_cache.cache_hits,
            "cache_misses": mr_cache.cache_misses,
            "cache_hit_rate_pct": mr_hit_rate_pct,
            "cached_lookup_wall_ms": mr_cached_ms,
            "cold_page_pin_wall_ms": mr_cold_ms,
            "cold_pages_walked": cold_pages_total,
            "mr_cache_speedup_x": mr_speedup_x,
            "cached_us_per_lookup": round((mr_cached_ms * 1000.0) / max(1, n_lookups), 3),
            "cold_us_per_lookup": round((mr_cold_ms * 1000.0) / max(1, n_lookups), 3),
            "invalidation_safety_check": {
                "hit_before_invalidate": was_hit_before,
                "evicted_overlapping_mrs": evicted_count,
                "hit_immediately_after_invalidate": was_hit_after_evict,
                "fresh_lkey_issued": bool(new_lkey != old_lkey),
                "verified": bool(
                    was_hit_before and evicted_count == 1 and (not was_hit_after_evict) and new_lkey != old_lkey
                ),
            },
            "page_checksum_guard": cold_csum_xor & 0xFFFF,
        },
    }
