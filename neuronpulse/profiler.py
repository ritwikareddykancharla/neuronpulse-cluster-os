"""Roofline LLM Serving & Neuron Kernel (NKI) Profiler for Llama-3.1-70B on Trainium2 (`trn2`).

Models prefill (compute-bound MatMul + attention FLOPs + TP All-Reduce) and decode
(HBM3e bandwidth-bound weight/KV streaming + NeuronLink collective latency) across:
  - Tensor Parallelism (`TP = 2, 4, 8, 16`)
  - Precision (`BF16`, `FP8`, `W4A16`)
  - Continuous Batching Size (`1 .. 64`)
  - NKI FlashAttention-2 + Fused RMSNorm/SwiGLU Kernel (`nki_flash_fusion: bool`)
"""
from __future__ import annotations

from typing import Any, Dict, List


MODEL_PARAMS_BILLIONS = 70.6  # Llama-3.1-70B
NUM_LAYERS = 80
KV_HEADS = 8
HEAD_DIM = 128
HBM_CAPACITY_PER_CORE_GB = 48.0  # Logical NeuronCore-v3 HBM3e slice
PEAK_HBM_BW_PER_CORE_GBPS = 2900.0  # Effective sustained HBM3e GB/s per logical core
PEAK_BF16_TFLOPS_PER_CORE = 315.0  # Effective sustained BF16 TFLOPs per logical core

PRECISION_SPECS: Dict[str, Dict[str, Any]] = {
    "BF16": {
        "bytes_per_param": 2.0,
        "kv_bytes_per_elem": 2.0,
        "compute_speedup": 1.0,
        "compiler_flag": "--auto-cast=none --model-type=transformer",
    },
    "FP8": {
        "bytes_per_param": 1.0,
        "kv_bytes_per_elem": 1.0,
        "compute_speedup": 1.85,
        "compiler_flag": "--auto-cast=fp8_e4m3 --quantized-kv-cache=fp8",
    },
    "W4A16": {
        "bytes_per_param": 0.55,  # 4-bit weights + groupwise scale metadata
        "kv_bytes_per_elem": 2.0,
        "compute_speedup": 1.28,  # Dequantize into BF16 TensorEngine + massive HBM decode win
        "compiler_flag": "--quantization-type=int4_awq --quantization-group-size=128",
    },
}


def simulate_neuron_serving(
    tp_degree: int = 8,
    precision: str = "FP8",
    batch_size: int = 16,
    nki_flash_fusion: bool = True,
    seq_len: int = 4096,
) -> Dict[str, Any]:
    """Compute TTFT (ms), TPOT (ms), aggregate Throughput (tok/s), and per-core HBM3e footprint."""
    tp = int(tp_degree)
    if tp not in (2, 4, 8, 16):
        raise ValueError("tp_degree must be one of 2, 4, 8, or 16")
    prec = precision.upper()
    if prec not in PRECISION_SPECS:
        raise ValueError("precision must be one of BF16, FP8, or W4A16")
    bsz = max(1, min(128, int(batch_size)))
    ctx = max(512, min(32768, int(seq_len)))

    spec = PRECISION_SPECS[prec]
    total_weight_gb = MODEL_PARAMS_BILLIONS * spec["bytes_per_param"]
    weights_per_core_gb = total_weight_gb / tp

    # GQA KV cache footprint across `bsz` sequences of length `ctx`
    # 2 (K+V) * layers * kv_heads * head_dim * ctx * bsz * bytes
    total_kv_bytes = 2 * NUM_LAYERS * KV_HEADS * HEAD_DIM * ctx * bsz * spec["kv_bytes_per_elem"]
    total_kv_gb = total_kv_bytes / 1e9
    kv_per_core_gb = total_kv_gb / tp

    # Workspace + NKI scratchpad overhead (NKI FlashAttention eliminates O(S^2) attention score materialization)
    activation_overhead_gb = (0.85 if nki_flash_fusion else 2.65) * (bsz / 8.0) * (ctx / 4096.0)
    hbm_used_per_core_gb = round(weights_per_core_gb + kv_per_core_gb + activation_overhead_gb, 2)
    fits_in_hbm = hbm_used_per_core_gb <= (HBM_CAPACITY_PER_CORE_GB - 1.5)
    hbm_util_pct = round(min(140.0, (hbm_used_per_core_gb / HBM_CAPACITY_PER_CORE_GB) * 100.0), 1)

    # Prefill TTFT (ms): compute-bound 2 * N * B_prefill * S FLOPs + quadratic attention term
    # Assume prefill processes prompt of length `ctx` (chunked/batched effective concurrency min(bsz, 4))
    eff_prefill_batch = min(bsz, 4)
    linear_tflops = (2.0 * MODEL_PARAMS_BILLIONS * 1e9 * eff_prefill_batch * ctx) / 1e12
    attn_tflops = (4.0 * NUM_LAYERS * 8192 * eff_prefill_batch * (ctx ** 2)) / 1e12
    nki_attn_factor = 0.52 if nki_flash_fusion else 1.35
    cluster_tflops = tp * PEAK_BF16_TFLOPS_PER_CORE * spec["compute_speedup"]
    compute_ms = ((linear_tflops + attn_tflops * nki_attn_factor) / cluster_tflops) * 1000.0

    # Collective All-Reduce latency across TP ranks (TP=16 crosses intra-server boundary slightly higher)
    tp_allreduce_ms = {2: 1.8, 4: 2.6, 8: 3.9, 16: 6.8}[tp] * (0.82 if nki_flash_fusion else 1.18)
    ttft_ms = round(compute_ms + tp_allreduce_ms, 1)

    # Decode TPOT (ms): HBM3e bandwidth-bound weight + KV read per generated token + collective sync
    bytes_read_per_core_gb = weights_per_core_gb + (kv_per_core_gb * 0.85)
    bw_efficiency = 0.84 if nki_flash_fusion else 0.63
    mem_stream_ms = (bytes_read_per_core_gb / (PEAK_HBM_BW_PER_CORE_GBPS * bw_efficiency)) * 1000.0
    decode_compute_ms = ((2.0 * MODEL_PARAMS_BILLIONS * 1e9 * bsz) / 1e12 / cluster_tflops) * 1000.0
    decode_comm_ms = {2: 0.7, 4: 1.0, 8: 1.45, 16: 2.55}[tp] * (0.80 if nki_flash_fusion else 1.15)
    tpot_ms = round(mem_stream_ms + decode_compute_ms + decode_comm_ms, 2)

    replica_tok_s = round((bsz * 1000.0) / max(tpot_ms, 0.5), 1) if fits_in_hbm else 0.0
    replicas_on_ultraserver = 64 // tp
    ultraserver_tok_s = round(replica_tok_s * replicas_on_ultraserver, 1)

    nki_flags = (
        "--lnc=2 --enable-fast-loading-neuron-binaries "
        + spec["compiler_flag"]
        + (" --neuron-nki-flash-attn=enable --fuse-qkv --fuse-mlp" if nki_flash_fusion else " --neuron-nki-flash-attn=disable")
    )

    bottleneck = (
        "HBM_OOM_EXCEEDED"
        if not fits_in_hbm
        else ("COLLECTIVE_ALLREDUCE_LATENCY" if tp == 16 and bsz <= 4 else "HBM3E_MEMORY_BANDWIDTH")
    )

    return {
        "model": "meta-llama/Llama-3.1-70B-Instruct",
        "tp_degree": tp,
        "replicas_on_64_cores": replicas_on_ultraserver,
        "precision": prec,
        "batch_size": bsz,
        "seq_len": ctx,
        "nki_flash_fusion": bool(nki_flash_fusion),
        "fits_in_hbm": fits_in_hbm,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "replica_throughput_tok_s": replica_tok_s,
        "ultraserver_throughput_tok_s": ultraserver_tok_s,
        "hbm_used_per_core_gb": hbm_used_per_core_gb,
        "hbm_capacity_per_core_gb": HBM_CAPACITY_PER_CORE_GB,
        "hbm_utilization_pct": hbm_util_pct,
        "memory_breakdown_gb": {
            "weights_per_core_gb": round(weights_per_core_gb, 2),
            "kv_cache_per_core_gb": round(kv_per_core_gb, 2),
            "activations_scratch_gb": round(activation_overhead_gb, 2),
        },
        "bottleneck": bottleneck,
        "neuron_cc_flags": nki_flags,
    }


def compare_serving_configs(
    tp_degree: int = 8,
    precision: str = "FP8",
    batch_size: int = 16,
    nki_flash_fusion: bool = True,
    seq_len: int = 4096,
) -> Dict[str, Any]:
    """Return selected config alongside comparison matrix for visual TTFT vs Throughput bars."""
    selected = simulate_neuron_serving(
        tp_degree=tp_degree,
        precision=precision,
        batch_size=batch_size,
        nki_flash_fusion=nki_flash_fusion,
        seq_len=seq_len,
    )

    matrix: List[Dict[str, Any]] = [
        {
            "label": "BF16 · TP=8 · Unfused",
            **simulate_neuron_serving(tp_degree=8, precision="BF16", batch_size=batch_size, nki_flash_fusion=False, seq_len=seq_len),
        },
        {
            "label": "BF16 · TP=8 · NKI Flash",
            **simulate_neuron_serving(tp_degree=8, precision="BF16", batch_size=batch_size, nki_flash_fusion=True, seq_len=seq_len),
        },
        {
            "label": "FP8 · TP=8 · NKI Flash",
            **simulate_neuron_serving(tp_degree=8, precision="FP8", batch_size=batch_size, nki_flash_fusion=True, seq_len=seq_len),
        },
        {
            "label": "W4A16 · TP=4 · NKI Flash",
            **simulate_neuron_serving(tp_degree=4, precision="W4A16", batch_size=batch_size, nki_flash_fusion=True, seq_len=seq_len),
        },
        {
            "label": f"Active ({precision} · TP={tp_degree} · {'NKI' if nki_flash_fusion else 'Std'})",
            **selected,
        },
    ]

    baseline = matrix[0]
    speedup_tok = (
        round(selected["ultraserver_throughput_tok_s"] / max(baseline["ultraserver_throughput_tok_s"], 1.0), 2)
        if selected["fits_in_hbm"]
        else 0.0
    )
    ttft_reduction_pct = round(
        ((baseline["ttft_ms"] - selected["ttft_ms"]) / max(baseline["ttft_ms"], 1.0)) * 100.0,
        1,
    )

    return {
        "selected": selected,
        "comparison_bars": matrix,
        "vs_bf16_unfused_baseline": {
            "throughput_speedup_x": speedup_tok,
            "ttft_reduction_pct": ttft_reduction_pct,
        },
    }
