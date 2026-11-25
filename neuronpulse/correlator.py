"""Distributed ML Worker Causal Fault Correlator & Live Serverless AWS Bedrock Profiler.

Contains two 100%-real executable engines:
  1. `CrossNodeErrorCorrelator`: Ranks shuffled out-of-order distributed worker fault logs
     by causal subsystem precedence + monotonic nanosecond timestamps to isolate the single
     root-cause worker failure amidst `N - 1` symptomatic collective barrier timeouts.
  2. `BedrockModelProfiler`: Executes real live `boto3.client("bedrock-runtime").converse(...)`
     calls against `us.amazon.nova-micro-v1:0` and `us.amazon.nova-lite-v1:0` in `us-west-2`,
     measuring real `p50`/`p95` wall-clock & server latency (`ms`), real output `tokens/sec`,
     exact token counts, and real USD per 1,000 requests.
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Part 1: Distributed Worker Causal Fault Correlator
# ---------------------------------------------------------------------------

SUBSYSTEM_CAUSAL_PRIORITY: Dict[str, int] = {
    "WORKER_MEMORY_ALLOCATOR": 100,
    "RPC_SOCKET_TRANSPORT": 90,
    "KV_BLOCK_POOL_MANAGER": 80,
    "BATCH_DECODE_SCHEDULER": 20,
    "COLLECTIVE_BARRIER_RING": 10,
}

DISTRIBUTED_FAULT_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "oom_allocator_trap": {
        "scenario_id": "oom_allocator_trap",
        "title": "Worker Slab OOM Trap -> 63 Peer AllReduce Barrier Timeouts",
        "root_subsystem": "WORKER_MEMORY_ALLOCATOR",
        "root_code": "ERR_WORKER_SLAB_OOM_0x41",
        "root_message": "Tensor arena allocation failed on rank; local worker execution halted before collective step.",
        "symptom_subsystem": "COLLECTIVE_BARRIER_RING",
        "symptom_code": "ERR_ALLREDUCE_BARRIER_TIMEOUT",
        "symptom_message": "Collective ring barrier timed out waiting on upstream peer rank.",
        "propagation_delay_ns": 1_250_000_000,
        "recommended_remediation": "QUARANTINE_FAULTED_WORKER_AND_RESTORE_CHECKPOINT",
    },
    "rpc_socket_reset": {
        "scenario_id": "rpc_socket_reset",
        "title": "RPC Transport Connection Reset -> 63 Peer Step Straggler Alarms",
        "root_subsystem": "RPC_SOCKET_TRANSPORT",
        "root_code": "ERR_RPC_CONN_RESET_0x58",
        "root_message": "Inter-worker RPC socket dropped with ECONNRESET during tensor shard exchange.",
        "symptom_subsystem": "COLLECTIVE_BARRIER_RING",
        "symptom_code": "WARN_COLLECTIVE_STRAGGLER_STALL",
        "symptom_message": "Distributed step latency breached SLA waiting on unresponsive RPC peer.",
        "propagation_delay_ns": 640_000_000,
        "recommended_remediation": "DRAIN_BROKEN_TRANSPORT_WORKER_AND_REINIT_PROCESS_GROUP",
    },
    "kv_pool_exhaustion": {
        "scenario_id": "kv_pool_exhaustion",
        "title": "Paged KV Block Watermark Exhausted -> 63 Scheduler Backpressure Warnings",
        "root_subsystem": "KV_BLOCK_POOL_MANAGER",
        "root_code": "ERR_KV_FREE_BLOCKS_ZERO_0x33",
        "root_message": "Free physical KV block list reached 0 blocks during burst prefill admission.",
        "symptom_subsystem": "BATCH_DECODE_SCHEDULER",
        "symptom_code": "WARN_DECODE_QUEUE_BACKPRESSURE",
        "symptom_message": "Continuous batching scheduler preempted pending decode step due to KV pool pressure.",
        "propagation_delay_ns": 85_000_000,
        "recommended_remediation": "EVICT_IDLE_PREFIX_CACHE_OR_REDUCE_MAX_BATCH_TOKENS",
    },
}


def build_worker_fault_stream(
    scenario_id: str = "oom_allocator_trap",
    num_workers: int = 64,
    root_rank: int = 19,
    shuffle_seed: Optional[int] = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build a 64-worker distributed log event burst with realistic out-of-order collector arrival."""
    if scenario_id not in DISTRIBUTED_FAULT_SCENARIOS:
        raise KeyError(f"Unknown scenario_id: {scenario_id}")

    n_workers = max(8, min(256, int(num_workers)))
    target_rank = int(root_rank) % n_workers
    spec = DISTRIBUTED_FAULT_SCENARIOS[scenario_id]
    base_ns = 1_764_050_000_000_000_000

    events: List[Dict[str, Any]] = []
    for rank in range(n_workers):
        worker_group = f"worker-group-{rank // 8:02d}"
        worker_id = f"worker-{rank:02d}"
        if rank == target_rank:
            events.append(
                {
                    "monotonic_ns": base_ns,
                    "worker_group": worker_group,
                    "worker_id": worker_id,
                    "rank": rank,
                    "subsystem": spec["root_subsystem"],
                    "severity": "ERROR",
                    "code": spec["root_code"],
                    "message": spec["root_message"],
                }
            )
        else:
            events.append(
                {
                    "monotonic_ns": base_ns + spec["propagation_delay_ns"] + rank * 15_000,
                    "worker_group": worker_group,
                    "worker_id": worker_id,
                    "rank": rank,
                    "subsystem": spec["symptom_subsystem"],
                    "severity": "WARN",
                    "code": spec["symptom_code"],
                    "message": spec["symptom_message"],
                }
            )

    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        rng.shuffle(events)

    truth = {
        "scenario_id": scenario_id,
        "title": spec["title"],
        "expected_worker_id": f"worker-{target_rank:02d}",
        "expected_rank": target_rank,
        "expected_code": spec["root_code"],
    }
    return events, truth


class CrossNodeErrorCorrelator:
    """Separates root-cause distributed worker faults from `N - 1` symptomatic barrier alarms."""

    def correlate(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        if not events:
            return {
                "status": "HEALTHY",
                "raw_alert_count": 0,
                "suppressed_symptom_count": 0,
                "root_cause": None,
                "wall_clock_ms": round((time.perf_counter() - t0) * 1000.0, 4),
            }

        earliest_ns = min(int(e["monotonic_ns"]) for e in events)
        ranked = sorted(
            events,
            key=lambda e: (
                -SUBSYSTEM_CAUSAL_PRIORITY.get(str(e.get("subsystem")), 0),
                int(e["monotonic_ns"]) - earliest_ns,
                int(e.get("rank", 0)),
            ),
        )
        root = ranked[0]
        downstream = [
            e
            for e in events
            if not (e["worker_id"] == root["worker_id"] and e["code"] == root["code"])
        ]

        arrival_first = events[0]
        cascade_window_ms = round(
            (max(int(e["monotonic_ns"]) for e in events) - int(root["monotonic_ns"])) / 1_000_000.0,
            2,
        )

        remediation = "INSPECT_WORKER_LOGS"
        for spec in DISTRIBUTED_FAULT_SCENARIOS.values():
            if spec["root_code"] == root["code"]:
                remediation = spec["recommended_remediation"]
                break

        wall_ms = round((time.perf_counter() - t0) * 1000.0, 4)
        return {
            "status": "ROOT_CAUSE_ISOLATED",
            "raw_alert_count": len(events),
            "suppressed_symptom_count": len(downstream),
            "alert_compression_ratio": f"{len(events)}:1",
            "cascade_window_ms": cascade_window_ms,
            "recommended_remediation": remediation,
            "naive_first_arrival_pick": {
                "worker_id": arrival_first["worker_id"],
                "rank": arrival_first["rank"],
                "code": arrival_first["code"],
                "subsystem": arrival_first["subsystem"],
            },
            "root_cause": {
                "worker_group": root["worker_group"],
                "worker_id": root["worker_id"],
                "rank": root["rank"],
                "subsystem": root["subsystem"],
                "code": root["code"],
                "message": root["message"],
            },
            "sample_suppressed_symptoms": downstream[:5],
            "wall_clock_ms": wall_ms,
        }


def run_correlator_benchmark(
    trials: int = 48,
    num_workers: int = 64,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate `CrossNodeErrorCorrelator` across 48 shuffled distributed worker log streams."""
    n_trials = max(6, min(240, int(trials)))
    n_workers = max(8, min(256, int(num_workers)))
    rng = random.Random(seed)
    correlator = CrossNodeErrorCorrelator()
    scenario_keys = list(DISTRIBUTED_FAULT_SCENARIOS.keys())

    t0 = time.perf_counter()
    causal_hits = 0
    arrival_hits = 0
    total_raw_alarms = 0
    sample_trial_detail: Optional[Dict[str, Any]] = None

    for i in range(n_trials):
        scen = scenario_keys[i % len(scenario_keys)]
        target_rank = (i * 11 + 5) % n_workers
        events, truth = build_worker_fault_stream(
            scenario_id=scen,
            num_workers=n_workers,
            root_rank=target_rank,
            shuffle_seed=None,
        )
        rng.shuffle(events)
        total_raw_alarms += len(events)

        if events[0]["worker_id"] == truth["expected_worker_id"]:
            arrival_hits += 1

        res = correlator.correlate(events)
        rc = res["root_cause"]
        if rc and rc["worker_id"] == truth["expected_worker_id"] and rc["code"] == truth["expected_code"]:
            causal_hits += 1

        if sample_trial_detail is None:
            sample_trial_detail = {
                "scenario": truth,
                "correlation": res,
            }

    wall_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return {
        "trials": n_trials,
        "workers_per_trial": n_workers,
        "total_raw_worker_alarms": total_raw_alarms,
        "isolated_root_incidents": n_trials,
        "alert_compression_ratio": f"{n_workers}:1",
        "alert_reduction_pct": round(((total_raw_alarms - n_trials) / max(1, total_raw_alarms)) * 100.0, 2),
        "causal_correlator_correct": causal_hits,
        "causal_correlator_accuracy_pct": round((causal_hits / n_trials) * 100.0, 1),
        "arrival_order_correct": arrival_hits,
        "arrival_order_accuracy_pct": round((arrival_hits / n_trials) * 100.0, 1),
        "accuracy_gain_pts": round(((causal_hits - arrival_hits) / n_trials) * 100.0, 1),
        "wall_clock_ms": wall_ms,
        "scenarios_available": list(DISTRIBUTED_FAULT_SCENARIOS.values()),
        "sample_trial": sample_trial_detail,
    }


# ---------------------------------------------------------------------------
# Part 2: Live Serverless AWS Bedrock Model Profiler (Nova Micro vs Nova Lite)
# ---------------------------------------------------------------------------

# Official AWS Bedrock On-Demand US Cross-Region Inference Profile pricing (USD per 1,000,000 tokens)
BEDROCK_MODEL_PRICING_PER_1M: Dict[str, Dict[str, Any]] = {
    "us.amazon.nova-micro-v1:0": {
        "label": "Amazon Nova Micro",
        "input_usd_per_1m": 0.035,
        "output_usd_per_1m": 0.140,
    },
    "us.amazon.nova-lite-v1:0": {
        "label": "Amazon Nova Lite",
        "input_usd_per_1m": 0.060,
        "output_usd_per_1m": 0.240,
    },
}

DEFAULT_BENCHMARK_PROMPTS: List[str] = [
    (
        "A 64-worker distributed training step logged 1 ERR_WORKER_SLAB_OOM_0x41 at t=0ms on worker-19 "
        "followed by 63 ERR_ALLREDUCE_BARRIER_TIMEOUT alarms at t=1250ms. In 2 concise bullet points, "
        "identify the root cause and immediate recovery action."
    ),
    (
        "Explain in 2 short sentences why online-softmax block-tiled FlashAttention avoids materializing "
        "the O(N^2) score matrix while remaining numerically identical to standard softmax attention."
    ),
    (
        "Given 64 concurrent multi-turn LLM requests sharing a 256-token system prompt, explain in 2 "
        "concise bullets how block-size-16 Paged KV-Cache with Copy-on-Write eliminates internal fragmentation."
    ),
    (
        "In an ML computation graph, why does fusing MatMul + AddBias + GELU into a single epilogue kernel "
        "reduce peak intermediate memory allocation? Answer in 2 concise sentences."
    ),
    (
        "Compare latency-optimized micro models vs multimodal-capable lite models for high-QPS automated "
        "incident log triage in 2 short bullet points."
    ),
]


def _percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return sorted_vals[lo]
    frac = rank - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def _build_bedrock_client(region_name: str = "us-west-2") -> Any:
    """Create real `boto3` bedrock-runtime client (supports Lambda IAM Role + local `aws login` profile)."""
    import boto3  # type: ignore

    try:
        client = boto3.client("bedrock-runtime", region_name=region_name)
        # Trigger credential resolution eagerly so MissingDependencyException is caught if `aws login` is active
        sess = boto3.DEFAULT_SESSION or boto3.Session(region_name=region_name)
        creds = sess.get_credentials()
        if creds is not None:
            creds.get_frozen_credentials()
        return client
    except Exception as exc:
        if "MissingDependencyException" not in type(exc).__name__ and "botocore[crt]" not in str(exc):
            raise

    aws_bin = shutil.which("aws")
    if not aws_bin:
        raise RuntimeError("AWS CLI not found to export `aws login` session credentials for boto3.")

    profile = os.environ.get("AWS_PROFILE", "portfolio")
    env = dict(os.environ)
    env.pop("AWS_ACCESS_KEY_ID", None)
    env.pop("AWS_SECRET_ACCESS_KEY", None)
    env.pop("AWS_SESSION_TOKEN", None)

    proc = subprocess.run(
        [aws_bin, "configure", "export-credentials", "--profile", profile, "--format", "process"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    exported = json.loads(proc.stdout)
    session = boto3.Session(
        aws_access_key_id=exported["AccessKeyId"],
        aws_secret_access_key=exported["SecretAccessKey"],
        aws_session_token=exported.get("SessionToken"),
        region_name=region_name,
    )
    return session.client("bedrock-runtime", region_name=region_name)


class BedrockModelProfiler:
    """Profiles real serverless AWS Bedrock inference (`us.amazon.nova-micro-v1:0` vs `us.amazon.nova-lite-v1:0`)."""

    def __init__(
        self,
        region_name: str = "us-west-2",
        converse_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> None:
        self.region_name = region_name
        self._converse_fn = converse_fn
        self._client: Any = None

    def _invoke_converse(self, model_id: str, prompt: str, max_tokens: int = 120) -> Tuple[Dict[str, Any], float]:
        system_prompt = (
            "You are a concise ML systems engineer. Respond accurately in under 75 words with concrete technical precision."
        )
        if self._converse_fn is not None:
            t0 = time.perf_counter()
            resp = self._converse_fn(
                modelId=model_id,
                system=[{"text": system_prompt}],
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
            )
            wall_ms = (time.perf_counter() - t0) * 1000.0
            return resp, wall_ms

        if self._client is None:
            self._client = _build_bedrock_client(region_name=self.region_name)

        t0 = time.perf_counter()
        resp = self._client.converse(
            modelId=model_id,
            system=[{"text": system_prompt}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
        )
        wall_ms = (time.perf_counter() - t0) * 1000.0
        return resp, wall_ms

    def profile_model(
        self,
        model_id: str,
        prompts: List[str],
        max_tokens: int = 120,
        warmup: bool = True,
    ) -> Dict[str, Any]:
        if model_id not in BEDROCK_MODEL_PRICING_PER_1M:
            raise ValueError(f"Unsupported model_id for pricing lookup: {model_id}")

        pricing = BEDROCK_MODEL_PRICING_PER_1M[model_id]

        # Connection warmup so TCP/TLS handshake does not distort measured inference p50
        warmup_ms: Optional[float] = None
        if warmup:
            _, w_ms = self._invoke_converse(model_id=model_id, prompt="Reply OK.", max_tokens=6)
            warmup_ms = round(w_ms, 2)

        samples: List[Dict[str, Any]] = []
        for idx, prompt in enumerate(prompts):
            resp, wall_ms = self._invoke_converse(model_id=model_id, prompt=prompt, max_tokens=max_tokens)
            usage = resp.get("usage") or {}
            in_tok = int(usage.get("inputTokens", 0))
            out_tok = int(usage.get("outputTokens", 0))
            tot_tok = int(usage.get("totalTokens", in_tok + out_tok))
            server_ms = float((resp.get("metrics") or {}).get("latencyMs", wall_ms))
            text = (
                (((resp.get("output") or {}).get("message") or {}).get("content") or [{}])[0].get("text", "")
            ).strip()

            req_cost_usd = (
                (in_tok / 1_000_000.0) * pricing["input_usd_per_1m"]
                + (out_tok / 1_000_000.0) * pricing["output_usd_per_1m"]
            )
            tok_per_s = round(out_tok / max(0.001, server_ms / 1000.0), 2)
            wall_tok_per_s = round(out_tok / max(0.001, wall_ms / 1000.0), 2)

            samples.append(
                {
                    "sample_index": idx + 1,
                    "prompt_preview": prompt[:96] + ("..." if len(prompt) > 96 else ""),
                    "wall_clock_ms": round(wall_ms, 2),
                    "bedrock_latency_ms": round(server_ms, 2),
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "total_tokens": tot_tok,
                    "server_tokens_per_sec": tok_per_s,
                    "e2e_tokens_per_sec": wall_tok_per_s,
                    "request_cost_usd": round(req_cost_usd, 7),
                    "response_text": text,
                }
            )

        wall_sorted = sorted(s["wall_clock_ms"] for s in samples)
        srv_sorted = sorted(s["bedrock_latency_ms"] for s in samples)
        total_in = sum(s["input_tokens"] for s in samples)
        total_out = sum(s["output_tokens"] for s in samples)
        total_srv_sec = sum(s["bedrock_latency_ms"] for s in samples) / 1000.0
        total_wall_sec = sum(s["wall_clock_ms"] for s in samples) / 1000.0
        total_cost_usd = sum(s["request_cost_usd"] for s in samples)
        n = max(1, len(samples))

        usd_per_1k_requests = round((total_cost_usd / n) * 1000.0, 5)

        return {
            "model_id": model_id,
            "label": pricing["label"],
            "region": self.region_name,
            "num_requests": len(samples),
            "tls_warmup_ms": warmup_ms,
            "p50_wall_ms": round(_percentile(wall_sorted, 50.0), 2),
            "p95_wall_ms": round(_percentile(wall_sorted, 95.0), 2),
            "mean_wall_ms": round(sum(wall_sorted) / n, 2),
            "p50_bedrock_ms": round(_percentile(srv_sorted, 50.0), 2),
            "p95_bedrock_ms": round(_percentile(srv_sorted, 95.0), 2),
            "mean_bedrock_ms": round(sum(srv_sorted) / n, 2),
            "tokens_per_sec": round(total_out / max(0.001, total_srv_sec), 2),
            "e2e_tokens_per_sec": round(total_out / max(0.001, total_wall_sec), 2),
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "avg_input_tokens": round(total_in / n, 1),
            "avg_output_tokens": round(total_out / n, 1),
            "usd_per_1k_requests": usd_per_1k_requests,
            "actual_run_cost_usd": round(total_cost_usd, 6),
            "pricing_per_1m_tokens_usd": {
                "input": pricing["input_usd_per_1m"],
                "output": pricing["output_usd_per_1m"],
            },
            "samples": samples,
        }

    def compare_nova_micro_vs_lite(
        self,
        prompts: Optional[List[str]] = None,
        max_tokens: int = 120,
        warmup: bool = True,
    ) -> Dict[str, Any]:
        eval_prompts = prompts if prompts else list(DEFAULT_BENCHMARK_PROMPTS)
        t0 = time.perf_counter()
        micro = self.profile_model(
            model_id="us.amazon.nova-micro-v1:0",
            prompts=eval_prompts,
            max_tokens=max_tokens,
            warmup=warmup,
        )
        lite = self.profile_model(
            model_id="us.amazon.nova-lite-v1:0",
            prompts=eval_prompts,
            max_tokens=max_tokens,
            warmup=warmup,
        )
        suite_wall_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        p50_speedup_x = round(lite["p50_bedrock_ms"] / max(0.1, micro["p50_bedrock_ms"]), 2)
        tok_s_ratio_x = round(micro["tokens_per_sec"] / max(0.1, lite["tokens_per_sec"]), 2)
        cost_savings_pct = round(
            ((lite["usd_per_1k_requests"] - micro["usd_per_1k_requests"]) / max(1e-9, lite["usd_per_1k_requests"]))
            * 100.0,
            1,
        )

        return {
            "region": self.region_name,
            "prompts_evaluated_per_model": len(eval_prompts),
            "total_suite_wall_ms": suite_wall_ms,
            "total_benchmark_cost_usd": round(
                micro["actual_run_cost_usd"] + lite["actual_run_cost_usd"],
                6,
            ),
            "nova_micro": micro,
            "nova_lite": lite,
            "head_to_head": {
                "micro_vs_lite_p50_speedup_x": p50_speedup_x,
                "micro_vs_lite_tok_s_ratio_x": tok_s_ratio_x,
                "micro_cost_savings_vs_lite_pct": cost_savings_pct,
            },
        }
