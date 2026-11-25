"""Unit & Lambda API Integration Tests for NeuronPulse Real ML Systems Suite."""
from __future__ import annotations

import json
import unittest
from typing import Any, Dict

import neuronpulse.server as server_mod
from neuronpulse.compiler_ir import run_compiler_pass_benchmark
from neuronpulse.correlator import (
    BedrockModelProfiler,
    CrossNodeErrorCorrelator,
    build_worker_fault_stream,
    run_correlator_benchmark,
)
from neuronpulse.flash_attention import benchmark_attention_pair, run_attention_scaling_sweep
from neuronpulse.paged_kv import run_paged_kv_benchmark
from neuronpulse.server import lambda_handler


def _mock_bedrock_converse(**kwargs: Any) -> Dict[str, Any]:
    model_id = kwargs.get("modelId", "")
    is_micro = "nova-micro" in model_id
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "text": (
                            f"[{model_id}] Isolate root rank first and suppress downstream barrier timeouts."
                        )
                    }
                ],
            }
        },
        "usage": {
            "inputTokens": 62,
            "outputTokens": 28 if is_micro else 34,
            "totalTokens": 90 if is_micro else 96,
        },
        "metrics": {
            "latencyMs": 240 if is_micro else 390,
        },
    }


class TestNeuronPulseRealMLSystems(unittest.TestCase):
    def setUp(self) -> None:
        server_mod._TEST_BEDROCK_CONVERSE_FN = _mock_bedrock_converse

    def tearDown(self) -> None:
        server_mod._TEST_BEDROCK_CONVERSE_FN = None

    def test_01_compiler_ir_dce_cse_fusion_equivalence_and_memory_savings(self) -> None:
        for preset in ("transformer_mlp_block", "stacked_2layer_mlp", "wide_ffn_epilogue"):
            res = run_compiler_pass_benchmark(preset=preset, batch_tokens=32, hidden_dim=32, repeats=1)
            self.assertTrue(res["numerically_equivalent"])
            self.assertLess(res["max_abs_diff"], 1e-9)
            self.assertLess(res["optimized_op_nodes"], res["unoptimized_op_nodes"])
            self.assertGreater(res["peak_memory_saved_pct"], 30.0)
            self.assertIn("FusedMatMulBiasGelu", res["optimized_ir"])
            self.assertNotIn("debug_probe_mm", res["optimized_ir"])

    def test_02_tiled_flash_attention_exactness_and_quadratic_memory_reduction(self) -> None:
        sweep = run_attention_scaling_sweep(seq_lengths=(128, 256, 512), head_dim=32, block_size=32, causal=False)
        for row in sweep["sweep"]:
            self.assertTrue(row["numerically_exact"])
            self.assertLess(row["max_abs_diff"], 1e-9)

        n512 = sweep["n512_summary"]
        self.assertGreater(n512["memory_reduction_pct"], 85.0)
        self.assertGreater(sweep["scaling_proof"]["naive_memory_growth_x"], 10.0)

        # Also verify causal tiled attention skips upper-triangular tiles cleanly
        causal_pair = benchmark_attention_pair(seq_len=256, head_dim=32, block_size=64, causal=True)
        self.assertTrue(causal_pair["numerically_exact"])
        self.assertLess(causal_pair["max_abs_diff"], 1e-9)
        self.assertGreater(causal_pair["tiles_skipped_causal"], 0)

    def test_03_paged_kv_allocator_fragmentation_prefix_sharing_and_cow(self) -> None:
        res = run_paged_kv_benchmark(
            num_requests=64,
            block_size=16,
            max_seq_len=1024,
            system_prompt_tokens=256,
            memory_cap_mb=1280,
        )
        self.assertEqual(res["workload"]["num_requests"], 64)
        self.assertEqual(res["workload"]["min_seq_len"], 48)
        self.assertEqual(res["workload"]["max_seq_len"], 1024)
        self.assertLess(
            res["paged_cow_allocator"]["internal_fragmentation_pct"],
            5.0,
        )
        self.assertGreater(
            res["contiguous_baseline"]["internal_fragmentation_pct"],
            35.0,
        )
        self.assertGreater(res["comparison"]["memory_reduction_pct"], 50.0)
        self.assertGreater(res["paged_cow_allocator"]["cow_prefix_hit_rate_pct"], 30.0)
        self.assertGreater(
            res["comparison"]["paged_max_concurrent"],
            res["comparison"]["contiguous_max_concurrent"] * 2,
        )
        self.assertTrue(res["cow_isolation_check"]["verified"])
        self.assertEqual(res["cow_isolation_check"]["peer_corruptions"], 0)

    def test_04_distributed_fault_correlator_48_shuffled_streams(self) -> None:
        events, truth = build_worker_fault_stream(
            scenario_id="rpc_socket_reset",
            num_workers=64,
            root_rank=27,
            shuffle_seed=99,
        )
        single = CrossNodeErrorCorrelator().correlate(events)
        self.assertEqual(single["status"], "ROOT_CAUSE_ISOLATED")
        self.assertEqual(single["root_cause"]["worker_id"], truth["expected_worker_id"])
        self.assertEqual(single["suppressed_symptom_count"], 63)

        bench = run_correlator_benchmark(trials=48, num_workers=64, seed=42)
        self.assertEqual(bench["causal_correlator_accuracy_pct"], 100.0)
        self.assertLess(bench["arrival_order_accuracy_pct"], 15.0)
        self.assertEqual(bench["total_raw_worker_alarms"], 3072)

    def test_05_bedrock_profiler_metrics_aggregation(self) -> None:
        profiler = BedrockModelProfiler(region_name="us-west-2", converse_fn=_mock_bedrock_converse)
        cmp_res = profiler.compare_nova_micro_vs_lite(prompts=["Diagnose ring barrier timeout."], warmup=False)
        self.assertEqual(cmp_res["nova_micro"]["p50_bedrock_ms"], 240.0)
        self.assertEqual(cmp_res["nova_lite"]["p50_bedrock_ms"], 390.0)
        self.assertGreater(cmp_res["head_to_head"]["micro_cost_savings_vs_lite_pct"], 30.0)

    def test_06_lambda_api_routes_end_to_end(self) -> None:
        h = lambda_handler({"rawPath": "/api/health", "requestContext": {"http": {"method": "GET"}}})
        self.assertEqual(h["statusCode"], 200)
        h_body = json.loads(h["body"])
        self.assertEqual(h_body["ec2_or_gpu_instances"], 0)

        comp = lambda_handler(
            {
                "rawPath": "/api/compiler/optimize",
                "requestContext": {"http": {"method": "POST"}},
                "body": json.dumps({"preset": "transformer_mlp_block", "batch_tokens": 32, "hidden_dim": 32}),
            }
        )
        self.assertEqual(comp["statusCode"], 200)
        self.assertTrue(json.loads(comp["body"])["numerically_equivalent"])

        attn = lambda_handler(
            {
                "rawPath": "/api/attention/run",
                "requestContext": {"http": {"method": "POST"}},
                "body": json.dumps({"seq_len": 128, "head_dim": 32, "block_size": 32}),
            }
        )
        self.assertEqual(attn["statusCode"], 200)
        self.assertTrue(json.loads(attn["body"])["numerically_exact"])

        kv = lambda_handler(
            {
                "rawPath": "/api/kvcache/run",
                "requestContext": {"http": {"method": "POST"}},
                "body": json.dumps({"num_requests": 32, "block_size": 16}),
            }
        )
        self.assertEqual(kv["statusCode"], 200)
        self.assertTrue(json.loads(kv["body"])["cow_isolation_check"]["verified"])

        corr = lambda_handler(
            {
                "rawPath": "/api/correlator/run",
                "requestContext": {"http": {"method": "POST"}},
                "body": json.dumps({"scenario_id": "oom_allocator_trap", "num_workers": 64}),
            }
        )
        self.assertEqual(corr["statusCode"], 200)
        self.assertEqual(
            json.loads(corr["body"])["shuffled_suite_eval"]["causal_correlator_accuracy_pct"],
            100.0,
        )

        br = lambda_handler(
            {
                "rawPath": "/api/bedrock/compare",
                "requestContext": {"http": {"method": "POST"}},
                "body": json.dumps({"prompt": "Why fuse MatMul+Bias+GELU?", "warmup": False}),
            }
        )
        self.assertEqual(br["statusCode"], 200)
        self.assertIn("nova_micro", json.loads(br["body"]))


if __name__ == "__main__":
    unittest.main()
