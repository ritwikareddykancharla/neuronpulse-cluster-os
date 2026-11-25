"""Unit & Lambda API Integration Tests for NeuronPulse v2 Real ML Systems Suite."""
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
from neuronpulse.quant_spec_transport import (
    LockFreeSQCQRing,
    MRIntervalTreeCache,
    run_aot_bucket_and_packing_benchmark,
    run_quant_spec_suite,
    run_quantization_moe_benchmark,
    run_speculative_decoding_benchmark,
    run_transport_ring_mr_benchmark,
)
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

    def test_07_smoothquant_w8a8_w4a16_moe_and_tp_sharding(self) -> None:
        qm = run_quantization_moe_benchmark(batch_tokens=48, in_features=128, out_features=256, alpha=0.5, group_size=32)
        fb = qm["footprint_bytes"]
        self.assertEqual(fb["w8a8_compression_vs_fp32_x"], 4.0)
        self.assertEqual(fb["w4a16_compression_vs_fp16_x"], 3.56)
        self.assertEqual(fb["w4a16_effective_bits_per_weight"], 4.5)

        # SmoothQuant alpha=0.5 must substantially improve SQNR & cosine similarity over naive W8A8
        self.assertGreater(qm["smoothquant_w8a8"]["cosine_similarity"], 0.999)
        self.assertGreater(qm["smoothquant_w8a8"]["sqnr_gain_over_naive_db"], 8.0)
        self.assertGreater(qm["smoothquant_w8a8"]["max_error_reduction_x"], 2.5)
        self.assertGreater(qm["group_w4a16"]["cosine_similarity"], 0.99)

        # Top-2 SwiGLU MoE load balancing & TP=4 multi-rank sharding parity
        moe = qm["moe_swiglu"]
        self.assertEqual(moe["active_expert_fraction_pct"], 25.0)
        self.assertGreater(moe["unbalanced_router"]["drop_rate_pct"], moe["balanced_router"]["drop_rate_pct"])
        self.assertLess(moe["balanced_router"]["load_cv"], moe["unbalanced_router"]["load_cv"])
        self.assertTrue(moe["tp_sharding_verification"]["numerically_equivalent"])
        self.assertLess(moe["tp_sharding_verification"]["max_abs_diff"], 1e-12)

    def test_08_lossless_speculative_decoding_and_dp_aot_bucketing_bfd(self) -> None:
        sd = run_speculative_decoding_benchmark(gamma=4, num_target_tokens=256, draft_alignment=0.80)
        self.assertTrue(sd["lossless_distribution_preserved"])
        self.assertLess(sd["exact_marginal_max_abs_diff"], 1e-12)
        self.assertGreaterEqual(sd["tokens_per_target_step_x"], 2.3)
        self.assertLessEqual(sd["tokens_per_target_step_x"], 3.4)
        self.assertLess(sd["monte_carlo_tvd_speculative"], sd["monte_carlo_tvd_unverified_draft"])
        self.assertEqual(sd["kv_final_consistent_slots"], sd["emitted_tokens"] + 2)

        aot = run_aot_bucket_and_packing_benchmark(num_sequences=256, max_buckets=5, max_seq_len=2048)
        self.assertTrue(aot["dp_optimal_buckets"]["verified_global_optimum"])
        self.assertLessEqual(
            aot["dp_optimal_buckets"]["padding_waste_tokens"],
            aot["pow2_static_buckets"]["padding_waste_tokens"],
        )
        self.assertLess(
            aot["dp_optimal_buckets"]["padding_waste_pct"],
            aot["single_static_bucket"]["padding_waste_pct"] * 0.4,
        )

        bfd = aot["bfd_sequence_packing"]
        self.assertTrue(bfd["zero_cross_doc_leakage_verified"])
        self.assertEqual(bfd["cross_doc_attention_leakage"], 0.0)
        self.assertLess(bfd["bfd_padding_waste_pct"], 4.0)
        self.assertGreater(bfd["bin_reduction_x"], 2.5)

    def test_09_lock_free_sq_cq_ring_and_mr_interval_tree_cache(self) -> None:
        ring = LockFreeSQCQRing(capacity=64)
        for i in range(32):
            self.assertTrue(ring.post_send(wr_id=i, vaddr=0x1000 * (i + 1), length=4096, lkey=7, imm_seq=i))
        self.assertEqual(ring.sq_occupancy, 32)
        progressed = ring.progress_srd_multipath(ooo_stride=3)
        self.assertEqual(progressed, 32)
        self.assertEqual(ring.sq_occupancy, 0)
        self.assertEqual(ring.cq_occupancy, 32)
        reaped, nbytes = ring.poll_cq_batch(max_reap=64)
        self.assertEqual(reaped, 32)
        self.assertEqual(nbytes, 32 * 4096)
        self.assertEqual(ring.cq_occupancy, 0)

        cache = MRIntervalTreeCache()
        lkey1, hit1 = cache.acquire_mr(0x7000_0000, 1024 * 1024)
        self.assertFalse(hit1)
        lkey2, hit2 = cache.acquire_mr(0x7000_0000 + 65536, 256 * 1024)
        self.assertTrue(hit2)
        self.assertEqual(lkey1, lkey2)
        self.assertEqual(cache.invalidate_range(0x7000_8000, 4096), 1)
        lkey3, hit3 = cache.acquire_mr(0x7000_0000 + 65536, 256 * 1024)
        self.assertFalse(hit3)
        self.assertNotEqual(lkey1, lkey3)

        tr = run_transport_ring_mr_benchmark(ring_capacity=256, num_messages=2048, num_mr_regions=128, num_mr_lookups=512)
        self.assertGreater(tr["sq_cq_ring"]["heap_alloc_reduction_pct"], 95.0)
        self.assertTrue(tr["mr_interval_cache"]["avl_balanced_verified"])
        self.assertTrue(tr["mr_interval_cache"]["invalidation_safety_check"]["verified"])
        self.assertGreater(tr["mr_interval_cache"]["mr_cache_speedup_x"], 5.0)

    def test_10_quant_spec_and_transport_ring_api_routes(self) -> None:
        qs_get = lambda_handler({"rawPath": "/api/quant-spec", "requestContext": {"http": {"method": "GET"}}})
        self.assertEqual(qs_get["statusCode"], 200)
        qs_body = json.loads(qs_get["body"])
        self.assertEqual(qs_body["quantization_and_moe"]["footprint_bytes"]["w8a8_compression_vs_fp32_x"], 4.0)
        self.assertEqual(qs_body["quantization_and_moe"]["footprint_bytes"]["w4a16_compression_vs_fp16_x"], 3.56)
        self.assertTrue(qs_body["speculative_decoding"]["lossless_distribution_preserved"])

        tr_post = lambda_handler(
            {
                "rawPath": "/api/transport/ring",
                "requestContext": {"http": {"method": "POST"}},
                "body": json.dumps({"ring_capacity": 256, "num_messages": 2048, "num_mr_lookups": 512}),
            }
        )
        self.assertEqual(tr_post["statusCode"], 200)
        tr_body = json.loads(tr_post["body"])
        self.assertTrue(tr_body["mr_interval_cache"]["avl_balanced_verified"])

        qs_full = run_quant_spec_suite()
        self.assertGreater(qs_full["suite_wall_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
