"""Unit and API Integration Tests for NeuronPulse Trainium2 UltraServer OS."""
from __future__ import annotations

import json
import os
import unittest

from neuronpulse.correlator import CrossNodeErrorCorrelator, execute_automated_repair
from neuronpulse.profiler import compare_serving_configs, simulate_neuron_serving
from neuronpulse.server import lambda_handler, route_api_request
from neuronpulse.topology import UltraServerCluster


class TestNeuronPulseSuite(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["NEURONPULSE_OFFLINE_LLM"] = "1"
        route_api_request("POST", "/api/reset", {}, {})

    def test_01_nominal_64_neuroncore_topology(self) -> None:
        cluster = UltraServerCluster()
        snap = cluster.snapshot()
        self.assertEqual(snap["summary"]["total_neuron_cores"], 64)
        self.assertEqual(snap["summary"]["healthy_cores"], 64)
        self.assertEqual(len(snap["active_nodes"]), 4)
        self.assertEqual(len(snap["efa_links"]), 4)

    def test_02_hbm_ecc_injection_and_100pct_root_isolation(self) -> None:
        cluster = UltraServerCluster()
        cluster.inject_fault("fault_hbm_ecc")
        corr = CrossNodeErrorCorrelator().correlate(list(reversed(cluster.raw_events)))
        self.assertEqual(corr["status"], "ROOT_CAUSE_ISOLATED")
        self.assertEqual(corr["raw_alert_count"], 64)
        self.assertEqual(corr["suppressed_symptom_count"], 63)
        self.assertEqual(corr["root_cause"]["node_id"], "trn2-node-02")
        self.assertEqual(corr["root_cause"]["core_id"], "nc-09")
        self.assertEqual(corr["root_cause"]["global_rank"], 25)
        self.assertTrue(corr["requires_node_replacement"])

    def test_03_efa_straggler_840ms_and_warm_spare_swap_74s(self) -> None:
        cluster = UltraServerCluster()
        snap = cluster.inject_fault("fault_efa_straggler")
        self.assertEqual(snap["summary"]["collective_ring_p99_ms"], 840.0)

        rep = execute_automated_repair(cluster)
        self.assertEqual(rep["status"], "REPAIR_COMPLETED")
        self.assertEqual(rep["repair"]["impaired_node"], "trn2-node-04")
        self.assertEqual(rep["repair"]["replacement_node"], "trn2-spare-01")
        self.assertEqual(rep["repair"]["automated_recovery_sec"], 74.0)
        self.assertEqual(rep["repair"]["manual_reboot_baseline_sec"], 2520.0)
        self.assertIn("trn2-spare-01", rep["snapshot"]["active_nodes"])
        self.assertEqual(rep["snapshot"]["summary"]["healthy_cores"], 64)

    def test_04_kv_fragmentation_avoids_false_node_cordon(self) -> None:
        cluster = UltraServerCluster()
        cluster.inject_fault("fault_kv_frag")
        rep = execute_automated_repair(cluster)
        self.assertEqual(rep["repair"]["workflow_type"], "ONLINE_PAGED_KV_COMPACTION")
        self.assertEqual(cluster.cordoned_nodes, [])
        self.assertEqual(rep["snapshot"]["summary"]["healthy_cores"], 64)

    def test_05_nki_profiler_oom_and_fp8_speedup(self) -> None:
        # BF16 at TP=2 on 70B exceeds 48 GB/core HBM3e
        oom_cfg = simulate_neuron_serving(tp_degree=2, precision="BF16", batch_size=16, nki_flash_fusion=False)
        self.assertFalse(oom_cfg["fits_in_hbm"])
        self.assertEqual(oom_cfg["bottleneck"], "HBM_OOM_EXCEEDED")

        # FP8 + TP=8 + NKI FlashAttention fits cleanly and beats BF16 unfused
        cmp_out = compare_serving_configs(tp_degree=8, precision="FP8", batch_size=16, nki_flash_fusion=True)
        self.assertTrue(cmp_out["selected"]["fits_in_hbm"])
        self.assertGreater(cmp_out["vs_bf16_unfused_baseline"]["throughput_speedup_x"], 1.8)
        self.assertGreater(cmp_out["vs_bf16_unfused_baseline"]["ttft_reduction_pct"], 40.0)

    def test_06_lambda_api_routes_end_to_end(self) -> None:
        h = lambda_handler({"rawPath": "/api/health", "requestContext": {"http": {"method": "GET"}}})
        self.assertEqual(h["statusCode"], 200)
        self.assertEqual(json.loads(h["body"])["total_neuron_cores"], 64)

        inj = lambda_handler(
            {
                "rawPath": "/api/faults/inject",
                "requestContext": {"http": {"method": "POST"}},
                "body": json.dumps({"fault_id": "fault_hbm_ecc"}),
            }
        )
        self.assertEqual(inj["statusCode"], 200)
        inj_data = json.loads(inj["body"])
        self.assertEqual(inj_data["correlation"]["root_cause"]["node_id"], "trn2-node-02")

        ask = lambda_handler(
            {
                "rawPath": "/api/copilot/ask",
                "requestContext": {"http": {"method": "POST"}},
                "body": json.dumps({"question": "Diagnose active ECC trap and give neuron-ls commands"}),
            }
        )
        self.assertEqual(ask["statusCode"], 200)
        self.assertIn("trn2-node-02", json.loads(ask["body"])["answer"])


if __name__ == "__main__":
    unittest.main()
