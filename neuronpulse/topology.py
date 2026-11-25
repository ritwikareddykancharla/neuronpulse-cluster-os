"""64-NeuronCore Trainium2 UltraServer Topology, EFA All-Reduce Ring & Fault Injection Engine."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


FAULT_CATALOG: Dict[str, Dict[str, Any]] = {
    "fault_hbm_ecc": {
        "fault_id": "fault_hbm_ecc",
        "title": "HBM3e Uncorrectable Double-Bit ECC on trn2-node-02:nc-09",
        "short_label": "Fault 1 · HBM3e Double-Bit ECC (trn2-node-02:nc-09)",
        "category": "HARDWARE_HBM_ECC",
        "target_node": "trn2-node-02",
        "target_core": "nc-09",
        "target_rank": 25,
        "target_efa": "efa-2",
        "hardware_code": "NEURON_HW_ERR_HBM_DBE_0x4E02",
        "symptom_code": "NCCOM_ALLREDUCE_RING_TIMEOUT_0x7A11",
        "summary": (
            "Physical HBM3e Stack-2 Bank-5 raised an uncorrectable Double-Bit ECC trap (0x4E02) on "
            "trn2-node-02:nc-09 (Rank 25), halting its DMA engine and causing all 63 peer NeuronCores "
            "to stall 1.84s later inside nccomAllReduce."
        ),
    },
    "fault_efa_straggler": {
        "fault_id": "fault_efa_straggler",
        "title": "EFA All-Reduce Collective Ring Straggler on trn2-node-04:efa-2 (p99 stall 840ms)",
        "short_label": "Fault 2 · EFA Ring Straggler (trn2-node-04:efa-2 · 840ms)",
        "category": "NETWORK_EFA_STRAGGLER",
        "target_node": "trn2-node-04",
        "target_core": "nc-08",
        "target_rank": 56,
        "target_efa": "efa-2",
        "hardware_code": "EFA_SRD_RNR_RETRY_SURGE_0x9C14",
        "symptom_code": "COLLECTIVE_STEP_STRAGGLER_840MS",
        "summary": (
            "Optical transceiver lane degradation (-14.8 dBm Rx) on trn2-node-04:efa-2 triggered a 19.4% "
            "SRD packet retransmission surge across Ranks 56-59, inflating 64-rank All-Reduce p99 latency "
            "from 4.2ms to 840.0ms."
        ),
    },
    "fault_kv_frag": {
        "fault_id": "fault_kv_frag",
        "title": "Continuous Batching KV-Cache Fragmentation Stall",
        "short_label": "Fault 3 · KV-Cache Fragmentation Stall (Serving)",
        "category": "RUNTIME_KV_FRAGMENTATION",
        "target_node": "trn2-node-01",
        "target_core": "nc-03",
        "target_rank": 3,
        "target_efa": "efa-0",
        "hardware_code": "NRT_KV_BLOCK_ALLOC_STALL_0x3B08",
        "symptom_code": "SERVING_TTFT_SLA_BREACH_412MS",
        "summary": (
            "Variable-length 32K prompt bursts fragmented contiguous HBM3e KV-cache slabs (44.6% external "
            "fragmentation), causing continuous-batching scheduler stalls and spiking TTFT p99 to 412ms "
            "with zero underlying hardware ECC or EFA link errors."
        ),
    },
}


@dataclass
class NeuronCoreTelemetry:
    global_rank: int
    node_id: str
    core_id: str
    efa_nic: str
    status: str  # HEALTHY | ROOT_FAULT | RING_STALLED | KV_PRESSURE | SPARE_ACTIVE
    hbm3e_bw_gbps: float
    hbm_used_gb: float
    hbm_total_gb: float
    temp_c: float
    flops_util_pct: float
    collective_p99_ms: float
    kv_frag_pct: float
    ecc_sbe: int
    ecc_dbe: int
    error_code: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class UltraServerCluster:
    """Stateful 64-NeuronCore Trainium2 UltraServer (`4 trn2 Nodes x 16 NeuronCores` + `trn2-spare-01`)."""

    def __init__(self) -> None:
        self.cluster_id = "ultraserver-trn2-sea-07"
        self.availability_zone = "us-west-2b"
        self.nodes = ["trn2-node-01", "trn2-node-02", "trn2-node-03", "trn2-node-04"]
        self.warm_spare_node = "trn2-spare-01"
        self.spare_state = "STANDBY_READY"
        self.cordoned_nodes: List[str] = []
        self.active_fault_id: Optional[str] = None
        self.last_correlation: Optional[Dict[str, Any]] = None
        self.last_repair: Optional[Dict[str, Any]] = None
        self.raw_events: List[Dict[str, Any]] = []
        self.cores: List[NeuronCoreTelemetry] = []
        self.efa_links: List[Dict[str, Any]] = []
        self.reset()

    def reset(self) -> None:
        self.nodes = ["trn2-node-01", "trn2-node-02", "trn2-node-03", "trn2-node-04"]
        self.spare_state = "STANDBY_READY"
        self.cordoned_nodes = []
        self.active_fault_id = None
        self.last_correlation = None
        self.last_repair = None
        self.raw_events = []
        self.cores = self._build_nominal_cores(self.nodes)
        self.efa_links = self._build_nominal_efa_links(self.nodes)

    def _build_nominal_cores(self, node_list: List[str]) -> List[NeuronCoreTelemetry]:
        cores: List[NeuronCoreTelemetry] = []
        for n_idx, node_id in enumerate(node_list):
            for c_idx in range(16):
                rank = n_idx * 16 + c_idx
                efa_nic = f"efa-{c_idx // 4}"
                # Deterministic micro-variation so the 64-core grid looks alive and realistic
                jitter = ((rank * 7 + 3) % 11) - 5
                status = "SPARE_ACTIVE" if node_id == self.warm_spare_node else "HEALTHY"
                cores.append(
                    NeuronCoreTelemetry(
                        global_rank=rank,
                        node_id=node_id,
                        core_id=f"nc-{c_idx:02d}",
                        efa_nic=efa_nic,
                        status=status,
                        hbm3e_bw_gbps=round(2890.0 + jitter * 14.5, 1),
                        hbm_used_gb=round(34.2 + (rank % 4) * 0.6, 1),
                        hbm_total_gb=48.0,
                        temp_c=round(57.5 + jitter * 0.6, 1),
                        flops_util_pct=round(91.4 + jitter * 0.45, 1),
                        collective_p99_ms=round(4.2 + (rank % 3) * 0.15, 2),
                        kv_frag_pct=round(7.8 + (rank % 5) * 0.4, 1),
                        ecc_sbe=0,
                        ecc_dbe=0,
                        error_code="NONE",
                    )
                )
        return cores

    def _build_nominal_efa_links(self, node_list: List[str]) -> List[Dict[str, Any]]:
        links: List[Dict[str, Any]] = []
        for idx, src in enumerate(node_list):
            dst = node_list[(idx + 1) % len(node_list)]
            links.append(
                {
                    "ring_hop": f"{src} -> {dst}",
                    "src_node": src,
                    "dst_node": dst,
                    "active_nics": "efa-0..efa-3 (4x 800Gbps SRD)",
                    "bandwidth_gbps": 3120.0,
                    "p99_allreduce_ms": 4.2,
                    "srd_retransmit_pct": 0.02,
                    "optical_rx_dbm": -2.1,
                    "status": "HEALTHY",
                }
            )
        return links

    def inject_fault(self, fault_id: str) -> Dict[str, Any]:
        if fault_id not in FAULT_CATALOG:
            raise KeyError(f"Unknown fault_id: {fault_id}")

        # Always start from clean 4-node baseline so repeat demos are crisp and deterministic
        self.reset()
        self.active_fault_id = fault_id
        spec = FAULT_CATALOG[fault_id]
        base_ns = 1_764_038_400_000_000_000
        events: List[Dict[str, Any]] = []

        if fault_id == "fault_hbm_ecc":
            for core in self.cores:
                if core.node_id == "trn2-node-02" and core.core_id == "nc-09":
                    core.status = "ROOT_FAULT"
                    core.ecc_dbe = 2
                    core.ecc_sbe = 7
                    core.hbm3e_bw_gbps = 0.0
                    core.flops_util_pct = 0.0
                    core.temp_c = 79.4
                    core.collective_p99_ms = 2000.0
                    core.error_code = spec["hardware_code"]
                    events.append(
                        {
                            "hw_monotonic_ns": base_ns,
                            "node_id": core.node_id,
                            "core_id": core.core_id,
                            "global_rank": core.global_rank,
                            "efa_nic": core.efa_nic,
                            "subsystem": "HBM3E_ECC_CONTROLLER",
                            "severity": "FATAL",
                            "code": spec["hardware_code"],
                            "message": "Uncorrectable Double-Bit ECC trap at HBM3e Stack-2 Channel-5 row 0x1A8F; NeuronCore DMA halted.",
                        }
                    )
                else:
                    core.status = "RING_STALLED"
                    core.hbm3e_bw_gbps = 112.0
                    core.flops_util_pct = 4.2
                    core.collective_p99_ms = 1840.0
                    core.error_code = spec["symptom_code"]
                    events.append(
                        {
                            "hw_monotonic_ns": base_ns + 1_840_000_000 + core.global_rank * 12_000,
                            "node_id": core.node_id,
                            "core_id": core.core_id,
                            "global_rank": core.global_rank,
                            "efa_nic": core.efa_nic,
                            "subsystem": "NCCOM_COLLECTIVE_RING",
                            "severity": "ERROR",
                            "code": spec["symptom_code"],
                            "message": f"Collective nccomAllReduce ring barrier timed out after 1840ms waiting on upstream neighbor.",
                        }
                    )
            for link in self.efa_links:
                if link["src_node"] == "trn2-node-02" or link["dst_node"] == "trn2-node-02":
                    link["status"] = "BLOCKED_BY_PEER_HALT"
                    link["bandwidth_gbps"] = 0.0
                    link["p99_allreduce_ms"] = 1840.0

        elif fault_id == "fault_efa_straggler":
            for core in self.cores:
                if core.node_id == "trn2-node-04" and core.efa_nic == "efa-2":
                    is_primary = core.core_id == "nc-08"
                    core.status = "ROOT_FAULT" if is_primary else "RING_STALLED"
                    core.hbm3e_bw_gbps = 640.0
                    core.flops_util_pct = 24.5
                    core.collective_p99_ms = 840.0
                    core.error_code = spec["hardware_code"]
                    events.append(
                        {
                            "hw_monotonic_ns": base_ns + (0 if is_primary else 180_000),
                            "node_id": core.node_id,
                            "core_id": core.core_id,
                            "global_rank": core.global_rank,
                            "efa_nic": "efa-2",
                            "subsystem": "EFA_SRD_TRANSPORT",
                            "severity": "CRITICAL",
                            "code": spec["hardware_code"],
                            "message": "EFA NIC efa-2 SRD RNR retransmit rate 19.4% (optical Rx -14.8 dBm, FCS CRC errors spiking).",
                        }
                    )
                else:
                    core.status = "RING_STALLED"
                    core.hbm3e_bw_gbps = 910.0
                    core.flops_util_pct = 31.8
                    core.collective_p99_ms = 840.0
                    core.error_code = spec["symptom_code"]
                    events.append(
                        {
                            "hw_monotonic_ns": base_ns + 620_000_000 + core.global_rank * 9_000,
                            "node_id": core.node_id,
                            "core_id": core.core_id,
                            "global_rank": core.global_rank,
                            "efa_nic": core.efa_nic,
                            "subsystem": "NCCOM_COLLECTIVE_RING",
                            "severity": "WARN",
                            "code": spec["symptom_code"],
                            "message": "All-Reduce collective step-time degraded to 840.0ms p99 waiting on ring straggler.",
                        }
                    )
            for link in self.efa_links:
                if link["src_node"] == "trn2-node-04":
                    link["status"] = "DEGRADED_STRAGGLER"
                    link["bandwidth_gbps"] = 485.0
                    link["p99_allreduce_ms"] = 840.0
                    link["srd_retransmit_pct"] = 19.4
                    link["optical_rx_dbm"] = -14.8
                else:
                    link["p99_allreduce_ms"] = 840.0

        elif fault_id == "fault_kv_frag":
            for core in self.cores:
                is_lead = core.node_id == "trn2-node-01" and core.core_id == "nc-03"
                core.status = "ROOT_FAULT" if is_lead else "KV_PRESSURE"
                core.kv_frag_pct = 46.8 if is_lead else round(42.5 + (core.global_rank % 5) * 0.9, 1)
                core.hbm_used_gb = 46.4 if is_lead else 45.1
                core.hbm3e_bw_gbps = 1380.0
                core.flops_util_pct = 41.0
                core.collective_p99_ms = 11.4
                core.error_code = spec["hardware_code"] if is_lead else spec["symptom_code"]
                events.append(
                    {
                        "hw_monotonic_ns": base_ns + (0 if is_lead else 45_000_000 + core.global_rank * 5_000),
                        "node_id": core.node_id,
                        "core_id": core.core_id,
                        "global_rank": core.global_rank,
                        "efa_nic": core.efa_nic,
                        "subsystem": "NEURON_KV_ALLOCATOR" if is_lead else "VLLM_NEURON_SCHEDULER",
                        "severity": "ERROR" if is_lead else "WARN",
                        "code": core.error_code,
                        "message": (
                            "Contiguous KV slab allocation failed (46.8% external HBM fragmentation; 0 ECC/EFA faults)."
                            if is_lead
                            else "Continuous batching decode queue stalled on KV block compaction; TTFT p99 = 412ms."
                        ),
                    }
                )

        # Sort in reverse/noisy arrival order as seen by central log collector before correlation
        self.raw_events = sorted(events, key=lambda e: e["hw_monotonic_ns"])
        return self.snapshot()

    def apply_spare_swap(self, impaired_node: str) -> None:
        if impaired_node in self.nodes:
            idx = self.nodes.index(impaired_node)
            self.cordoned_nodes.append(impaired_node)
            self.nodes[idx] = self.warm_spare_node
        self.spare_state = f"ACTIVE_REPLACED_{impaired_node}"
        self.active_fault_id = None
        self.cores = self._build_nominal_cores(self.nodes)
        self.efa_links = self._build_nominal_efa_links(self.nodes)

    def apply_kv_compaction(self) -> None:
        self.active_fault_id = None
        self.cores = self._build_nominal_cores(self.nodes)
        for core in self.cores:
            core.kv_frag_pct = 6.2
        self.efa_links = self._build_nominal_efa_links(self.nodes)

    def summary_metrics(self) -> Dict[str, Any]:
        n = len(self.cores) or 1
        healthy = sum(1 for c in self.cores if c.status in ("HEALTHY", "SPARE_ACTIVE"))
        avg_bw = round(sum(c.hbm3e_bw_gbps for c in self.cores) / n, 1)
        avg_flops = round(sum(c.flops_util_pct for c in self.cores) / n, 1)
        max_p99 = round(max(c.collective_p99_ms for c in self.cores), 2)
        max_temp = round(max(c.temp_c for c in self.cores), 1)
        avg_frag = round(sum(c.kv_frag_pct for c in self.cores) / n, 1)
        total_dbe = sum(c.ecc_dbe for c in self.cores)
        return {
            "total_neuron_cores": n,
            "healthy_cores": healthy,
            "impaired_cores": n - healthy,
            "avg_hbm3e_bw_gbps": avg_bw,
            "avg_flops_util_pct": avg_flops,
            "collective_ring_p99_ms": max_p99,
            "peak_temp_c": max_temp,
            "avg_kv_frag_pct": avg_frag,
            "uncorrectable_ecc_dbe": total_dbe,
            "cluster_status": (
                "HEALTHY"
                if self.active_fault_id is None and not self.cordoned_nodes
                else ("RECOVERED_ON_WARM_SPARE" if self.active_fault_id is None else "IMPAIRED_FAULT_ACTIVE")
            ),
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "availability_zone": self.availability_zone,
            "active_nodes": list(self.nodes),
            "warm_spare_node": self.warm_spare_node,
            "spare_state": self.spare_state,
            "cordoned_nodes": list(self.cordoned_nodes),
            "active_fault_id": self.active_fault_id,
            "active_fault": copy.deepcopy(FAULT_CATALOG[self.active_fault_id]) if self.active_fault_id else None,
            "fault_catalog": list(FAULT_CATALOG.values()),
            "summary": self.summary_metrics(),
            "cores": [c.to_dict() for c in self.cores],
            "efa_links": copy.deepcopy(self.efa_links),
            "correlation": copy.deepcopy(self.last_correlation),
            "last_repair": copy.deepcopy(self.last_repair),
        }
