"""Amazon Bedrock (`us.amazon.nova-lite-v1:0`) Trainium2 Cluster & Neuron Compiler Copilot."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from typing import Any, Dict, Optional


def _invoke_bedrock_nova(system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
    """Invoke Amazon Bedrock `us.amazon.nova-lite-v1:0` via boto3 (Lambda) or AWS CLI (`portfolio` local)."""
    if os.environ.get("NEURONPULSE_OFFLINE_LLM") == "1":
        return None

    model_id = os.environ.get("NEURONPULSE_BEDROCK_MODEL", "us.amazon.nova-lite-v1:0")
    region = os.environ.get("AWS_REGION", "us-west-2")
    t0 = time.perf_counter()

    # Path A: boto3 in AWS Lambda Python 3.12 runtime (or local venv with boto3)
    try:
        import boto3  # type: ignore

        client = boto3.client("bedrock-runtime", region_name=region)
        resp = client.converse(
            modelId=model_id,
            system=[{"text": system_prompt}],
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            inferenceConfig={"maxTokens": 450, "temperature": 0.1},
        )
        text = resp["output"]["message"]["content"][0]["text"]
        return {
            "text": text.strip(),
            "engine": f"aws-bedrock ({model_id})",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "tokens": resp.get("usage", {}).get("totalTokens", 0),
        }
    except ImportError:
        pass
    except Exception:
        pass

    # Path B: Local AWS CLI fallback
    aws_bin = shutil.which("aws")
    if not aws_bin:
        return None
    profile = os.environ.get("AWS_PROFILE", "portfolio")
    env = dict(os.environ)
    env.pop("AWS_ACCESS_KEY_ID", None)
    env.pop("AWS_SECRET_ACCESS_KEY", None)
    env.pop("AWS_SESSION_TOKEN", None)
    env["AWS_PROFILE"] = profile

    cmd = [
        aws_bin,
        "bedrock-runtime",
        "converse",
        "--region",
        region,
        "--model-id",
        model_id,
        "--system",
        json.dumps([{"text": system_prompt}]),
        "--messages",
        json.dumps([{"role": "user", "content": [{"text": user_prompt}]}]),
        "--inference-config",
        json.dumps({"maxTokens": 450, "temperature": 0.1}),
        "--output",
        "json",
    ]
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=14, check=False)
        if proc.returncode == 0 and proc.stdout.strip():
            data = json.loads(proc.stdout)
            text = data["output"]["message"]["content"][0]["text"]
            return {
                "text": text.strip(),
                "engine": f"aws-bedrock ({model_id})",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                "tokens": data.get("usage", {}).get("totalTokens", 0),
            }
    except Exception:
        pass
    return None


def _deterministic_fallback(
    question: str,
    cluster_snapshot: Dict[str, Any],
    profiler_result: Dict[str, Any],
) -> str:
    active_fault = cluster_snapshot.get("active_fault")
    sel = profiler_result.get("selected", profiler_result)
    if active_fault:
        fid = active_fault["fault_id"]
        if fid == "fault_hbm_ecc":
            return (
                "### Root-Cause Diagnosis: HBM3e Uncorrectable Double-Bit ECC (`0x4E02`)\n"
                "- **Causal Isolation:** `trn2-node-02:nc-09` (Rank 25) tripped a physical HBM3e Stack-2 Bank-5 "
                "double-bit ECC fault `1,840 ms` before the 63 healthy ranks stalled in `nccomAllReduce`.\n"
                "- **Immediate CLI Triage:**\n"
                "  ```bash\n"
                "  neuron-ls -j | jq '.[] | select(.neuron_device==4)'\n"
                "  neuron-monitor -c /etc/neuron/ecc_watch.conf --one-shot\n"
                "  ultraserver-ctl cordon trn2-node-02 --promote-spare trn2-spare-01\n"
                "  ```\n"
                "- **Checkpoint Recovery:** Resume from `NEURON_COMPILE_CACHE_URL` + S3 Express shard (`~74s` total)."
            )
        if fid == "fault_efa_straggler":
            return (
                "### Root-Cause Diagnosis: EFA SRD All-Reduce Straggler (`trn2-node-04:efa-2`)\n"
                "- **Network Isolation:** Optical Rx power on `trn2-node-04:efa-2` dropped to `-14.8 dBm` (`19.4%` SRD RNR "
                "retransmits), inflating 64-rank All-Reduce `p99` from `4.2ms` to `840.0ms`.\n"
                "- **Immediate CLI Triage:**\n"
                "  ```bash\n"
                "  fi_info -p efa -t FI_EP_RDM && ethtool -S efa2 | grep -E 'rx_pkts|retrans'\n"
                "  FI_EFA_USE_DEVICE_RDMA=1 NCCL_PROTO=Simple ultraserver-ctl drain trn2-node-04 --swap trn2-spare-01\n"
                "  ```"
            )
        return (
            "### Root-Cause Diagnosis: Continuous Batching KV-Cache Fragmentation (`44.6%`)\n"
            "- **Hardware Check:** `0` HBM ECC errors and `0` EFA link drops — **do NOT cordon or reboot nodes**.\n"
            "- **Compiler & Serving Remediation:** Enable NKI PagedAttention block compaction (`block_size=16`):\n"
            "  ```bash\n"
            "  export NEURON_CC_FLAGS=\"--model-type=transformer --neuron-nki-flash-attn=enable --lnc=2\"\n"
            "  vllm-neuron-ctl compact-kv-cache --block-size 16 --watermark 0.08\n"
            "  ```"
        )

    return (
        f"### Trainium2 UltraServer & `torch_neuronx` Optimization Guidance\n"
        f"- **Cluster Health:** `{cluster_snapshot['summary']['healthy_cores']}/64` NeuronCores healthy "
        f"(`{cluster_snapshot['summary']['collective_ring_p99_ms']} ms` All-Reduce p99).\n"
        f"- **Active Serving Profile (`{sel['precision']}`, `TP={sel['tp_degree']}`, `Batch={sel['batch_size']}`):** "
        f"TTFT `{sel['ttft_ms']} ms`, UltraServer throughput `{sel['ultraserver_throughput_tok_s']} tok/s`, "
        f"HBM3e `{sel['hbm_used_per_core_gb']}/{sel['hbm_capacity_per_core_gb']} GB` per core.\n"
        f"- **Recommended `NEURON_CC_FLAGS`:**\n"
        f"  ```bash\n"
        f"  export NEURON_CC_FLAGS=\"{sel['neuron_cc_flags']}\"\n"
        f"  neuron-ls --topology && torchrun --nproc_per_node={sel['tp_degree']} serve_llama70b_nki.py\n"
        f"  ```"
    )


def ask_cluster_copilot(
    question: str,
    cluster_snapshot: Dict[str, Any],
    profiler_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Answer operator questions using live Amazon Bedrock Nova Lite grounded in telemetry & NKI profile."""
    summary = cluster_snapshot.get("summary", {})
    active_fault = cluster_snapshot.get("active_fault")
    corr = cluster_snapshot.get("correlation")
    sel = profiler_result.get("selected", profiler_result)

    system_prompt = (
        "You are NeuronPulse Copilot, an expert AWS Annapurna Labs Trainium2 UltraServer, EFAv3 SRD networking, "
        "and Neuron SDK (`neuron-ls`, `torch_neuronx`, `neuronxcc.nki`) systems engineer. "
        "Be concise, concrete, and technical (under 190 words). Cite exact node/core IDs, error codes, "
        "TTFT/throughput numbers from the live telemetry context, and include 2-3 copy-ready CLI commands or "
        "`NEURON_CC_FLAGS` flags."
    )

    telemetry_context = {
        "cluster_id": cluster_snapshot.get("cluster_id"),
        "cluster_summary": summary,
        "active_fault": active_fault,
        "cross_node_correlation": corr,
        "last_repair": cluster_snapshot.get("last_repair"),
        "nki_serving_profile": {
            "model": sel.get("model"),
            "tp_degree": sel.get("tp_degree"),
            "precision": sel.get("precision"),
            "batch_size": sel.get("batch_size"),
            "nki_flash_fusion": sel.get("nki_flash_fusion"),
            "ttft_ms": sel.get("ttft_ms"),
            "ultraserver_throughput_tok_s": sel.get("ultraserver_throughput_tok_s"),
            "hbm_used_per_core_gb": sel.get("hbm_used_per_core_gb"),
            "fits_in_hbm": sel.get("fits_in_hbm"),
            "neuron_cc_flags": sel.get("neuron_cc_flags"),
        },
    }
    user_prompt = (
        f"Live UltraServer Telemetry & NKI Profiler Context:\n{json.dumps(telemetry_context, indent=2)}\n\n"
        f"Operator Question: {question}"
    )

    live = _invoke_bedrock_nova(system_prompt=system_prompt, user_prompt=user_prompt)
    if live:
        return {
            "question": question,
            "answer": live["text"],
            "engine": live["engine"],
            "latency_ms": live["latency_ms"],
            "tokens": live["tokens"],
            "grounded_on_fault": active_fault["fault_id"] if active_fault else "HEALTHY_CLUSTER",
        }

    return {
        "question": question,
        "answer": _deterministic_fallback(question, cluster_snapshot, profiler_result),
        "engine": "deterministic-neuron-runbook-fallback",
        "latency_ms": 1.2,
        "tokens": 0,
        "grounded_on_fault": active_fault["fault_id"] if active_fault else "HEALTHY_CLUSTER",
    }
