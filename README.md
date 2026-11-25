# NeuronPulse — ML Graph Compiler Pass, Tiled FlashAttention, Paged KV-Cache & Live Bedrock Serving Profiler

> **Live Amazon CloudFront App (Private S3 OAC + 256 MB Python 3.12 Lambda + Live Bedrock):** [https://d3dfklxckt5f00.cloudfront.net](https://d3dfklxckt5f00.cloudfront.net)  
> **CloudFront Distribution ID:** `E2ZSOS7XVJ55C7` (`us-west-2` Lambda `/api/*` + `s3://neuronpulse-web-796330847946-us-west-2`)  
> **Live Health Endpoint:** [`https://d3dfklxckt5f00.cloudfront.net/api/health`](https://d3dfklxckt5f00.cloudfront.net/api/health)  
> **Infrastructure Footprint:** **100% Free-Tier Serverless (`$0.00/hr` idle · `0` EC2/GPU/SageMaker instances)** — every metric below is measured from real Python 3.12 execution (`tracemalloc`, `time.perf_counter()`, IEEE-754 `max_abs_diff < 1e-9`) and live `boto3` AWS Bedrock Converse API calls (`us.amazon.nova-micro-v1:0` & `us.amazon.nova-lite-v1:0`).

---

## Overview

**NeuronPulse** implements four core ML compiler, attention kernel, serving memory-allocator, and distributed fault-triage systems from first principles in Python 3.12 stdlib, deployed behind an interactive Neo-Brutalist web workbench where every click executes live on a 256 MB AWS Lambda function:

1. **SSA Computation-Graph Compiler Pass (`neuronpulse/compiler_ir.py`)**
   - Typed Static Single Assignment (SSA) tensor IR supporting `Input`, `Constant`, `LayerNorm`, `MatMul`, `AddBias`, `GELU`, `Mul`, `ResidualAdd`, and epilogue-fused `FusedMatMulBiasGelu`.
   - Runs three sequential compiler passes: **`ConstantFold + CSE`**, backward-reachability **`DeadCodeElimination`**, and single-consumer **`MatMulBiasGeluFusion`** paired with SSA `last_use` buffer liveness freeing.
   - Executes both the unoptimized eager graph and the optimized fused graph on identical float64 matrices (`64×64 → 64×256 → 64×64`), cutting operator count from **`27 → 10` (`-63.0%`)**, reducing `tracemalloc` peak intermediate heap bytes from **`7,805,608 B → 951,648 B` (`-87.8%`)**, accelerating execution **`1.92×` (`59.844 ms → 31.109 ms`)**, and verifying **`max_abs_diff = 0.0` (`< 1e-9`)**.

2. **Online-Softmax Block-Tiled FlashAttention (`neuronpulse/flash_attention.py`)**
   - Implements Milakov-Gimelshein (2018) / Dao et al. (2022) online-softmax block-tiled attention (`block_size = 32` and `64` with optional causal upper-triangular tile skipping) alongside standard materialized `Softmax((Q @ K^T) / sqrt(d)) @ V`.
   - Never materializes the `N × N` attention matrix: at `N = 512, d = 32, B = 32`, `tracemalloc` peak working heap drops from **`17,451,268 B → 840,460 B` (`-95.18%`, `20.76×` compression)** with **`max_abs_diff = 1.21 × 10⁻¹⁷`**. Across `N = 128 → 512` (`4×`), naive memory grows **`14.52×`** (`O(N²)`) while tiled FlashAttention grows only **`3.38×`** (`O(N)`). With `causal=True` (`B = 64`), skipping `28` masked tiles also reduces wall-clock latency from **`115.816 ms → 83.621 ms` (`-27.8%`)**.

3. **Paged KV-Cache Block Allocator + Copy-on-Write Prefix Sharing (`neuronpulse/paged_kv.py`)**
   - Evaluates `64` concurrent multi-turn LLM requests (`seq_len` `48..1024`, sharing a `256`-token system prompt) comparing contiguous `max_seq_len = 1024` reservation against demand-paged `block_size = 16` block tables with chained SHA-256 prefix deduplication.
   - Cuts total allocated KV memory from **`3.22 GB → 1.01 GB` (`-68.55%`)**, drops internal fragmentation from **`46.18% → 2.37%`**, achieves a **`43.5%` prefix-block cache hit rate** (`max_block_ref_count = 64`), admits **`64` vs. `26` concurrent requests (`2.46×`)** under a `1,280 MB` memory cap, and verifies Copy-on-Write (`mutate_token_cow`) isolation with **`0` peer corruptions**.

4. **Distributed Worker Causal Fault Correlator & Live Serverless Bedrock Profiler (`neuronpulse/correlator.py`)**
   - `CrossNodeErrorCorrelator` ranks shuffled out-of-order 64-worker log streams by subsystem causal priority + monotonic nanosecond timestamps, achieving **`100.0%` (`48/48`)** root-worker isolation vs. **`2.1%` (`1/48`)** arrival-order heuristics (`64:1` alert compression across `3,072` raw alarms in **`3.743 ms`**).
   - `BedrockModelProfiler` invokes live `boto3.client("bedrock-runtime", region_name="us-west-2").converse(...)` calls on **`us.amazon.nova-micro-v1:0`** and **`us.amazon.nova-lite-v1:0`** (`$0.000129` total benchmark run cost), recording real server `p50`/`p95` latency, `tokens/sec`, and `USD / 1k requests`.

---

## Serverless Architecture

![NeuronPulse Architecture](docs/architecture.svg)

---

## Executed Benchmark Summary (`benchmarks/latest_results.json`)

Measured on **CPython `3.12.14`** + live **AWS Bedrock (`us-west-2`)**:

| Module / Benchmark | NeuronPulse Measured Result | Measured Baseline | Improvement / Delta | Verification |
| :--- | :---: | :---: | :---: | :---: |
| **1. SSA Compiler Pass (`stacked_2layer_mlp`)** | `10` ops · `951.6 KB` · `31.109 ms` | `27` ops · `7.81 MB` · `59.844 ms` | **`-87.8%` peak bytes · `1.92×` faster** | `max_abs_diff = 0.0` |
| **2A. Tiled FlashAttention (`N=512, d=32, B=32`)** | `840.5 KB` (`32×32` tile) | `17.45 MB` (`512×512` materialized) | **`-95.18%` memory (`20.76×`)** | `max_abs_diff = 1.21e-17` |
| **2B. Causal Tiled FlashAttention (`N=512, B=64`)** | `887.6 KB` · `83.621 ms` (`28` tiles skipped) | `14.31 MB` · `115.816 ms` | **`-93.80%` mem · `-27.8%` ms** | `max_abs_diff = 2.78e-17` |
| **3. Paged KV-Cache + CoW (`64 reqs, B=16`)** | `1.01 GB` · **`2.37%` frag** · `64` reqs @ 1280 MB | `3.22 GB` · `46.18%` frag · `26` reqs @ 1280 MB | **`-68.55%` bytes · `2.46×` concurrency** | `0` CoW peer corruptions |
| **4A. Causal Worker Correlator (`48 × 64` logs)** | **`100.0%` (`48/48`)** · `64:1` compression | `2.1%` (`1/48` arrival order) | **`+97.9 pts`** in `3.743 ms` | `3,072 → 48` root incidents |
| **4B. Live Bedrock (`Nova Micro` vs `Nova Lite`)** | Micro: `p50 356.0ms` (`p95 455.4ms`) · `125.33 tok/s` · `$0.00896/1k` | Lite: `p50 668.0ms` (`p95 904.6ms`) · `72.89 tok/s` · `$0.01676/1k` | **`1.88×` faster p50 · `-46.5%` cost** | `$0.000129` live run cost |

---

## API Routes (`neuronpulse/server.py`)

- `GET /api/health` — Live runtime health, Python version, region, and zero-instance serverless verification.
- `GET /api/state` — Saved benchmark ledger (`benchmarks/latest_results.json`) + live execution snapshot.
- `POST /api/compiler/optimize` — Compiles & executes unoptimized vs. fused SSA graphs live (`preset`, `batch_tokens`, `hidden_dim`).
- `POST /api/attention/run` — Executes materialized vs. online-softmax tiled FlashAttention live (`seq_len`, `head_dim`, `block_size`, `causal`, `include_sweep`).
- `POST /api/kvcache/run` — Executes contiguous vs. Paged KV + SHA-256 CoW allocator live (`num_requests`, `block_size`, `system_prompt_tokens`, `memory_cap_mb`).
- `POST /api/correlator/run` — Runs out-of-order 64-worker fault correlation + 48-trial shuffled evaluation (`scenario_id`, `root_rank`, `num_workers`).
- `POST /api/bedrock/compare` — Invokes live `us.amazon.nova-micro-v1:0` and `us.amazon.nova-lite-v1:0` via `boto3` Converse in `us-west-2`.

---

## Quickstart & Deployment

```bash
# 1. Run unit & Lambda API integration tests
PYTHONPATH=. python3 -m unittest discover -s tests -v

# 2. Reproduce all Python 3.12 + live AWS Bedrock benchmarks (~$0.00013 total Bedrock cost)
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
AWS_PROFILE=portfolio AWS_DEFAULT_REGION=us-west-2 PYTHONPATH=. python3 -m benchmarks.run_benchmarks

# 3. Start local interactive studio at http://127.0.0.1:3096
PYTHONPATH=. python3 -m neuronpulse.server

# 4. Deploy 256 MB Lambda + Private S3 (OAC) + CloudFront invalidation
chmod +x infra/deploy_cloudfront.sh
./infra/deploy_cloudfront.sh
```
