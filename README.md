# NeuronPulse v2: ML Graph Compiler, Tiled FlashAttention, Paged KV, Quantization, Speculative Decoding, Transport Ring & Live Bedrock Profiler

> **Live Amazon CloudFront App (Private S3 OAC + 256 MB Python 3.12 Lambda + Live Bedrock):** [https://d3dfklxckt5f00.cloudfront.net](https://d3dfklxckt5f00.cloudfront.net)  
> **CloudFront Distribution ID:** `E2ZSOS7XVJ55C7` (`us-west-2` Lambda `/api/*` + `s3://neuronpulse-web-796330847946-us-west-2`)  
> **Live Health & Quantization Endpoints:** [`/api/health`](https://d3dfklxckt5f00.cloudfront.net/api/health) · [`/api/quant-spec`](https://d3dfklxckt5f00.cloudfront.net/api/quant-spec) · [`/api/transport/ring`](https://d3dfklxckt5f00.cloudfront.net/api/transport/ring)  
> **Infrastructure Footprint:** **100% Free-Tier Serverless (`$0.00/hr` idle · `0` EC2, GPU, or SageMaker instances)**. Every metric below is measured from real Python execution (`tracemalloc`, `time.perf_counter()`, `struct`/`bytearray` binary packing, IEEE-754 parity checks) and live `boto3` AWS Bedrock Converse API calls (`us.amazon.nova-micro-v1:0` & `us.amazon.nova-lite-v1:0`).

---

## Overview

**NeuronPulse v2** implements core LLM compiler passes, attention kernels, serving memory allocators, post-training quantization, speculative decoding, static sequence bucketing, user-space descriptor rings, and distributed fault triage in Python, deployed behind an interactive Neo-Brutalist web workbench where every click executes live on a 256 MB AWS Lambda function:

1. **SSA Computation-Graph Compiler Pass (`neuronpulse/compiler_ir.py`)**
   - Typed Static Single Assignment (SSA) tensor IR supporting `Input`, `Constant`, `LayerNorm`, `MatMul`, `AddBias`, `GELU`, `Mul`, `ResidualAdd`, and epilogue-fused `FusedMatMulBiasGelu`.
   - Runs three sequential compiler passes: **`ConstantFold + CSE`**, backward-reachability **`DeadCodeElimination`**, and single-consumer **`MatMulBiasGeluFusion`** paired with SSA `last_use` buffer liveness freeing.
   - Executes both the unoptimized eager graph and the optimized fused graph on identical float64 matrices (`64x64 -> 64x256 -> 64x64`), cutting operator count from **`27 -> 10` (`-63.0%`)**, reducing `tracemalloc` peak intermediate heap bytes from **`7,806,120 B -> 954,136 B` (`-87.8%`)**, accelerating execution **`1.75x` (`38.78 ms -> 22.185 ms`)**, and verifying **`max_abs_diff = 0.0` (`< 1e-9`)**.

2. **Online-Softmax Block-Tiled FlashAttention (`neuronpulse/flash_attention.py`)**
   - Implements Milakov-Gimelshein (2018) / Dao et al. (2022) online-softmax block-tiled attention (`block_size = 32` and `64` with optional causal upper-triangular tile skipping) alongside standard materialized `Softmax((Q @ K^T) / sqrt(d)) @ V`.
   - Never materializes the `N x N` attention matrix: at `N = 512, d = 32, B = 32`, `tracemalloc` peak working heap drops from **`17,452,024 B -> 841,504 B` (`-95.18%`, `20.74x` compression)** with **`max_abs_diff = 1.21e-17`**. Across `N = 128 -> 512` (`4x`), naive memory grows **`14.51x`** (`O(N^2)`) while tiled FlashAttention grows only **`3.37x`** (`O(N)`). With `causal=True` (`B = 64`), skipping `28` masked tiles reduces wall-clock latency from **`80.177 ms -> 60.827 ms` (`-24.1%`)**.

3. **Paged KV-Cache Block Allocator + Copy-on-Write Prefix Sharing (`neuronpulse/paged_kv.py`)**
   - Evaluates `64` concurrent multi-turn LLM requests (`seq_len` `48..1024`, sharing a `256`-token system prompt) comparing contiguous `max_seq_len = 1024` reservation against demand-paged `block_size = 16` block tables with chained SHA-256 prefix deduplication.
   - Cuts total allocated KV memory from **`3.22 GB -> 1.01 GB` (`-68.55%`)**, drops internal fragmentation from **`46.18% -> 2.37%`**, achieves a **`43.5%` prefix-block cache hit rate** (`max_block_ref_count = 64`), admits **`64` vs. `26` concurrent requests (`2.46x`)** under a `1,280 MB` memory cap, and verifies Copy-on-Write (`mutate_token_cow`) isolation with **`0` peer corruptions**.

4. **Quantization (`W8A8` / `W4A16`), Top-2 `SwiGLU` MoE + `TP=4` Sharding, Lossless Speculative Decoding, DP AOT Buckets & Lock-Free Transport Ring (`neuronpulse/quant_spec_transport.py`)**
   - **SmoothQuant `W8A8` & 32-Group `W4A16` + Top-2 `SwiGLU` MoE + `TP=4` Sharding:** Executes per-channel activation outlier migration (`alpha = 0.5`) `int8` `W8A8` (`131,072 B -> 32,768 B`, **`4.0x` vs FP32**, `cosine_similarity = 0.999956`, **`+15.46 dB` SQNR** and **`4.77x` lower max logit error** vs naive `W8A8`) and packed 2-nibble-per-byte 32-group `int4` `W4A16` (`65,536 B -> 18,432 B` including FP16 scales, **`3.56x` vs FP16** / `4.5` bits/weight, `cosine_similarity = 0.995348`). Pairs online bias-balanced Top-2 `SwiGLU` MoE routing (`8` experts, `25.0%` active FFN parameters/token, cutting capacity token drops from **`68.36% -> 0.0%`** and load CV from **`1.704 -> 0.095`**) with 4-rank `ColumnParallel + RowParallel + Ring AllReduce` verification (**`max_abs_diff = 2.78e-17`**).
   - **Lossless Rejection-Sampled Speculative Decoding + Exact DP Static AOT Sequence Bucketing & BFD Packing:** Implements Leviathan/Chen rejection sampling (`min(1, p_target/q_draft)` + `(p_target - q_draft)+` residual resampling + tentative KV-slot rollback), emitting **`3.05x` tokens per target forward step** (`256` tokens in `84` target steps, **`-67.2%` target forward steps**) while preserving the exact target marginal distribution (**`max_abs_diff = 1.11e-16`**). Solves ahead-of-time static sequence-bucket selection (`K = 5` buckets -> `[256, 512, 768, 1536, 2048]`, cutting padding waste from **`73.99% -> 23.46%`**, `-89.23%` wasted pad tokens vs single `2048` bucket, verified global optimum) and Best-Fit-Decreasing (`BFD`) sequence packing (**`0.61%` padding waste**, **`99.39%` utilization**, **`3.82x` fewer `2048`-token bins**, `0.0` cross-document attention leakage).
   - **Libfabric-Style Binary `SQ/CQ` Descriptor Ring & Augmented AVL Interval-Tree `MR` Pinning Cache:** Pre-allocates cache-line-sized `64 B` WQE and `32 B` CQE binary ring buffers (`memoryview` in-place state transitions `FREE -> POSTED -> IN_FLIGHT -> COMPLETED -> FREE` + 64-bit multipath SRD out-of-order sliding-window bitmap reordering `1,408` packets across `4,096` messages), cutting dynamic heap allocation **`-95.86%`** (`25,824 B -> 1,068 B`). Pairs with a self-balancing Augmented AVL Interval-Tree virtual-address Memory Registration (`MR`) cache (`213` registered slabs, AVL height `8 <= 12`, `5.97` avg comparisons/lookup, **`91.62%` hit rate**), delivering a **`6.16x` lookup speedup** (`2.144 ms` vs `13.198 ms` cold `4 KiB` page-pinning walk) and verified `munmap` range invalidation safety.

5. **Distributed Worker Causal Fault Correlator & Live Serverless Bedrock Profiler (`neuronpulse/correlator.py`)**
   - `CrossNodeErrorCorrelator` ranks shuffled out-of-order 64-worker log streams by subsystem causal priority + monotonic nanosecond timestamps, achieving **`100.0%` (`48/48`)** root-worker isolation vs. **`2.1%` (`1/48`)** arrival-order heuristics (`64:1` alert compression across `3,072` raw alarms in **`2.966 ms`**).
   - `BedrockModelProfiler` invokes live `boto3.client("bedrock-runtime", region_name="us-west-2").converse(...)` calls on **`us.amazon.nova-micro-v1:0`** and **`us.amazon.nova-lite-v1:0`** (`$0.000132` total benchmark run cost), recording real server `p50`/`p95` latency, `tokens/sec`, and `USD / 1k requests`.

---

## Serverless Architecture

![NeuronPulse Architecture](docs/architecture.svg)

---

## Executed Benchmark Summary (`benchmarks/latest_results.json`)

Measured on **CPython `3.14.6`** + live **AWS Bedrock (`us-west-2`)**:

| Module / Benchmark | NeuronPulse Measured Result | Measured Baseline | Improvement / Delta | Verification |
| :--- | :---: | :---: | :---: | :---: |
| **1. SSA Compiler Pass (`stacked_2layer_mlp`)** | `10` ops · `954.1 KB` · `22.185 ms` | `27` ops · `7.81 MB` · `38.78 ms` | **`-87.8%` peak bytes · `1.75x` faster** | `max_abs_diff = 0.0` |
| **2A. Tiled FlashAttention (`N=512, d=32, B=32`)** | `841.5 KB` (`32x32` tile) | `17.45 MB` (`512x512` materialized) | **`-95.18%` memory (`20.74x`)** | `max_abs_diff = 1.21e-17` |
| **2B. Causal Tiled FlashAttention (`N=512, B=64`)** | `888.6 KB` · `60.827 ms` (`28` tiles skipped) | `14.31 MB` · `80.177 ms` | **`-93.79%` mem · `-24.1%` ms** | `max_abs_diff = 2.78e-17` |
| **3. Paged KV-Cache + CoW (`64 reqs, B=16`)** | `1.01 GB` · **`2.37%` frag** · `64` reqs @ 1280 MB | `3.22 GB` · `46.18%` frag · `26` reqs @ 1280 MB | **`-68.55%` bytes · `2.46x` concurrency** | `0` CoW peer corruptions |
| **4A. SmoothQuant `W8A8` & `32g W4A16`** | `W8A8`: `32.8 KB` (`cos 0.999956`) · `W4A16`: `18.4 KB` (`cos 0.995348`) | `FP32`: `131.1 KB` · `FP16`: `65.5 KB` · Naive `W8A8` (`25.13 dB`) | **`4.0x` vs FP32 · `3.56x` vs FP16 · `+15.46 dB`** | `4.5 bits/weight` (`W4A16`) |
| **4B. Top-2 `SwiGLU` MoE + `TP=4` Sharding** | **`0.0%` drops** · `CV 0.095` · `25.0%` active FFN | Skewed router: `68.36%` drops · `CV 1.704` | **`-68.36 pts` drop rate · `4.0x` rank shard** | `TP=4 diff = 2.78e-17` |
| **4C. Speculative Decoding (`gamma = 4`)** | `84` target steps for `256` tokens (**`3.05x` tok/step**) | `256` autoregressive target steps (`1.0x`) | **`-67.2%` target forward passes** | `marginal diff = 1.11e-16` |
| **4D. Exact DP AOT Buckets + BFD Packing** | DP `K=5`: **`23.46%` pad** · BFD Pack: **`0.61%` pad** (`67` bins) | Static `[2048]`: `73.99%` pad (`256` bins) | **`-89.23%` AOT pad · `3.82x` fewer bins** | Global DP optimum · `0.0` leak |
| **4E. Binary `SQ/CQ` Ring & AVL `MR` Cache** | `1,068 B` dyn heap · `MR` `2.144 ms` (`91.62%` hit, `h=8`) | `25,824 B` dict queue · `13.198 ms` cold `4 KiB` pin | **`-95.86%` ring heap · `6.16x` MR speedup** | `16,384` transitions · `munmap` safe |
| **5A. Causal Worker Correlator (`48 x 64` logs)** | **`100.0%` (`48/48`)** · `64:1` compression | `2.1%` (`1/48` arrival order) | **`+97.9 pts`** in `2.966 ms` | `3,072 -> 48` root incidents |
| **5B. Live Bedrock (`Nova Micro` vs `Nova Lite`)** | Micro: `p50 474.0ms` (`p95 577.2ms`) · `98.82 tok/s` · `$0.00892/1k` | Lite: `p50 651.0ms` (`p95 798.8ms`) · `82.48 tok/s` · `$0.0173/1k` | **`1.37x` faster p50 · `-48.4%` cost** | `$0.000132` live run cost |

---

## API Routes (`neuronpulse/server.py`)

- `GET /api/health` : Live runtime health, active module manifest, Python version, region, and zero-instance serverless verification.
- `GET /api/state` : Saved benchmark ledger (`benchmarks/latest_results.json`) + live execution snapshot across all modules.
- `POST /api/compiler/optimize` : Compiles & executes unoptimized vs. fused SSA graphs live (`preset`, `batch_tokens`, `hidden_dim`).
- `POST /api/attention/run` : Executes materialized vs. online-softmax tiled FlashAttention live (`seq_len`, `head_dim`, `block_size`, `causal`, `include_sweep`).
- `POST /api/kvcache/run` : Executes contiguous vs. Paged KV + SHA-256 CoW allocator live (`num_requests`, `block_size`, `system_prompt_tokens`, `memory_cap_mb`).
- `GET/POST /api/quant-spec` : Executes SmoothQuant `W8A8` + 32-group `W4A16` + Top-k `SwiGLU` MoE + `TP=4` sharding, lossless speculative decoding, and Exact DP AOT sequence bucketing + BFD packing live (`alpha`, `group_size`, `gamma`, `draft_alignment`, `max_buckets`, `top_k`).
- `GET/POST /api/transport/ring` : Executes pre-allocated `64 B` WQE `SQ/CQ` binary descriptor ring + SRD OOO bitmap and Augmented AVL Interval-Tree `MR` cache live (`ring_capacity`, `num_messages`, `batch_size`, `num_mr_regions`, `num_mr_lookups`, `tensor_bytes`).
- `POST /api/correlator/run` : Runs out-of-order 64-worker fault correlation + 48-trial shuffled evaluation (`scenario_id`, `root_rank`, `num_workers`).
- `POST /api/bedrock/compare` : Invokes live `us.amazon.nova-micro-v1:0` and `us.amazon.nova-lite-v1:0` via `boto3` Converse in `us-west-2`.

---

## Quickstart & Deployment

```bash
# 1. Run all 10 unit & Lambda API integration tests
PYTHONPATH=. python3 -m unittest discover -s tests -v

# 2. Reproduce all Python + live AWS Bedrock benchmarks (~$0.00013 total Bedrock cost)
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
AWS_PROFILE=portfolio AWS_DEFAULT_REGION=us-west-2 PYTHONPATH=. python3 -m benchmarks.run_benchmarks

# 3. Start local interactive studio at http://127.0.0.1:3096
PYTHONPATH=. python3 -m neuronpulse.server

# 4. Deploy 256 MB Lambda + Private S3 (OAC) + CloudFront invalidation
chmod +x infra/deploy_cloudfront.sh
./infra/deploy_cloudfront.sh
```
