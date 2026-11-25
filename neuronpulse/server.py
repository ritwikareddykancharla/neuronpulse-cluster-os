"""HTTP Server & AWS CloudFront `/api/*` Lambda Origin Handler for NeuronPulse.

Every `/api/*` endpoint executes the real Python 3.12 ML systems algorithms live
(`compiler_ir`, `flash_attention`, `paged_kv`, `correlator`, and live `boto3`
Amazon Bedrock `us.amazon.nova-micro-v1:0` & `us.amazon.nova-lite-v1:0`) and
returns real measured `wall_clock_ms`, `tracemalloc` `peak_bytes`, and `max_abs_diff`.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import platform
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from .compiler_ir import run_compiler_pass_benchmark
from .correlator import (
    DEFAULT_BENCHMARK_PROMPTS,
    DISTRIBUTED_FAULT_SCENARIOS,
    BedrockModelProfiler,
    CrossNodeErrorCorrelator,
    build_worker_fault_stream,
    run_correlator_benchmark,
)
from .flash_attention import benchmark_attention_pair, run_attention_scaling_sweep
from .paged_kv import run_paged_kv_benchmark
from .quant_spec_transport import run_quant_spec_suite, run_transport_ring_mr_benchmark

ROOT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT_DIR / "web"
DOCS_DIR = ROOT_DIR / "docs"
BENCHMARK_JSON_PATH = ROOT_DIR / "benchmarks" / "latest_results.json"

# Optional hook used strictly by offline unit tests so `unittest` stays fast & offline
_TEST_BEDROCK_CONVERSE_FN: Optional[Callable[..., Dict[str, Any]]] = None

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
}


def _load_saved_benchmarks() -> Optional[Dict[str, Any]]:
    if BENCHMARK_JSON_PATH.is_file():
        try:
            return json.loads(BENCHMARK_JSON_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _merged_params(query: Dict[str, str], body: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(query)
    merged.update(body)
    return merged


def _to_bool(val: Any, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def route_api_request(
    method: str,
    path: str,
    query: Dict[str, str],
    body: Dict[str, Any],
    request_id: str = "local",
) -> Tuple[int, Dict[str, Any]]:
    """Pure router shared by local HTTP server and AWS Lambda `/api/*` origin behind CloudFront."""
    t_req_start = time.perf_counter()
    params = _merged_params(query, body)

    try:
        if method == "GET" and path == "/api/health":
            return 200, {
                "status": "ok",
                "service": "neuronpulse",
                "version": "2.1.0",
                "architecture": "100%-serverless-s3-cloudfront-lambda-bedrock",
                "ec2_or_gpu_instances": 0,
                "idle_cost_usd": 0.0,
                "delivery": os.environ.get("NEURONPULSE_DELIVERY", "cloudfront-s3-oac + python3.12-lambda"),
                "aws_region": os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-west-2")),
                "modules": [
                    "compiler_ir",
                    "flash_attention",
                    "paged_kv",
                    "quant_spec_moe_aot",
                    "transport_sq_cq_mr_cache",
                    "correlator_bedrock",
                ],
                "bedrock_models": [
                    "us.amazon.nova-micro-v1:0",
                    "us.amazon.nova-lite-v1:0",
                ],
                "python_version": platform.python_version(),
                "request_id": request_id,
                "handler_wall_ms": round((time.perf_counter() - t_req_start) * 1000.0, 3),
            }

        if method == "GET" and path == "/api/state":
            saved = _load_saved_benchmarks()
            live_compiler = run_compiler_pass_benchmark(
                preset="stacked_2layer_mlp",
                batch_tokens=48,
                hidden_dim=48,
                repeats=1,
            )
            live_attn = benchmark_attention_pair(seq_len=256, head_dim=32, block_size=32, causal=False)
            live_kv = run_paged_kv_benchmark(
                num_requests=64,
                block_size=16,
                max_seq_len=1024,
                system_prompt_tokens=256,
                memory_cap_mb=1280,
            )
            live_quant_spec = run_quant_spec_suite()
            live_transport = run_transport_ring_mr_benchmark(num_messages=2048, num_mr_lookups=512)
            live_corr = run_correlator_benchmark(trials=48, num_workers=64, seed=42)
            return 200, {
                "request_id": request_id,
                "python_version": platform.python_version(),
                "aws_region": os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-west-2")),
                "handler_wall_ms": round((time.perf_counter() - t_req_start) * 1000.0, 3),
                "saved_benchmarks": saved,
                "live_snapshot": {
                    "compiler_ir": live_compiler,
                    "flash_attention": live_attn,
                    "paged_kv": live_kv,
                    "quant_spec": live_quant_spec,
                    "transport_ring": live_transport,
                    "correlator": live_corr,
                },
            }

        if method in ("GET", "POST") and path == "/api/compiler/optimize":
            preset = str(params.get("preset") or "stacked_2layer_mlp").strip()
            batch_tokens = int(params.get("batch_tokens", 64))
            hidden_dim = int(params.get("hidden_dim", 64))
            repeats = max(1, min(5, int(params.get("repeats", 2))))
            res = run_compiler_pass_benchmark(
                preset=preset,
                batch_tokens=batch_tokens,
                hidden_dim=hidden_dim,
                repeats=repeats,
            )
            res["request_id"] = request_id
            res["python_version"] = platform.python_version()
            res["handler_wall_ms"] = round((time.perf_counter() - t_req_start) * 1000.0, 3)
            return 200, res

        if method in ("GET", "POST") and path == "/api/attention/run":
            seq_len = int(params.get("seq_len", 256))
            head_dim = int(params.get("head_dim", 32))
            block_size = int(params.get("block_size", 32))
            causal = _to_bool(params.get("causal"), default=False)
            include_sweep = _to_bool(params.get("include_sweep"), default=False)

            res = benchmark_attention_pair(
                seq_len=seq_len,
                head_dim=head_dim,
                block_size=block_size,
                causal=causal,
            )
            if include_sweep:
                res["scaling_sweep"] = run_attention_scaling_sweep(
                    seq_lengths=(128, 256, 512),
                    head_dim=head_dim,
                    block_size=min(block_size, 64),
                    causal=causal,
                )
            res["request_id"] = request_id
            res["python_version"] = platform.python_version()
            res["handler_wall_ms"] = round((time.perf_counter() - t_req_start) * 1000.0, 3)
            return 200, res

        if method in ("GET", "POST") and path == "/api/kvcache/run":
            num_requests = int(params.get("num_requests", 64))
            block_size = int(params.get("block_size", 16))
            max_seq_len = int(params.get("max_seq_len", 1024))
            system_prompt_tokens = int(params.get("system_prompt_tokens", 256))
            memory_cap_mb = int(params.get("memory_cap_mb", 1280))
            res = run_paged_kv_benchmark(
                num_requests=num_requests,
                block_size=block_size,
                max_seq_len=max_seq_len,
                system_prompt_tokens=system_prompt_tokens,
                memory_cap_mb=memory_cap_mb,
            )
            res["request_id"] = request_id
            res["python_version"] = platform.python_version()
            res["handler_wall_ms"] = round((time.perf_counter() - t_req_start) * 1000.0, 3)
            return 200, res

        if method in ("GET", "POST") and path == "/api/correlator/run":
            scenario_id = str(params.get("scenario_id") or "oom_allocator_trap").strip()
            num_workers = int(params.get("num_workers", 64))
            root_rank = int(params.get("root_rank", 19))
            trials = int(params.get("trials", 48))
            seed = int(params.get("seed", 42))

            events, truth = build_worker_fault_stream(
                scenario_id=scenario_id,
                num_workers=num_workers,
                root_rank=root_rank,
                shuffle_seed=seed,
            )
            single = CrossNodeErrorCorrelator().correlate(events)
            suite = run_correlator_benchmark(trials=trials, num_workers=num_workers, seed=seed)
            return 200, {
                "request_id": request_id,
                "python_version": platform.python_version(),
                "scenario": truth,
                "single_stream_correlation": single,
                "shuffled_suite_eval": suite,
                "scenarios_available": list(DISTRIBUTED_FAULT_SCENARIOS.values()),
                "handler_wall_ms": round((time.perf_counter() - t_req_start) * 1000.0, 3),
            }

        if method in ("GET", "POST") and path == "/api/quant-spec":
            alpha = float(params.get("alpha", 0.5))
            group_size = int(params.get("group_size", 32))
            gamma = int(params.get("gamma", 4))
            draft_alignment = float(params.get("draft_alignment", 0.80))
            max_buckets = int(params.get("max_buckets", 5))
            top_k = int(params.get("top_k", 2))
            res = run_quant_spec_suite(
                alpha=alpha,
                group_size=group_size,
                gamma=gamma,
                draft_alignment=draft_alignment,
                max_buckets=max_buckets,
                top_k=top_k,
            )
            res["request_id"] = request_id
            res["python_version"] = platform.python_version()
            res["handler_wall_ms"] = round((time.perf_counter() - t_req_start) * 1000.0, 3)
            return 200, res

        if method in ("GET", "POST") and path == "/api/transport/ring":
            ring_capacity = int(params.get("ring_capacity", 256))
            num_messages = int(params.get("num_messages", 4096))
            batch_size = int(params.get("batch_size", 32))
            num_mr_regions = int(params.get("num_mr_regions", 128))
            num_mr_lookups = int(params.get("num_mr_lookups", 1024))
            tensor_bytes = int(params.get("tensor_bytes", 1048576))
            res = run_transport_ring_mr_benchmark(
                ring_capacity=ring_capacity,
                num_messages=num_messages,
                batch_size=batch_size,
                num_mr_regions=num_mr_regions,
                num_mr_lookups=num_mr_lookups,
                tensor_bytes=tensor_bytes,
            )
            res["request_id"] = request_id
            res["python_version"] = platform.python_version()
            res["handler_wall_ms"] = round((time.perf_counter() - t_req_start) * 1000.0, 3)
            return 200, res

        if method in ("GET", "POST") and path == "/api/bedrock/compare":
            prompt_text = str(params.get("prompt") or "").strip()
            num_prompts = max(1, min(5, int(params.get("num_prompts", 1))))
            max_tokens = max(32, min(250, int(params.get("max_tokens", 120))))
            warmup = _to_bool(params.get("warmup"), default=True)

            if prompt_text:
                prompts = [prompt_text]
            else:
                prompts = DEFAULT_BENCHMARK_PROMPTS[:num_prompts]

            region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))
            profiler = BedrockModelProfiler(
                region_name=region,
                converse_fn=_TEST_BEDROCK_CONVERSE_FN,
            )
            res = profiler.compare_nova_micro_vs_lite(
                prompts=prompts,
                max_tokens=max_tokens,
                warmup=warmup,
            )
            res["request_id"] = request_id
            res["python_version"] = platform.python_version()
            res["handler_wall_ms"] = round((time.perf_counter() - t_req_start) * 1000.0, 3)
            return 200, res

        return 404, {"error": f"Unknown API route: {method} {path}"}
    except KeyError as exc:
        return 404, {"error": str(exc)}
    except ValueError as exc:
        return 400, {"error": str(exc)}
    except Exception as exc:
        return 500, {"error": f"{type(exc).__name__}: {exc}"}


def lambda_handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """AWS Lambda handler mounted behind Amazon CloudFront `/api/*` origin."""
    req_ctx = event.get("requestContext") or {}
    method = ((req_ctx.get("http") or {}).get("method") or event.get("httpMethod") or "GET").upper()
    raw_path = event.get("rawPath") or event.get("path") or "/"
    query_params = {k: str(v) for k, v in (event.get("queryStringParameters") or {}).items() if v is not None}
    req_id = getattr(context, "aws_request_id", "cloudfront-lambda")

    if method == "OPTIONS":
        return {"statusCode": 204, "headers": CORS_HEADERS, "body": ""}

    if raw_path.startswith("/api/"):
        body: Dict[str, Any] = {}
        raw_body = event.get("body")
        if raw_body:
            if event.get("isBase64Encoded") and isinstance(raw_body, str):
                raw_body = base64.b64decode(raw_body).decode("utf-8")
            body = json.loads(raw_body) if isinstance(raw_body, str) else dict(raw_body)
        status, payload = route_api_request(method, raw_path, query_params, body, request_id=req_id)
        return {
            "statusCode": status,
            "headers": {
                **CORS_HEADERS,
                "Content-Type": "application/json; charset=utf-8",
                "Cache-Control": "no-store",
            },
            "body": json.dumps(payload, default=str),
        }

    target = WEB_DIR / ("index.html" if raw_path in ("/", "") else raw_path.lstrip("/"))
    if not target.is_file() and raw_path.endswith(".svg"):
        target = DOCS_DIR / Path(raw_path).name
    if target.is_file():
        ctype = mimetypes.guess_type(str(target))[0] or "text/html; charset=utf-8"
        return {
            "statusCode": 200,
            "headers": {**CORS_HEADERS, "Content-Type": ctype},
            "body": target.read_text(encoding="utf-8"),
        }
    return {
        "statusCode": 404,
        "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
        "body": json.dumps({"error": "Not found", "path": raw_path}),
    }


class LocalNeuronPulseHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = {k: v[0] for k, v in parse_qs(parsed.query).items() if v}

        if path.startswith("/api/"):
            status, payload = route_api_request("GET", path, query, {}, request_id="local-dev")
            raw = json.dumps(payload, indent=2, default=str).encode("utf-8")
            self.send_response(status)
            for k, v in CORS_HEADERS.items():
                self.send_header(k, v)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        candidate = (WEB_DIR / rel).resolve()
        if not candidate.is_file() and path.endswith(".svg"):
            candidate = (DOCS_DIR / Path(path).name).resolve()
        if not candidate.is_file():
            candidate = WEB_DIR / "index.html"

        data = candidate.read_bytes()
        ctype = mimetypes.guess_type(str(candidate))[0] or "text/html"
        if ctype.startswith("text/") or ctype in ("application/javascript", "image/svg+xml"):
            ctype += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(parsed.query).items() if v}
        length = int(self.headers.get("Content-Length") or "0")
        raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        body = json.loads(raw_body) if raw_body.strip() else {}
        status, payload = route_api_request("POST", parsed.path, query, body, request_id="local-dev")
        raw = json.dumps(payload, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def run_local_server(port: int = 3096) -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", port), LocalNeuronPulseHandler)
    print(f"NeuronPulse ML Systems Lab running at http://localhost:{port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    run_local_server(port=int(os.environ.get("PORT", "3096")))
