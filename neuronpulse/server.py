"""HTTP Server & AWS CloudFront `/api/*` Lambda Origin Handler for NeuronPulse."""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import platform
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.parse import parse_qs, urlparse

from .copilot import ask_cluster_copilot
from .correlator import CrossNodeErrorCorrelator, execute_automated_repair
from .profiler import compare_serving_configs
from .topology import UltraServerCluster

ROOT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT_DIR / "web"
DOCS_DIR = ROOT_DIR / "docs"

CLUSTER = UltraServerCluster()
CORRELATOR = CrossNodeErrorCorrelator()
ACTIVE_PROFILE_CFG: Dict[str, Any] = {
    "tp_degree": 8,
    "precision": "FP8",
    "batch_size": 16,
    "nki_flash_fusion": True,
    "seq_len": 4096,
}

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
}


def route_api_request(
    method: str,
    path: str,
    query: Dict[str, str],
    body: Dict[str, Any],
    request_id: str = "local",
) -> Tuple[int, Dict[str, Any]]:
    """Pure router shared by local HTTP server and AWS Lambda `/api/*` origin behind CloudFront."""
    try:
        if method == "GET" and path == "/api/health":
            snap = CLUSTER.snapshot()
            return 200, {
                "status": "ok",
                "service": "neuronpulse-cluster-os",
                "delivery": os.environ.get("NEURONPULSE_DELIVERY", "cloudfront-s3-oac + api-origin"),
                "aws_region": os.environ.get("AWS_REGION", "us-west-2"),
                "bedrock_model": os.environ.get("NEURONPULSE_BEDROCK_MODEL", "us.amazon.nova-lite-v1:0"),
                "python_version": platform.python_version(),
                "request_id": request_id,
                "cluster_id": snap["cluster_id"],
                "total_neuron_cores": snap["summary"]["total_neuron_cores"],
                "healthy_cores": snap["summary"]["healthy_cores"],
            }

        if method == "GET" and path == "/api/state":
            snap = CLUSTER.snapshot()
            prof = compare_serving_configs(**ACTIVE_PROFILE_CFG)
            return 200, {
                "request_id": request_id,
                "cluster": snap,
                "profiler": prof,
            }

        if method == "POST" and path == "/api/faults/inject":
            fault_id = str(body.get("fault_id") or "fault_hbm_ecc").strip()
            CLUSTER.inject_fault(fault_id)
            corr = CORRELATOR.correlate(CLUSTER.raw_events)
            CLUSTER.last_correlation = corr
            return 200, {
                "request_id": request_id,
                "status": "FAULT_INJECTED",
                "fault_id": fault_id,
                "correlation": corr,
                "cluster": CLUSTER.snapshot(),
            }

        if method == "POST" and path == "/api/repair/execute":
            res = execute_automated_repair(CLUSTER)
            res["request_id"] = request_id
            return 200, res

        if method == "POST" and path == "/api/profiler/simulate":
            tp_degree = int(body.get("tp_degree", ACTIVE_PROFILE_CFG["tp_degree"]))
            precision = str(body.get("precision", ACTIVE_PROFILE_CFG["precision"])).upper()
            batch_size = int(body.get("batch_size", ACTIVE_PROFILE_CFG["batch_size"]))
            nki_flash = bool(body.get("nki_flash_fusion", ACTIVE_PROFILE_CFG["nki_flash_fusion"]))
            seq_len = int(body.get("seq_len", ACTIVE_PROFILE_CFG["seq_len"]))
            prof = compare_serving_configs(
                tp_degree=tp_degree,
                precision=precision,
                batch_size=batch_size,
                nki_flash_fusion=nki_flash,
                seq_len=seq_len,
            )
            ACTIVE_PROFILE_CFG.update(
                {
                    "tp_degree": tp_degree,
                    "precision": precision,
                    "batch_size": batch_size,
                    "nki_flash_fusion": nki_flash,
                    "seq_len": seq_len,
                }
            )
            return 200, {
                "request_id": request_id,
                "profiler": prof,
            }

        if method == "POST" and path == "/api/copilot/ask":
            question = str(body.get("question") or "").strip()
            if not question:
                return 400, {"error": "question is required"}
            prof = compare_serving_configs(**ACTIVE_PROFILE_CFG)
            out = ask_cluster_copilot(
                question=question,
                cluster_snapshot=CLUSTER.snapshot(),
                profiler_result=prof,
            )
            out["request_id"] = request_id
            return 200, out

        if method == "POST" and path == "/api/reset":
            CLUSTER.reset()
            ACTIVE_PROFILE_CFG.update(
                {
                    "tp_degree": 8,
                    "precision": "FP8",
                    "batch_size": 16,
                    "nki_flash_fusion": True,
                    "seq_len": 4096,
                }
            )
            return 200, {
                "status": "reset_complete",
                "cluster": CLUSTER.snapshot(),
                "profiler": compare_serving_configs(**ACTIVE_PROFILE_CFG),
            }

        return 404, {"error": f"Unknown API route: {method} {path}"}
    except KeyError as exc:
        return 404, {"error": str(exc)}
    except ValueError as exc:
        return 400, {"error": str(exc)}
    except Exception as exc:
        return 500, {"error": str(exc)}


def lambda_handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """AWS Lambda handler mounted behind Amazon CloudFront `/api/*` origin."""
    req_ctx = event.get("requestContext") or {}
    method = ((req_ctx.get("http") or {}).get("method") or event.get("httpMethod") or "GET").upper()
    raw_path = event.get("rawPath") or event.get("path") or "/"
    query_params = {k: str(v) for k, v in (event.get("queryStringParameters") or {}).items() if v is not None}
    req_id = getattr(context, "aws_request_id", "cloudfront-api")

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
        length = int(self.headers.get("Content-Length") or "0")
        raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        body = json.loads(raw_body) if raw_body.strip() else {}
        status, payload = route_api_request("POST", parsed.path, {}, body, request_id="local-dev")
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
    print(f"NeuronPulse UltraServer OS running at http://localhost:{port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    run_local_server(port=int(os.environ.get("PORT", "3096")))
