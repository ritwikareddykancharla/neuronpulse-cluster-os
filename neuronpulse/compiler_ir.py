"""Real ML Computation-Graph Compiler Pass & Typed Tensor IR Interpreter.

Implements a typed Static Single Assignment (SSA) tensor computation graph and three
executable compiler optimization passes:
  1. ConstantFold + Common Subexpression Elimination (CSE)
  2. Backward Liveness Dead Code Elimination (DCE)
  3. Epilogue Operator Fusion (`MatMul` + `AddBias` + `GELU` -> `FusedMatMulBiasGelu`)

Executes BOTH the Unoptimized IR and Optimized Fused IR on real IEEE-754 float64
matrices in Python, verifies exact numerical equivalence (`max_abs_diff < 1e-9`),
and measures real `tracemalloc` peak intermediate tensor bytes saved (`%`),
IR operator node reduction, and `time.perf_counter()` execution latency (`ms`).
"""
from __future__ import annotations

import gc
import math
import random
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

INV_SQRT2 = 1.0 / math.sqrt(2.0)
_SUMPROD = math.sumprod


@dataclass(frozen=True)
class IRNode:
    """Single typed SSA node inside a tensor computation graph."""

    name: str
    op: str
    inputs: Tuple[str, ...] = ()
    shape: Tuple[int, ...] = ()
    dtype: str = "float64"
    attrs: Dict[str, Any] = field(default_factory=dict)

    def format_ir(self) -> str:
        shape_str = "x".join(str(d) for d in self.shape) if self.shape else "scalar"
        args = ", ".join(f"%{inp}" for inp in self.inputs)
        attr_str = ""
        if self.op == "Constant":
            val = self.attrs.get("value")
            attr_str = f" [value={val}]"
        elif self.attrs:
            kv = ", ".join(f"{k}={v}" for k, v in sorted(self.attrs.items()))
            attr_str = f" [{kv}]"
        return f"%{self.name}: f64<{shape_str}> = {self.op}({args}){attr_str}"


@dataclass
class ComputationGraph:
    """Ordered SSA DAG of typed tensor operations."""

    name: str
    nodes: List[IRNode]
    outputs: List[str]

    def clone(self) -> "ComputationGraph":
        return ComputationGraph(
            name=self.name,
            nodes=[
                IRNode(
                    name=n.name,
                    op=n.op,
                    inputs=tuple(n.inputs),
                    shape=tuple(n.shape),
                    dtype=n.dtype,
                    attrs=dict(n.attrs),
                )
                for n in self.nodes
            ],
            outputs=list(self.outputs),
        )

    def compute_op_nodes(self) -> List[IRNode]:
        """Return executable operator nodes (excluding Input declarations)."""
        return [n for n in self.nodes if n.op != "Input"]

    def dump_ir(self) -> str:
        lines = [f"graph @{self.name} {{"]
        for node in self.nodes:
            lines.append(f"  {node.format_ir()}")
        outs = ", ".join(f"%{o}" for o in self.outputs)
        lines.append(f"  return ({outs})")
        lines.append("}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Real Numerical Tensor Kernels (Pure Python 3.12 IEEE-754 float64)
# ---------------------------------------------------------------------------


def _gelu_scalar(x: float) -> float:
    return 0.5 * x * (1.0 + math.erf(x * INV_SQRT2))


def _op_layer_norm(
    x: List[List[float]],
    gamma: List[float],
    beta: List[float],
    eps: float = 1e-5,
) -> List[List[float]]:
    cols = len(gamma)
    inv_cols = 1.0 / cols
    out: List[List[float]] = []
    for row in x:
        mean = sum(row) * inv_cols
        var = sum((v - mean) * (v - mean) for v in row) * inv_cols
        inv_std = 1.0 / math.sqrt(var + eps)
        out.append([(row[c] - mean) * inv_std * gamma[c] + beta[c] for c in range(cols)])
    return out


def _op_matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    # Transpose B once into contiguous column tuples for zero-allocation C-speed math.sumprod
    b_t = list(zip(*b))
    sp = _SUMPROD
    return [[sp(row_a, col_b) for col_b in b_t] for row_a in a]


def _op_add_bias(x: List[List[float]], bias: List[float]) -> List[List[float]]:
    cols = len(bias)
    return [[row[c] + bias[c] for c in range(cols)] for row in x]


def _op_gelu(x: List[List[float]]) -> List[List[float]]:
    erf = math.erf
    inv_s2 = INV_SQRT2
    return [[0.5 * v * (1.0 + erf(v * inv_s2)) for v in row] for row in x]


def _op_fused_matmul_bias_gelu(
    a: List[List[float]],
    w: List[List[float]],
    bias: List[float],
) -> List[List[float]]:
    """Fused epilogue kernel: computes GELU((A @ W) + b) in a single output allocation.

    Eliminates both the intermediate `(M, H)` MatMul tensor and the intermediate
    `(M, H)` AddBias tensor from heap memory.
    """
    w_t = list(zip(*w))
    erf = math.erf
    inv_s2 = INV_SQRT2
    sp = _SUMPROD
    out: List[List[float]] = []
    for row_a in a:
        out_row: List[float] = []
        for col_w, b_val in zip(w_t, bias):
            z = sp(row_a, col_w) + b_val
            out_row.append(0.5 * z * (1.0 + erf(z * inv_s2)))
        out.append(out_row)
    return out


def _op_mul(a: Any, b: Any) -> Any:
    a_scalar = isinstance(a, (int, float))
    b_scalar = isinstance(b, (int, float))
    if a_scalar and b_scalar:
        return float(a) * float(b)
    if a_scalar and not b_scalar:
        s = float(a)
        if s == 1.0:
            return [list(row) for row in b]
        return [[s * v for v in row] for row in b]
    if not a_scalar and b_scalar:
        s = float(b)
        if s == 1.0:
            return [list(row) for row in a]
        return [[v * s for v in row] for row in a]
    return [[ra[c] * rb[c] for c in range(len(ra))] for ra, rb in zip(a, b)]


def _op_residual_add(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    cols = len(a[0])
    return [[ra[c] + rb[c] for c in range(cols)] for ra, rb in zip(a, b)]


# ---------------------------------------------------------------------------
# Three Real Compiler Optimization Passes
# ---------------------------------------------------------------------------


def pass_constant_fold_and_cse(graph: ComputationGraph) -> Tuple[ComputationGraph, Dict[str, Any]]:
    """Pass 1: Fold compile-time Constant arithmetic & deduplicate identical SSA expressions."""
    g = graph.clone()
    alias: Dict[str, str] = {}
    const_values: Dict[str, Any] = {}
    expr_table: Dict[Tuple[Any, ...], str] = {}
    folded_constants = 0
    cse_eliminated = 0
    new_nodes: List[IRNode] = []

    for node in g.nodes:
        resolved_inputs = tuple(alias.get(inp, inp) for inp in node.inputs)

        if node.op == "Constant":
            val = node.attrs["value"]
            key = ("Constant", node.shape, val)
            if key in expr_table:
                alias[node.name] = expr_table[key]
                cse_eliminated += 1
                continue
            expr_table[key] = node.name
            const_values[node.name] = val
            new_nodes.append(node)
            continue

        # Constant Folding for pure scalar/constant Mul
        if node.op == "Mul" and all(inp in const_values for inp in resolved_inputs):
            folded_val = float(const_values[resolved_inputs[0]]) * float(const_values[resolved_inputs[1]])
            folded_node = IRNode(
                name=node.name,
                op="Constant",
                inputs=(),
                shape=node.shape,
                dtype=node.dtype,
                attrs={"value": folded_val},
            )
            const_values[node.name] = folded_val
            expr_table[("Constant", node.shape, folded_val)] = node.name
            folded_constants += 1
            new_nodes.append(folded_node)
            continue

        # Algebraic identity: Mul(X, Constant(1.0)) -> X
        if node.op == "Mul" and len(resolved_inputs) == 2:
            lhs, rhs = resolved_inputs
            if rhs in const_values and float(const_values[rhs]) == 1.0:
                alias[node.name] = lhs
                folded_constants += 1
                continue
            if lhs in const_values and float(const_values[lhs]) == 1.0:
                alias[node.name] = rhs
                folded_constants += 1
                continue

        if node.op != "Input":
            canon_inputs = tuple(sorted(resolved_inputs)) if node.op in ("Mul", "ResidualAdd") else resolved_inputs
            cse_key = (
                node.op,
                canon_inputs,
                node.shape,
                tuple(sorted(node.attrs.items())),
            )
            if cse_key in expr_table:
                alias[node.name] = expr_table[cse_key]
                cse_eliminated += 1
                continue
            expr_table[cse_key] = node.name

        updated = IRNode(
            name=node.name,
            op=node.op,
            inputs=resolved_inputs,
            shape=node.shape,
            dtype=node.dtype,
            attrs=dict(node.attrs),
        )
        new_nodes.append(updated)

    new_outputs = [alias.get(o, o) for o in g.outputs]
    out_graph = ComputationGraph(name=g.name, nodes=new_nodes, outputs=new_outputs)
    return out_graph, {
        "pass": "ConstantFold + CSE",
        "folded_constants": folded_constants,
        "cse_eliminated": cse_eliminated,
        "op_nodes_after": len(out_graph.compute_op_nodes()),
    }


def pass_dead_code_elimination(graph: ComputationGraph) -> Tuple[ComputationGraph, Dict[str, Any]]:
    """Pass 2: Backward reachability liveness sweep pruning unused ops, debug taps & dead inputs."""
    node_map = {n.name: n for n in graph.nodes}
    live: Set[str] = set()
    stack = list(graph.outputs)
    while stack:
        cur = stack.pop()
        if cur in live or cur not in node_map:
            continue
        live.add(cur)
        for inp in node_map[cur].inputs:
            if inp not in live:
                stack.append(inp)

    before_ops = len(graph.compute_op_nodes())
    before_total = len(graph.nodes)
    kept_nodes = [n for n in graph.nodes if n.name in live]
    out_graph = ComputationGraph(name=graph.name, nodes=kept_nodes, outputs=list(graph.outputs))
    after_ops = len(out_graph.compute_op_nodes())
    return out_graph, {
        "pass": "DeadCodeElimination",
        "pruned_op_nodes": before_ops - after_ops,
        "pruned_total_nodes": before_total - len(kept_nodes),
        "op_nodes_after": after_ops,
    }


def pass_matmul_bias_gelu_fusion(graph: ComputationGraph) -> Tuple[ComputationGraph, Dict[str, Any]]:
    """Pass 3: Pattern-match single-use `MatMul -> AddBias -> GELU` chains into `FusedMatMulBiasGelu`."""
    use_counts: Dict[str, int] = {n.name: 0 for n in graph.nodes}
    for n in graph.nodes:
        for inp in n.inputs:
            use_counts[inp] = use_counts.get(inp, 0) + 1
    for out_name in graph.outputs:
        use_counts[out_name] = use_counts.get(out_name, 0) + 1

    node_map = {n.name: n for n in graph.nodes}
    absorbed: Set[str] = set()
    rewritten: Dict[str, IRNode] = {}
    fused_chains = 0

    for node in graph.nodes:
        if node.op != "GELU" or len(node.inputs) != 1:
            continue
        bias_name = node.inputs[0]
        bias_node = node_map.get(bias_name)
        if not bias_node or bias_node.op != "AddBias" or use_counts.get(bias_name, 0) != 1:
            continue
        mm_name = bias_node.inputs[0]
        mm_node = node_map.get(mm_name)
        if not mm_node or mm_node.op != "MatMul" or use_counts.get(mm_name, 0) != 1:
            continue

        # Safe to fuse 3-op chain into 1 epilogue-fused operator!
        absorbed.add(mm_name)
        absorbed.add(bias_name)
        rewritten[node.name] = IRNode(
            name=node.name,
            op="FusedMatMulBiasGelu",
            inputs=(mm_node.inputs[0], mm_node.inputs[1], bias_node.inputs[1]),
            shape=node.shape,
            dtype=node.dtype,
            attrs={"fused_ops": "MatMul+AddBias+GELU"},
        )
        fused_chains += 1

    final_nodes: List[IRNode] = []
    for n in graph.nodes:
        if n.name in absorbed:
            continue
        final_nodes.append(rewritten.get(n.name, n))

    out_graph = ComputationGraph(name=f"{graph.name}_opt", nodes=final_nodes, outputs=list(graph.outputs))
    return out_graph, {
        "pass": "MatMulBiasGeluFusion",
        "fused_epilogue_chains": fused_chains,
        "eliminated_intermediate_tensors": fused_chains * 2,
        "op_nodes_after": len(out_graph.compute_op_nodes()),
    }


def compile_computation_graph(graph: ComputationGraph) -> Tuple[ComputationGraph, List[Dict[str, Any]]]:
    """Run all 3 compiler optimization passes sequentially and return optimized IR + pass trace."""
    g1, s1 = pass_constant_fold_and_cse(graph)
    g2, s2 = pass_dead_code_elimination(g1)
    g3, s3 = pass_matmul_bias_gelu_fusion(g2)
    return g3, [s1, s2, s3]


# ---------------------------------------------------------------------------
# Graph Presets & Deterministic Real Numerical Tensor Initialization
# ---------------------------------------------------------------------------


def build_preset_graph(
    preset: str = "transformer_mlp_block",
    batch_tokens: int = 64,
    hidden_dim: int = 64,
) -> Tuple[ComputationGraph, Dict[str, Any], Dict[str, Any]]:
    """Construct a real typed computation graph and deterministic numerical input matrices."""
    b = max(8, min(192, int(batch_tokens)))
    d = max(16, min(192, int(hidden_dim)))
    preset_key = preset.strip().lower()
    if preset_key not in ("transformer_mlp_block", "stacked_2layer_mlp", "wide_ffn_epilogue"):
        raise ValueError(
            "preset must be one of: transformer_mlp_block, stacked_2layer_mlp, wide_ffn_epilogue"
        )

    ffn_mult = 6 if preset_key == "wide_ffn_epilogue" else 4
    h = d * ffn_mult
    rng = random.Random(1337 + b * 31 + d * 7)

    def rand_mat(rows: int, cols: int, scale: float = 0.18) -> List[List[float]]:
        return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]

    def rand_vec(cols: int, center: float = 0.0, scale: float = 0.08) -> List[float]:
        return [center + rng.uniform(-scale, scale) for _ in range(cols)]

    nodes: List[IRNode] = []
    bindings: Dict[str, Any] = {}

    # Shared inputs
    nodes.append(IRNode("x", "Input", (), (b, d)))
    nodes.append(IRNode("ln1_gamma", "Input", (), (d,)))
    nodes.append(IRNode("ln1_beta", "Input", (), (d,)))
    nodes.append(IRNode("w_up1", "Input", (), (d, h)))
    nodes.append(IRNode("b_up1", "Input", (), (h,)))
    nodes.append(IRNode("w_down1", "Input", (), (h, d)))
    nodes.append(IRNode("b_down1", "Input", (), (d,)))
    nodes.append(IRNode("w_debug_probe", "Input", (), (d, h)))

    bindings["x"] = rand_mat(b, d, 0.5)
    bindings["ln1_gamma"] = rand_vec(d, center=1.0, scale=0.05)
    bindings["ln1_beta"] = rand_vec(d, center=0.0, scale=0.03)
    bindings["w_up1"] = rand_mat(d, h, 1.0 / math.sqrt(d))
    bindings["b_up1"] = rand_vec(h, center=0.01, scale=0.04)
    bindings["w_down1"] = rand_mat(h, d, 1.0 / math.sqrt(h))
    bindings["b_down1"] = rand_vec(d, center=0.0, scale=0.03)
    bindings["w_debug_probe"] = rand_mat(d, h, 0.1)

    # Compile-time foldable constants
    nodes.append(IRNode("c_half", "Constant", (), (1,), attrs={"value": 0.5}))
    nodes.append(IRNode("c_two", "Constant", (), (1,), attrs={"value": 2.0}))
    nodes.append(IRNode("c_unit_scale", "Mul", ("c_half", "c_two"), (1,)))
    nodes.append(IRNode("c_probe_damp", "Constant", (), (1,), attrs={"value": 0.01}))

    # Layer 1 with duplicate CSE subexpressions + dead debug probe + MatMul->AddBias->GELU chain
    nodes.append(IRNode("ln1_main", "LayerNorm", ("x", "ln1_gamma", "ln1_beta"), (b, d), attrs={"eps": 1e-5}))
    nodes.append(IRNode("ln1_dup", "LayerNorm", ("x", "ln1_gamma", "ln1_beta"), (b, d), attrs={"eps": 1e-5}))

    nodes.append(IRNode("mm_up1", "MatMul", ("ln1_main", "w_up1"), (b, h)))
    nodes.append(IRNode("mm_up1_dup", "MatMul", ("ln1_dup", "w_up1"), (b, h)))
    nodes.append(IRNode("bias_up1", "AddBias", ("mm_up1", "b_up1"), (b, h)))
    nodes.append(IRNode("act_up1", "GELU", ("bias_up1",), (b, h)))

    # Dead debug / telemetry tap subgraphs (present in eager export, unused by final output)
    nodes.append(IRNode("debug_dup_bias", "AddBias", ("mm_up1_dup", "b_up1"), (b, h)))
    nodes.append(IRNode("debug_dup_act", "GELU", ("debug_dup_bias",), (b, h)))
    nodes.append(IRNode("debug_probe_mm", "MatMul", ("ln1_dup", "w_debug_probe"), (b, h)))
    nodes.append(IRNode("debug_probe_scaled", "Mul", ("debug_probe_mm", "c_probe_damp"), (b, h)))

    nodes.append(IRNode("mm_down1", "MatMul", ("act_up1", "w_down1"), (b, d)))
    nodes.append(IRNode("bias_down1", "AddBias", ("mm_down1", "b_down1"), (b, d)))
    nodes.append(IRNode("scaled_down1", "Mul", ("bias_down1", "c_unit_scale"), (b, d)))
    nodes.append(IRNode("res_out1", "ResidualAdd", ("x", "scaled_down1"), (b, d)))

    final_output = "res_out1"

    if preset_key == "stacked_2layer_mlp":
        nodes.append(IRNode("ln2_gamma", "Input", (), (d,)))
        nodes.append(IRNode("ln2_beta", "Input", (), (d,)))
        nodes.append(IRNode("w_up2", "Input", (), (d, h)))
        nodes.append(IRNode("b_up2", "Input", (), (h,)))
        nodes.append(IRNode("w_down2", "Input", (), (h, d)))
        nodes.append(IRNode("b_down2", "Input", (), (d,)))

        bindings["ln2_gamma"] = rand_vec(d, center=1.0, scale=0.05)
        bindings["ln2_beta"] = rand_vec(d, center=0.0, scale=0.03)
        bindings["w_up2"] = rand_mat(d, h, 1.0 / math.sqrt(d))
        bindings["b_up2"] = rand_vec(h, center=0.01, scale=0.04)
        bindings["w_down2"] = rand_mat(h, d, 1.0 / math.sqrt(h))
        bindings["b_down2"] = rand_vec(d, center=0.0, scale=0.03)

        nodes.append(IRNode("ln2_main", "LayerNorm", ("res_out1", "ln2_gamma", "ln2_beta"), (b, d), attrs={"eps": 1e-5}))
        nodes.append(IRNode("ln2_dup", "LayerNorm", ("res_out1", "ln2_gamma", "ln2_beta"), (b, d), attrs={"eps": 1e-5}))
        nodes.append(IRNode("debug_ln2_probe", "MatMul", ("ln2_dup", "w_debug_probe"), (b, h)))

        nodes.append(IRNode("mm_up2", "MatMul", ("ln2_main", "w_up2"), (b, h)))
        nodes.append(IRNode("bias_up2", "AddBias", ("mm_up2", "b_up2"), (b, h)))
        nodes.append(IRNode("act_up2", "GELU", ("bias_up2",), (b, h)))
        nodes.append(IRNode("mm_down2", "MatMul", ("act_up2", "w_down2"), (b, d)))
        nodes.append(IRNode("bias_down2", "AddBias", ("mm_down2", "b_down2"), (b, d)))
        nodes.append(IRNode("res_out2", "ResidualAdd", ("res_out1", "bias_down2"), (b, d)))
        final_output = "res_out2"

    graph = ComputationGraph(name=preset_key, nodes=nodes, outputs=[final_output])
    meta = {
        "preset": preset_key,
        "batch_tokens": b,
        "hidden_dim": d,
        "ffn_dim": h,
    }
    return graph, bindings, meta


# ---------------------------------------------------------------------------
# Eager Unoptimized vs Liveness-Scheduled Fused IR Execution
# ---------------------------------------------------------------------------


def _eval_single_node(node: IRNode, env: Dict[str, Any]) -> Any:
    op = node.op
    if op == "Constant":
        return node.attrs["value"]
    if op == "LayerNorm":
        return _op_layer_norm(
            env[node.inputs[0]],
            env[node.inputs[1]],
            env[node.inputs[2]],
            float(node.attrs.get("eps", 1e-5)),
        )
    if op == "MatMul":
        return _op_matmul(env[node.inputs[0]], env[node.inputs[1]])
    if op == "AddBias":
        return _op_add_bias(env[node.inputs[0]], env[node.inputs[1]])
    if op == "GELU":
        return _op_gelu(env[node.inputs[0]])
    if op == "FusedMatMulBiasGelu":
        return _op_fused_matmul_bias_gelu(
            env[node.inputs[0]],
            env[node.inputs[1]],
            env[node.inputs[2]],
        )
    if op == "Mul":
        return _op_mul(env[node.inputs[0]], env[node.inputs[1]])
    if op == "ResidualAdd":
        return _op_residual_add(env[node.inputs[0]], env[node.inputs[1]])
    raise ValueError(f"Unsupported IR op: {op}")


def execute_graph_eager(graph: ComputationGraph, bindings: Dict[str, Any]) -> List[List[float]]:
    """Execute unoptimized IR retaining full eager intermediate activation tape."""
    env: Dict[str, Any] = {}
    for node in graph.nodes:
        if node.op == "Input":
            env[node.name] = bindings[node.name]
        else:
            env[node.name] = _eval_single_node(node, env)
    return env[graph.outputs[0]]


def execute_graph_optimized(graph: ComputationGraph, bindings: Dict[str, Any]) -> List[List[float]]:
    """Execute compiled fused IR with SSA last-use buffer liveness release."""
    input_names = {n.name for n in graph.nodes if n.op == "Input"}
    output_set = set(graph.outputs)

    # Precompute last execution step index for each intermediate SSA tensor
    last_use: Dict[str, int] = {}
    for idx, node in enumerate(graph.nodes):
        for inp in node.inputs:
            last_use[inp] = idx

    env: Dict[str, Any] = {}
    for idx, node in enumerate(graph.nodes):
        if node.op == "Input":
            env[node.name] = bindings[node.name]
            continue
        env[node.name] = _eval_single_node(node, env)
        for inp in node.inputs:
            if inp not in input_names and inp not in output_set and last_use.get(inp, -1) <= idx:
                env.pop(inp, None)
    return env[graph.outputs[0]]


def _max_abs_diff_2d(a: List[List[float]], b: List[List[float]]) -> float:
    max_err = 0.0
    for row_a, row_b in zip(a, b):
        for va, vb in zip(row_a, row_b):
            diff = abs(va - vb)
            if diff > max_err:
                max_err = diff
    return max_err


def _measure_execution(fn: Any, graph: ComputationGraph, bindings: Dict[str, Any], repeats: int = 3) -> Tuple[List[List[float]], int, float]:
    """Measure real `tracemalloc` peak heap bytes and median `time.perf_counter()` ms."""
    gc.collect()
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()
    out = fn(graph, bindings)
    _, peak_bytes = tracemalloc.get_traced_memory()
    if not was_tracing:
        tracemalloc.stop()

    timings_ms: List[float] = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        out = fn(graph, bindings)
        timings_ms.append((time.perf_counter() - t0) * 1000.0)
    timings_ms.sort()
    median_ms = timings_ms[len(timings_ms) // 2]
    return out, int(peak_bytes), round(median_ms, 3)


def run_compiler_pass_benchmark(
    preset: str = "stacked_2layer_mlp",
    batch_tokens: int = 64,
    hidden_dim: int = 64,
    repeats: int = 3,
) -> Dict[str, Any]:
    """Compile and execute both Unoptimized and Optimized IR graphs with real measured metrics."""
    t_compile_start = time.perf_counter()
    unopt_graph, bindings, meta = build_preset_graph(
        preset=preset,
        batch_tokens=batch_tokens,
        hidden_dim=hidden_dim,
    )
    opt_graph, pass_trace = compile_computation_graph(unopt_graph)
    compile_pass_ms = round((time.perf_counter() - t_compile_start) * 1000.0, 3)

    unopt_out, unopt_peak_bytes, unopt_ms = _measure_execution(
        execute_graph_eager, unopt_graph, bindings, repeats=repeats
    )
    opt_out, opt_peak_bytes, opt_ms = _measure_execution(
        execute_graph_optimized, opt_graph, bindings, repeats=repeats
    )

    max_diff = _max_abs_diff_2d(unopt_out, opt_out)
    unopt_ops = len(unopt_graph.compute_op_nodes())
    opt_ops = len(opt_graph.compute_op_nodes())
    node_reduction_pct = round(((unopt_ops - opt_ops) / max(1, unopt_ops)) * 100.0, 1)
    mem_saved_pct = round(((unopt_peak_bytes - opt_peak_bytes) / max(1, unopt_peak_bytes)) * 100.0, 1)
    speedup_x = round(unopt_ms / max(0.001, opt_ms), 2)
    latency_saved_pct = round(((unopt_ms - opt_ms) / max(0.001, unopt_ms)) * 100.0, 1)

    return {
        "preset": meta["preset"],
        "dimensions": {
            "batch_tokens": meta["batch_tokens"],
            "hidden_dim": meta["hidden_dim"],
            "ffn_dim": meta["ffn_dim"],
        },
        "compile_pass_wall_ms": compile_pass_ms,
        "unoptimized_op_nodes": unopt_ops,
        "optimized_op_nodes": opt_ops,
        "op_node_reduction_pct": node_reduction_pct,
        "unoptimized_total_nodes": len(unopt_graph.nodes),
        "optimized_total_nodes": len(opt_graph.nodes),
        "unoptimized_peak_bytes": unopt_peak_bytes,
        "optimized_peak_bytes": opt_peak_bytes,
        "peak_memory_saved_bytes": unopt_peak_bytes - opt_peak_bytes,
        "peak_memory_saved_pct": mem_saved_pct,
        "unoptimized_latency_ms": unopt_ms,
        "optimized_latency_ms": opt_ms,
        "execution_speedup_x": speedup_x,
        "latency_reduction_pct": latency_saved_pct,
        "max_abs_diff": max_diff,
        "numerically_equivalent": bool(max_diff < 1e-9),
        "pass_trace": pass_trace,
        "unoptimized_ir": unopt_graph.dump_ir(),
        "optimized_ir": opt_graph.dump_ir(),
        "output_preview_row0": [round(v, 6) for v in opt_out[0][:6]],
    }
