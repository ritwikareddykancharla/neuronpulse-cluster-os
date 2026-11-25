"""Real Paged KV-Cache Block Allocator (vLLM-style Block Tables + CoW Prefix Sharing).

Compares two real memory allocators executed step-by-step over a multi-turn LLM serving workload:
  1. `ContiguousKVArena`: Reserves `max_seq_len` contiguous token slots upfront per request
     with zero cross-request prefix sharing.
  2. `PagedKVBlockAllocator`: Demand-pages fixed-size physical blocks (`block_size=16`) via
     per-request logical-to-physical block tables, deduplicates shared system-prompt prefixes
     with chained SHA-256 content hashes (`ref_count` >= 1), and enforces Copy-on-Write (CoW)
     physical block forking when a shared block is mutated in-place.
"""
from __future__ import annotations

import hashlib
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Concrete Transformer KV slot geometry:
# 24 layers * 2 (K + V) * 8 KV heads * 64 head_dim * 2 bytes (FP16/BF16) = 49,152 bytes (48 KiB) / token
DEFAULT_BYTES_PER_TOKEN_KV = 24 * 2 * 8 * 64 * 2  # 49,152 bytes


@dataclass
class PhysicalKVBlock:
    """Single physical KV-cache block holding up to `block_size` token KV vectors."""

    block_id: int
    block_size: int
    token_ids: List[int] = field(default_factory=list)
    ref_count: int = 0
    prefix_hash: Optional[str] = None
    cow_forked: bool = False

    @property
    def used_slots(self) -> int:
        return len(self.token_ids)

    @property
    def wasted_slots(self) -> int:
        return self.block_size - len(self.token_ids)

    @property
    def is_full(self) -> bool:
        return len(self.token_ids) == self.block_size


@dataclass
class WorkloadRequest:
    request_id: str
    seq_len: int
    shared_prefix_tokens: int
    unique_turn_tokens: int
    token_ids: List[int]


def _chain_block_hash(parent_hash: str, tokens: Tuple[int, ...]) -> str:
    h = hashlib.sha256()
    h.update(parent_hash.encode("ascii"))
    h.update(b":")
    h.update(",".join(str(t) for t in tokens).encode("ascii"))
    return h.hexdigest()[:20]


class ContiguousKVArena:
    """Baseline contiguous slab allocator reserving `max_seq_len` slots per admitted sequence."""

    def __init__(
        self,
        max_seq_len: int = 1024,
        capacity_slots: Optional[int] = None,
        bytes_per_token: int = DEFAULT_BYTES_PER_TOKEN_KV,
    ) -> None:
        self.max_seq_len = int(max_seq_len)
        self.capacity_slots = capacity_slots
        self.bytes_per_token = int(bytes_per_token)
        self.allocations: Dict[str, int] = {}  # request_id -> actual used tokens

    @property
    def reserved_slots(self) -> int:
        return len(self.allocations) * self.max_seq_len

    @property
    def used_slots(self) -> int:
        return sum(self.allocations.values())

    def try_allocate(self, req: WorkloadRequest) -> bool:
        if req.seq_len > self.max_seq_len:
            raise ValueError(f"Request {req.request_id} seq_len={req.seq_len} exceeds max_seq_len={self.max_seq_len}")
        needed = self.max_seq_len
        if self.capacity_slots is not None and (self.reserved_slots + needed) > self.capacity_slots:
            return False
        self.allocations[req.request_id] = req.seq_len
        return True

    def stats(self) -> Dict[str, Any]:
        reserved = self.reserved_slots
        used = self.used_slots
        wasted = reserved - used
        frag_pct = round((wasted / max(1, reserved)) * 100.0, 2)
        return {
            "admitted_requests": len(self.allocations),
            "reserved_token_slots": reserved,
            "used_token_slots": used,
            "wasted_fragmentation_slots": wasted,
            "allocated_bytes": reserved * self.bytes_per_token,
            "wasted_bytes": wasted * self.bytes_per_token,
            "internal_fragmentation_pct": frag_pct,
        }


class PagedKVBlockAllocator:
    """Demand-paged KV-cache block table manager with SHA-256 prefix sharing & Copy-on-Write."""

    def __init__(
        self,
        block_size: int = 16,
        max_blocks: Optional[int] = None,
        bytes_per_token: int = DEFAULT_BYTES_PER_TOKEN_KV,
    ) -> None:
        self.block_size = max(4, int(block_size))
        self.max_blocks = max_blocks
        self.bytes_per_token = int(bytes_per_token)
        self.blocks: Dict[int, PhysicalKVBlock] = {}
        self.free_ids: List[int] = (
            list(range(max_blocks - 1, -1, -1)) if max_blocks is not None else []
        )
        self._next_unbounded_id = 0
        self.prefix_index: Dict[str, int] = {}  # prefix_hash -> block_id
        self.block_tables: Dict[str, List[int]] = {}  # request_id -> [block_id, ...]
        self.logical_prefix_lookups = 0
        self.prefix_cache_hits = 0
        self.cow_forks_executed = 0

    def _alloc_raw_block(self) -> Optional[PhysicalKVBlock]:
        if self.max_blocks is not None:
            if not self.free_ids:
                return None
            bid = self.free_ids.pop()
        else:
            bid = self._next_unbounded_id
            self._next_unbounded_id += 1
        blk = PhysicalKVBlock(block_id=bid, block_size=self.block_size, token_ids=[], ref_count=1)
        self.blocks[bid] = blk
        return blk

    def _free_block(self, bid: int) -> None:
        blk = self.blocks.get(bid)
        if not blk:
            return
        blk.ref_count -= 1
        if blk.ref_count <= 0:
            if blk.prefix_hash and self.prefix_index.get(blk.prefix_hash) == bid:
                self.prefix_index.pop(blk.prefix_hash, None)
            self.blocks.pop(bid, None)
            if self.max_blocks is not None:
                self.free_ids.append(bid)

    def try_allocate_sequence(self, req: WorkloadRequest) -> bool:
        """Demand-page physical blocks for `req.token_ids`, reusing full prefix-cached blocks."""
        bs = self.block_size
        tokens = req.token_ids
        num_logical_blocks = math.ceil(len(tokens) / bs)

        parent_hash = "ROOT"
        acquired_table: List[int] = []
        prefix_chain_active = True
        local_lookups = 0
        local_hits = 0

        for b_idx in range(num_logical_blocks):
            chunk = tokens[b_idx * bs : (b_idx + 1) * bs]
            is_full_chunk = len(chunk) == bs

            if is_full_chunk and prefix_chain_active:
                local_lookups += 1
                blk_hash = _chain_block_hash(parent_hash, tuple(chunk))
                cached_bid = self.prefix_index.get(blk_hash)
                if cached_bid is not None and cached_bid in self.blocks:
                    self.blocks[cached_bid].ref_count += 1
                    acquired_table.append(cached_bid)
                    parent_hash = blk_hash
                    local_hits += 1
                    continue

                # Allocate new physical block and register in prefix table
                new_blk = self._alloc_raw_block()
                if new_blk is None:
                    for prev_bid in acquired_table:
                        self._free_block(prev_bid)
                    return False
                new_blk.token_ids = list(chunk)
                new_blk.prefix_hash = blk_hash
                self.prefix_index[blk_hash] = new_blk.block_id
                acquired_table.append(new_blk.block_id)
                parent_hash = blk_hash
            else:
                prefix_chain_active = False
                new_blk = self._alloc_raw_block()
                if new_blk is None:
                    for prev_bid in acquired_table:
                        self._free_block(prev_bid)
                    return False
                new_blk.token_ids = list(chunk)
                acquired_table.append(new_blk.block_id)

        self.block_tables[req.request_id] = acquired_table
        self.logical_prefix_lookups += local_lookups
        self.prefix_cache_hits += local_hits
        return True

    def mutate_token_cow(self, request_id: str, token_pos: int, new_token_id: int) -> Dict[str, Any]:
        """Mutate a token in `request_id`; if target physical block is shared (`ref_count > 1`),
        execute Copy-on-Write into a freshly allocated private physical block.
        """
        table = self.block_tables[request_id]
        blk_idx = token_pos // self.block_size
        slot_offset = token_pos % self.block_size
        old_bid = table[blk_idx]
        old_blk = self.blocks[old_bid]
        old_ref_before = old_blk.ref_count

        if old_blk.ref_count > 1:
            private_blk = self._alloc_raw_block()
            if private_blk is None:
                raise MemoryError("Out of physical KV blocks during Copy-on-Write fork")
            private_blk.token_ids = list(old_blk.token_ids)
            private_blk.token_ids[slot_offset] = new_token_id
            private_blk.cow_forked = True
            old_blk.ref_count -= 1
            table[blk_idx] = private_blk.block_id
            self.cow_forks_executed += 1
            return {
                "cow_triggered": True,
                "shared_block_id": old_bid,
                "shared_ref_before": old_ref_before,
                "shared_ref_after": old_blk.ref_count,
                "new_private_block_id": private_blk.block_id,
                "mutated_token_pos": token_pos,
            }

        old_blk.token_ids[slot_offset] = new_token_id
        return {
            "cow_triggered": False,
            "shared_block_id": old_bid,
            "shared_ref_before": 1,
            "shared_ref_after": 1,
            "new_private_block_id": old_bid,
            "mutated_token_pos": token_pos,
        }

    def materialize_sequence_tokens(self, request_id: str) -> List[int]:
        out: List[int] = []
        for bid in self.block_tables[request_id]:
            out.extend(self.blocks[bid].token_ids)
        return out

    def stats(self) -> Dict[str, Any]:
        num_phys_blocks = len(self.blocks)
        allocated_slots = num_phys_blocks * self.block_size
        used_phys_slots = sum(b.used_slots for b in self.blocks.values())
        wasted_tail_slots = sum(b.wasted_slots for b in self.blocks.values())
        logical_blocks_total = sum(len(t) for t in self.block_tables.values())
        shared_blocks = sum(1 for b in self.blocks.values() if b.ref_count > 1)
        max_ref = max((b.ref_count for b in self.blocks.values()), default=0)

        internal_frag_pct = round((wasted_tail_slots / max(1, allocated_slots)) * 100.0, 2)
        prefix_hit_rate_pct = round(
            (self.prefix_cache_hits / max(1, self.logical_prefix_lookups)) * 100.0,
            2,
        )
        return {
            "admitted_requests": len(self.block_tables),
            "block_size": self.block_size,
            "physical_blocks_allocated": num_phys_blocks,
            "logical_blocks_mapped": logical_blocks_total,
            "shared_physical_blocks": shared_blocks,
            "max_block_ref_count": max_ref,
            "allocated_token_slots": allocated_slots,
            "used_physical_slots": used_phys_slots,
            "wasted_tail_slots": wasted_tail_slots,
            "allocated_bytes": allocated_slots * self.bytes_per_token,
            "wasted_bytes": wasted_tail_slots * self.bytes_per_token,
            "internal_fragmentation_pct": internal_frag_pct,
            "cow_prefix_hit_rate_pct": prefix_hit_rate_pct,
            "prefix_cache_hits": self.prefix_cache_hits,
            "prefix_cache_lookups": self.logical_prefix_lookups,
            "cow_forks_executed": self.cow_forks_executed,
        }


def build_multiturn_workload(
    num_requests: int = 64,
    min_seq_len: int = 48,
    max_seq_len: int = 1024,
    system_prompt_tokens: int = 256,
    seed: int = 42,
) -> List[WorkloadRequest]:
    """Generate deterministic multi-turn LLM request batch (`seq_len` `48..1024`)
    sharing a common `system_prompt_tokens` prefix.
    """
    n_req = max(8, min(256, int(num_requests)))
    max_len = max(128, min(4096, int(max_seq_len)))
    min_len = max(16, min(max_len // 2, int(min_seq_len)))
    sys_len = max(0, min(max_len - 32, int(system_prompt_tokens)))

    rng = random.Random(seed + n_req * 13 + sys_len)
    system_prompt = [10_000 + i for i in range(sys_len)]

    # Build realistic request lengths spanning [min_seq_len .. max_seq_len]:
    # - First ~12.5% are short tool/router probes starting at exact `min_seq_len` (48)
    # - Remaining ~87.5% are multi-turn conversations carrying the full 256-token system prompt
    #   up to exact `max_seq_len` (1024)
    short_count = max(2, n_req // 8)
    lengths: List[int] = [min_len]
    for i in range(1, short_count):
        upper_short = max(min_len + 8, min(sys_len - 8, 240)) if sys_len > min_len + 16 else min_len + 64
        lengths.append(rng.randint(min_len + 4, upper_short))

    long_floor = max(min_len, sys_len + 24)
    for i in range(short_count, n_req - 1):
        # Realistic skewed multi-turn length distribution (more medium turns, tail to max_len)
        u = rng.random()
        span = max_len - long_floor
        sampled = long_floor + int((u ** 1.25) * span)
        lengths.append(min(max_len, max(long_floor, sampled)))
    lengths.append(max_len)

    # Shuffle arrival order deterministically so admission under a memory cap sees realistic mix
    rng.shuffle(lengths)

    requests: List[WorkloadRequest] = []
    for idx, seq_len in enumerate(lengths):
        shared_n = min(sys_len, max(0, seq_len - 16))
        unique_n = seq_len - shared_n
        prefix_part = system_prompt[:shared_n]
        unique_part = [50_000 + idx * 4096 + t for t in range(unique_n)]
        requests.append(
            WorkloadRequest(
                request_id=f"req-{idx:03d}",
                seq_len=seq_len,
                shared_prefix_tokens=shared_n,
                unique_turn_tokens=unique_n,
                token_ids=prefix_part + unique_part,
            )
        )
    return requests


def run_paged_kv_benchmark(
    num_requests: int = 64,
    block_size: int = 16,
    max_seq_len: int = 1024,
    system_prompt_tokens: int = 256,
    memory_cap_mb: int = 1280,
) -> Dict[str, Any]:
    """Execute Contiguous vs Paged+CoW KV allocators and measure exact slots, fragmentation & admission."""
    t0 = time.perf_counter()
    bs = max(4, min(128, int(block_size)))
    max_len = max(256, min(4096, int(max_seq_len)))
    sys_toks = max(0, min(max_len - 32, int(system_prompt_tokens)))
    n_req = max(8, min(256, int(num_requests)))
    cap_mb = max(128, min(16384, int(memory_cap_mb)))

    workload = build_multiturn_workload(
        num_requests=n_req,
        min_seq_len=48,
        max_seq_len=max_len,
        system_prompt_tokens=sys_toks,
        seed=42,
    )

    # 1. Full-batch memory efficiency comparison (all N requests active simultaneously)
    contig_full = ContiguousKVArena(max_seq_len=max_len, capacity_slots=None)
    paged_full = PagedKVBlockAllocator(block_size=bs, max_blocks=None)
    for req in workload:
        contig_full.try_allocate(req)
        paged_full.try_allocate_sequence(req)

    contig_stats = contig_full.stats()
    paged_stats = paged_full.stats()

    # 2. Verify real Copy-on-Write (CoW) isolation on a request sharing the system prompt
    cow_candidates = [r for r in workload if r.shared_prefix_tokens >= bs]
    cow_verification: Dict[str, Any] = {"verified": True, "cow_triggered": False, "peer_corruptions": 0}
    if len(cow_candidates) >= 2:
        target_req = cow_candidates[0]
        peer_Summaries = {
            r.request_id: list(paged_full.materialize_sequence_tokens(r.request_id))
            for r in cow_candidates[1:]
        }
        cow_event = paged_full.mutate_token_cow(target_req.request_id, token_pos=bs // 2, new_token_id=999_999)
        mutated_tokens = paged_full.materialize_sequence_tokens(target_req.request_id)
        corruptions = sum(
            1
            for rid, orig_tokens in peer_Summaries.items()
            if paged_full.materialize_sequence_tokens(rid) != orig_tokens
        )
        cow_verification = {
            **cow_event,
            "target_request_id": target_req.request_id,
            "mutated_slot_verified": mutated_tokens[bs // 2] == 999_999,
            "peer_corruptions": corruptions,
            "verified": bool(cow_event["cow_triggered"] and corruptions == 0 and mutated_tokens[bs // 2] == 999_999),
        }

    # 3. Fixed Memory Cap Concurrent Request Admission Test
    cap_bytes = cap_mb * 1024 * 1024
    cap_slots = cap_bytes // DEFAULT_BYTES_PER_TOKEN_KV
    cap_blocks = cap_slots // bs

    contig_capped = ContiguousKVArena(max_seq_len=max_len, capacity_slots=cap_slots)
    paged_capped = PagedKVBlockAllocator(block_size=bs, max_blocks=cap_blocks)

    for req in workload:
        contig_capped.try_allocate(req)
        paged_capped.try_allocate_sequence(req)

    contig_admitted = len(contig_capped.allocations)
    paged_admitted = len(paged_capped.block_tables)

    wall_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    bytes_saved = contig_stats["allocated_bytes"] - paged_stats["allocated_bytes"]
    memory_reduction_pct = round(
        (bytes_saved / max(1, contig_stats["allocated_bytes"])) * 100.0,
        2,
    )
    concurrency_gain_x = round(paged_admitted / max(1, contig_admitted), 2)

    seq_lens = [r.seq_len for r in workload]
    sample_block_tables = [
        {
            "request_id": r.request_id,
            "seq_len": r.seq_len,
            "shared_prefix_tokens": r.shared_prefix_tokens,
            "logical_blocks": len(paged_full.block_tables[r.request_id]),
            "physical_block_ids_head": paged_full.block_tables[r.request_id][:6],
        }
        for r in workload[:6]
    ]

    return {
        "workload": {
            "num_requests": len(workload),
            "min_seq_len": min(seq_lens),
            "max_seq_len": max(seq_lens),
            "avg_seq_len": round(sum(seq_lens) / len(seq_lens), 1),
            "system_prompt_tokens": sys_toks,
            "max_seq_len_reservation": max_len,
            "block_size": bs,
            "bytes_per_token_kv": DEFAULT_BYTES_PER_TOKEN_KV,
        },
        "contiguous_baseline": contig_stats,
        "paged_cow_allocator": paged_stats,
        "comparison": {
            "memory_saved_bytes": bytes_saved,
            "memory_reduction_pct": memory_reduction_pct,
            "fragmentation_drop_pts": round(
                contig_stats["internal_fragmentation_pct"] - paged_stats["internal_fragmentation_pct"],
                2,
            ),
            "memory_cap_mb": cap_mb,
            "memory_cap_slots": cap_slots,
            "contiguous_max_concurrent": contig_admitted,
            "paged_max_concurrent": paged_admitted,
            "concurrency_gain_x": concurrency_gain_x,
        },
        "cow_isolation_check": cow_verification,
        "sample_block_tables": sample_block_tables,
        "wall_clock_ms": wall_ms,
    }
