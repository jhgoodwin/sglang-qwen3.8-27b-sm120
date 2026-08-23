#!/usr/bin/env python3
"""Deterministic, predicted KV/GDN sizing worksheet for Phase 3A.

This is an accounting aid, not a VRAM admission test.  Constants are the
approximate decimal values recorded in TODO.md; measured fixed allocations and
headroom must be supplied separately by a runtime result.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

LENGTHS = (16_000, 32_000, 64_000, 100_000, 128_000, 200_000, 262_000)
CONCURRENCIES = (1, 2, 4)
STATE_BYTES = {"float32": 153.9 * 1_000_000, "bfloat16": 78.4 * 1_000_000}
KV_BYTES = {"FP8": 32.8 * 1_000, "BF16": 65.5 * 1_000}
STRATEGIES = {
    "extra_buffer_lazy": 4,
    "extra_buffer": 5,
    "no_buffer": 3,
    "disable_radix_cache": 1,
}
# Only DSpark's D is documented. Unknown methods remain explicitly unresolved.
SPECULATION_D = {"none": 0, "MTP": None, "DSpark": 8, "DFlash2": None}
PROFILES = {
    "ordinary_coding": {"input_tokens": 8_000, "output_tokens": 1_000},
    "long_repo_coding": {"input_tokens": 100_000, "output_tokens": 16_000},
    # Exact native combined budget: 245,760 input + 16,384 output.
    "near_native_context": {"input_tokens": 245_760, "output_tokens": 16_384},
}


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def balanced_ratio(strategy: str, state_dtype: str, kv_dtype: str,
                   average_total_request_tokens: int, speculation: str) -> Optional[float]:
    """Return the dimensionless ratio, or None where TODO leaves D unresolved."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy}")
    if state_dtype not in STATE_BYTES:
        raise ValueError(f"unknown state dtype: {state_dtype}")
    if kv_dtype not in KV_BYTES:
        raise ValueError(f"unknown KV dtype: {kv_dtype}")
    if speculation not in SPECULATION_D:
        raise ValueError(f"unknown speculation method: {speculation}")
    tokens = _positive_int("average_total_request_tokens", average_total_request_tokens)
    d = SPECULATION_D[speculation]
    if d is None:
        return None
    return (STRATEGIES[strategy] + d) * STATE_BYTES[state_dtype] / (tokens * KV_BYTES[kv_dtype])


def max_mamba_cache_size(strategy: str, concurrency: int) -> int:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy}")
    return _positive_int("concurrency", concurrency) * STRATEGIES[strategy]


def _units(bytes_value: float) -> dict[str, float]:
    return {
        "bytes": bytes_value,
        "kB": bytes_value / 1_000,
        "MB": bytes_value / 1_000**2,
        "GB": bytes_value / 1_000**3,
        "KiB": bytes_value / 1024,
        "MiB": bytes_value / 1024**2,
        "GiB": bytes_value / 1024**3,
    }


@dataclass(frozen=True)
class Row:
    average_total_request_tokens: int
    concurrency: int
    state_dtype: str
    kv_dtype: str
    strategy: str
    speculation: str
    S: int
    D: Optional[int]
    balanced_ratio: Optional[float]
    max_mamba_cache_size_slots: int
    predicted_state_pool_bytes_decimal: float
    predicted_kv_bytes_decimal: float


def worksheet() -> dict:
    rows = []
    for length in LENGTHS:
        for concurrency in CONCURRENCIES:
            for state_dtype in STATE_BYTES:
                for kv_dtype in KV_BYTES:
                    for strategy in STRATEGIES:
                        for speculation, d in SPECULATION_D.items():
                            rows.append(asdict(Row(
                                length, concurrency, state_dtype, kv_dtype, strategy,
                                speculation, STRATEGIES[strategy], d,
                                balanced_ratio(strategy, state_dtype, kv_dtype, length, speculation),
                                max_mamba_cache_size(strategy, concurrency),
                                concurrency * STRATEGIES[strategy] * STATE_BYTES[state_dtype],
                                concurrency * length * KV_BYTES[kv_dtype])))
    return {
        "schema": 1,
        "accounting": "predicted_dynamic_memory_only",
        "unit_semantics": {"constants": "decimal (1 kB=1,000 bytes; 1 MB=1,000,000 bytes)",
                            "binary": "IEC fields use KiB/MiB/GiB (1 KiB=1,024 bytes)",
                            "headroom": "not predicted; provide measured fixed memory and free VRAM"},
        "constants": {"state_bytes": {k: _units(v) for k, v in STATE_BYTES.items()},
                       "kv_bytes_per_token": {k: _units(v) for k, v in KV_BYTES.items()}},
        "formula": "(S + D) * state_bytes / (average_total_request_tokens * kv_bytes_per_token)",
        "profiles": PROFILES,
        "speculation_D": {k: (v if v is not None else "UNRESOLVED_PARAMETER") for k, v in SPECULATION_D.items()},
        "rows": rows,
    }


def markdown(data: dict) -> str:
    lines = ["# Phase 3A capacity worksheet", "", "Predicted KV/GDN dynamic accounting only; no cell claims to fit a GPU. Decimal units use kB/MB/GB; binary units use KiB/MiB/GiB.", "", "Formula: `(S + D) * state_bytes / (average_total_request_tokens * kv_bytes_per_token)`", "", "The L sweep uses approximate labels 16K through 262K; the near-native observed profile below is an exact 262,144-token combined budget.", "", "## Observed workload profiles", "", "| Profile | Observed input | Observed output | Observed total |", "|---|---:|---:|---:|"]
    for name, p in data["profiles"].items():
        lines.append(f"| {name} | {p['input_tokens']:,} | {p['output_tokens']:,} | {p['input_tokens'] + p['output_tokens']:,} tokens |")
    lines += ["", "Unresolved speculation methods (MTP and DFlash2) have no fabricated D; their ratio is `UNRESOLVED_PARAMETER`. `max_mamba_cache_size` excludes D by design. Predicted state/KV byte columns scale with concurrency; the ratio does not.", "", "## Worksheet rows", "", "| L tokens | C | SSM | KV | strategy | speculation | S | D | ratio | max state slots | state bytes (decimal) | KV bytes (decimal) |", "|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in data["rows"]:
        ratio = "UNRESOLVED" if r["balanced_ratio"] is None else f"{r['balanced_ratio']:.9g}"
        d = "UNRESOLVED" if r["D"] is None else str(r["D"])
        lines.append(f"| {r['average_total_request_tokens']:,} | {r['concurrency']} | {r['state_dtype']} | {r['kv_dtype']} | {r['strategy']} | {r['speculation']} | {r['S']} | {d} | {ratio} | {r['max_mamba_cache_size_slots']} | {r['predicted_state_pool_bytes_decimal']:.0f} | {r['predicted_kv_bytes_decimal']:.0f} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", dest="json_path", type=Path, default=Path("bench/results/capacity-worksheet.json"))
    parser.add_argument("--markdown", dest="markdown_path", type=Path, default=Path("bench/results/capacity-worksheet.md"))
    args = parser.parse_args(argv)
    data = worksheet()
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(json.dumps(data, indent=2) + "\n")
    args.markdown_path.write_text(markdown(data))
    print(f"wrote {len(data['rows'])} rows: {args.json_path} and {args.markdown_path}")


if __name__ == "__main__":
    main()
