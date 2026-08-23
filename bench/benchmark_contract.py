#!/usr/bin/env python3
"""Phase 7 benchmark manifests, result validation, and analysis.

This module deliberately does not run a load generator.  It describes exact
workloads and validates/imports measurements produced by a real client/server
run; missing or rejected measurements stay unresolved.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

SCHEMA = "qwen38.phase7"
VERSION = 1
CONTEXT_LIMIT = 262_144
PROMPT_PATHS = tuple(f"prompt-{i}" for i in range(1, 6))
DECODE_CONTEXTS = (8_192, 32_768, 100_000, 128_000)
PREFILL_CONTEXTS = (8_192, 32_768, 65_536, 100_000, 128_000, 200_000, 261_120)
CONCURRENCIES = (1, 2, 4)
ENGINE_OUTPUTS = (1_024, 4_096, 16_384, 32_768)
IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _shape(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_shape(item) for item in value]
    return type(value).__name__


def _cell(kind: str, input_tokens: int, output_tokens: int, concurrency: int = 1) -> dict[str, Any]:
    if input_tokens + output_tokens > CONTEXT_LIMIT:
        raise ValueError("input_tokens + output_tokens exceeds native context")
    return {
        "id": f"{kind}-{input_tokens}-{output_tokens}-c{concurrency}",
        "kind": kind, "input_tokens": input_tokens, "output_tokens": output_tokens,
        "concurrency": concurrency, "prompt_paths": list(PROMPT_PATHS),
        "seed": 0, "streaming": True, "forced_output_length": True,
        "cache_modes": ["cold", "hot"], "warmups": 1,
    }


def build_manifest() -> dict[str, Any]:
    """Return the deterministic minimum controlled panel (no measurements)."""
    cells = []
    for context in DECODE_CONTEXTS:
        for concurrency in CONCURRENCIES:
            for output in ENGINE_OUTPUTS:
                cells.append(_cell("decode", context, output, concurrency))
    for context in PREFILL_CONTEXTS:
        for output in ENGINE_OUTPUTS:
            if context + output <= CONTEXT_LIMIT:
                cells.append(_cell("prefill", context, output))
    cells.extend([
        {"id": "production-balanced-c4", "kind": "production", "requests": [{"input_tokens": 25_000, "output_tokens": 4_000}] * 4,
         "prompt_paths": list(PROMPT_PATHS), "seed": 0, "streaming": True, "cache_modes": ["cold", "hot"], "warmups": 1},
        {"id": "production-asymmetric-c4", "kind": "production", "requests": [{"input_tokens": 100_000, "output_tokens": 16_000}] + [{"input_tokens": 8_000, "output_tokens": 4_000}] * 3,
         "prompt_paths": list(PROMPT_PATHS), "seed": 0, "streaming": True, "cache_modes": ["cold", "hot"], "warmups": 1},
    ])
    return {
        "schema": SCHEMA, "version": VERSION, "context_limit": CONTEXT_LIMIT,
        "purpose": "hardware-independent workload contract; measurements are external",
        "execution": {"unchanged_process_measured_runs": True, "warmup_runs": 1,
                       "alternate_process_blocks_for_close_finalists": True,
                       "engine_suite": {"sampling": "greedy", "fixed_seed": True,
                                        "streaming": True, "forced_lengths": True},
                       "natural_suite": {"status": "unresolved_until_corpus_pinned", "sampling": "model_recommended", "stopping": "natural"}},
        "required_identities": ["model_snapshot", "image_digest", "source_revision",
                                 "dependency_lock", "hardware_identity"],
        "required_run_metadata": ["server_args", "resolved_capacity", "raw_run"],
        "required_metrics": ["ttft_ms", "itl_ms", "max_itl_ms", "prompt_tokens_per_second",
                              "output_tokens_per_second", "request_latency_ms",
                              "total_vram_bytes", "free_vram_bytes", "gpu_utilization_pct", "pcie_bytes",
                              "power_w", "temperature_c", "energy_j"],
        "cells": cells,
    }


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    errors = []
    if not isinstance(manifest, dict):
        return ["manifest must be an object"]
    canonical = build_manifest()
    if manifest.get("schema") != SCHEMA or manifest.get("version") != VERSION:
        errors.append("unsupported schema/version")
    if manifest.get("context_limit") != CONTEXT_LIMIT:
        errors.append("context_limit must be 262144")
    for key in ("purpose", "execution", "required_identities", "required_run_metadata", "required_metrics"):
        if manifest.get(key) != canonical[key]:
            errors.append(f"{key} does not match the execution contract")
    cells = manifest.get("cells")
    if not isinstance(cells, list) or not cells:
        return errors + ["cells must be a non-empty list"]
    expected = {c["id"]: c for c in canonical["cells"]}
    seen = set()
    for i, cell in enumerate(cells):
        prefix = f"cells[{i}]"
        if not isinstance(cell, dict):
            errors.append(f"{prefix} must be an object")
            continue
        cell_id = cell.get("id")
        if not isinstance(cell_id, str):
            errors.append(f"{prefix} id must be a string")
            continue
        if cell_id in seen: errors.append(f"{prefix} duplicate id")
        seen.add(cell_id)
        expected_cell = expected.get(cell_id)
        if expected_cell is None:
            errors.append(f"{prefix} unexpected panel cell id")
            continue
        if cell != expected_cell:
            errors.append(f"{prefix} does not exactly match canonical cell {cell_id}")
        if cell.get("kind") == "production":
            requests = cell.get("requests")
            if not isinstance(requests, list) or not requests:
                errors.append(f"{prefix} production requests must be a non-empty list")
            else:
                for j, request in enumerate(requests):
                    if (not isinstance(request, dict) or type(request.get("input_tokens")) is not int or
                            type(request.get("output_tokens")) is not int or request["input_tokens"] < 0 or
                            request["output_tokens"] < 0):
                        errors.append(f"{prefix}.requests[{j}] has invalid token shape")
                    elif request["input_tokens"] + request["output_tokens"] > CONTEXT_LIMIT:
                        errors.append(f"{prefix}.requests[{j}] exceeds context limit")
        elif type(cell.get("input_tokens")) is int and type(cell.get("output_tokens")) is int:
            if cell["input_tokens"] + cell["output_tokens"] > CONTEXT_LIMIT:
                errors.append(f"{prefix} exceeds context limit")
    if set(seen) != set(expected): errors.append("minimum panel is incomplete")
    return errors


def validate_result(result: dict[str, Any], required_metrics: tuple[str, ...] | None = None) -> list[str]:
    """Validate one run/cell result; errors are explicit rejection reasons."""
    required = required_metrics or tuple(build_manifest()["required_metrics"])
    errors = []
    if not isinstance(result, dict):
        return ["result must be an object"]
    for key in ("schema", "run_id", "cell_id", "model_snapshot", "image_digest",
                "source_revision", "dependency_lock", "hardware_identity", "timestamps",
                "process", "server_args", "resolved_capacity", "raw_run", "cache_state", "occupancy", "interval", "metrics"):
        if key not in result:
            errors.append(f"missing {key}")
    for key in ("request_errors", "restarts", "ooms", "malformed_responses", "clamped"):
        if key not in result:
            errors.append(f"missing {key}")
    if result.get("schema") != SCHEMA:
        errors.append("unsupported result schema")
    if result.get("cell_id") not in {c["id"] for c in build_manifest()["cells"]}:
        errors.append("unknown cell_id")
    if not isinstance(result.get("image_digest"), str) or not IMAGE_DIGEST.fullmatch(result["image_digest"]):
        errors.append("malformed image digest")
    for key in ("model_snapshot", "source_revision", "dependency_lock", "hardware_identity"):
        value = result.get(key)
        if not isinstance(value, str) or not value.strip() or "UNRESOLVED" in value.upper():
            errors.append(f"invalid identity: {key}")
    if (not isinstance(result.get("server_args"), list) or not result["server_args"] or
            not all(isinstance(arg, str) for arg in result["server_args"])):
        errors.append("server_args must be non-empty")
    if not isinstance(result.get("raw_run"), str) or not result["raw_run"].strip(): errors.append("raw_run reference required")
    capacity = result.get("resolved_capacity")
    if (not isinstance(capacity, dict) or type(capacity.get("max_running_requests")) is not int or
            capacity["max_running_requests"] <= 0):
        errors.append("resolved max_running_requests must be positive")
    timestamps = result.get("timestamps")
    parsed_times = None
    if isinstance(timestamps, dict):
        try:
            parsed_times = (datetime.datetime.fromisoformat(timestamps["start"].replace("Z", "+00:00")),
                            datetime.datetime.fromisoformat(timestamps["end"].replace("Z", "+00:00")))
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
    try:
        ordered = parsed_times is not None and parsed_times[1] > parsed_times[0]
    except TypeError:
        ordered = False
    if not ordered:
        errors.append("timestamps must be parseable and ordered")
    if not isinstance(result.get("process"), dict) or result.get("process", {}).get("unchanged") is not True:
        errors.append("unchanged process metadata required")
    if result.get("cache_state") not in ("cold", "hot"): errors.append("invalid cache state")
    if (not isinstance(result.get("interval"), dict) or result["interval"].get("aligned") is not True or
            "queueing" not in result["interval"] or not isinstance(result["interval"].get("queueing"), bool)):
        errors.append("aligned interval required")
    metrics = result.get("metrics", {})
    if not isinstance(metrics, dict):
        errors.append("metrics must be an object")
        metrics = {}
    for metric in required:
        if metric not in metrics or metrics[metric] is None:
            errors.append(f"missing measurement: {metric}")
        elif not _is_number(metrics[metric]) or metrics[metric] < 0:
            errors.append(f"invalid measurement: {metric}")
    if _is_number(metrics.get("gpu_utilization_pct")) and not 0 <= metrics["gpu_utilization_pct"] <= 100:
        errors.append("invalid measurement: gpu_utilization_pct")
    if (_is_number(metrics.get("total_vram_bytes")) and metrics["total_vram_bytes"] <= 0):
        errors.append("invalid measurement: total_vram_bytes")
    if (_is_number(metrics.get("total_vram_bytes")) and _is_number(metrics.get("free_vram_bytes")) and
            metrics["free_vram_bytes"] > metrics["total_vram_bytes"]):
        errors.append("free VRAM exceeds total VRAM")
    for key, label in (("request_errors", "request errors present"), ("restarts", "process restart present"), ("ooms", "OOM present"), ("malformed_responses", "malformed response present")):
        value = result.get(key, 0)
        if type(value) is not int or value < 0 or value != 0:
            errors.append(label)
    occupancy = result.get("occupancy", {})
    cell = next((c for c in build_manifest()["cells"] if c["id"] == result.get("cell_id")), None)
    expected_occupancy = len(cell["requests"]) if cell and cell["kind"] == "production" else (cell["concurrency"] if cell else None)
    if (not isinstance(occupancy, dict) or type(occupancy.get("expected")) is not int or
            type(occupancy.get("observed")) is not int or occupancy.get("expected") != expected_occupancy or
            occupancy.get("observed") != occupancy.get("expected")):
        errors.append("wrong occupancy")
    if result.get("interval", {}).get("aligned") is False:
        errors.append("client/server intervals are not aligned")
    if result.get("interval", {}).get("queueing") is True:
        errors.append("queueing interval rejected")
    if not isinstance(result.get("clamped", False), bool):
        errors.append("invalid clamped flag")
    elif result.get("clamped", False):
        errors.append("request was silently clamped")
    return errors


def _numbers(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"status": "unresolved", "n": 0, "reason": "no accepted runs"}
    out: dict[str, Any] = {"status": "resolved", "n": len(values), "runs": values,
                           "median": statistics.median(values), "min": min(values), "max": max(values)}
    if len(values) >= 2:
        out["sample_stdev"] = statistics.stdev(values)
        out["cv"] = out["sample_stdev"] / statistics.mean(values) if statistics.mean(values) else None
    else:
        out["sample_stdev"] = None; out["cv"] = None
    for q, name in ((0.5, "p50"), (0.95, "p95"), (0.99, "p99")):
        if len(values) >= (2 if q == 0.5 else (20 if q == 0.95 else 100)):
            out[name] = statistics.quantiles(values, n=100, method="inclusive")[int(q * 100) - 1]
        else:
            out[name] = None
    return out


def analyze_cell(results: list[dict[str, Any]]) -> dict[str, Any]:
    accepted, rejected = [], []
    for result in results:
        reasons = validate_result(result)
        (rejected if reasons else accepted).append((result, reasons))
    accepted_ids = {r.get("cell_id") for r, _ in accepted}
    accepted_shapes = {json.dumps(_shape(r.get("process")), sort_keys=True) for r, _ in accepted}
    if len(accepted_ids) > 1 or len(accepted_shapes) > 1:
        reason = "mixed cell IDs or process shapes"
        rejected.extend((r, [reason]) for r, _ in accepted)
        accepted = []
    metrics = {}
    for metric in build_manifest()["required_metrics"]:
        metrics[metric] = _numbers([float(r["metrics"][metric]) for r, _ in accepted if r.get("metrics", {}).get(metric) is not None])
    return {"status": "resolved" if accepted else "unresolved", "accepted_runs": len(accepted),
            "rejected_runs": [{"run_id": r.get("run_id"), "reasons": reasons} for r, reasons in rejected],
            "metrics": metrics}


def matched_comparison(left: list[float], right: list[float]) -> dict[str, Any]:
    if len(left) != len(right) or not left:
        return {"status": "unresolved", "reason": "matched samples required"}
    deltas = [b - a for a, b in zip(left, right)]
    ratios = [((b - a) / a * 100) if a else (0.0 if b == 0 else None) for a, b in zip(left, right)]
    return {"status": "resolved", "n": len(left), "absolute_deltas": deltas,
            "percentage_deltas": ratios, "median_absolute_delta": statistics.median(deltas),
            "median_percentage_delta": statistics.median(x for x in ratios if x is not None) if any(x is not None for x in ratios) else None}


def evaluate_gates(summary: dict[str, Any]) -> dict[str, Any]:
    checks = {}
    def check(name: str, value: bool | None): checks[name] = "pass" if value is True else ("fail" if value is False else "unresolved")
    free = summary.get("free_vram_fraction")
    if free is None and summary.get("total_vram_bytes") and summary.get("free_vram_bytes") is not None:
        total, available = summary["total_vram_bytes"], summary["free_vram_bytes"]
        free = available / total if total > 0 else None
    check("free_vram_5_percent", None if free is None else free >= 0.05)
    max_itl = summary.get("max_itl_ms")
    check("itl_gap_under_1_second", None if max_itl is None else max_itl <= 1000)
    mixed, isolated = summary.get("mixed_itl_p99_ms"), summary.get("isolated_itl_p99_ms")
    check("mixed_p99_at_most_2x", None if mixed is None or isolated in (None, 0) else mixed <= 2 * isolated)
    for key in ("request_errors", "restarts", "ooms", "malformed_responses", "clamped"):
        value = summary.get(key)
        check(f"zero_{key}", None if value is None else value == 0 or value is False)
    checks["overall"] = "fail" if "fail" in checks.values() else ("unresolved" if "unresolved" in checks.values() else "pass")
    return checks


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("manifest"); gen.add_argument("output", type=Path)
    val = sub.add_parser("validate"); val.add_argument("input", type=Path)
    args = parser.parse_args(argv)
    if args.command == "manifest":
        data = build_manifest(); errors = validate_manifest(data)
        if errors: raise SystemExit("invalid generated manifest: " + "; ".join(errors))
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(data, indent=2) + "\n")
    else:
        data = json.loads(args.input.read_text()); errors = validate_manifest(data)
        print(json.dumps({"valid": not errors, "errors": errors}, indent=2)); raise SystemExit(bool(errors))


if __name__ == "__main__": main()
