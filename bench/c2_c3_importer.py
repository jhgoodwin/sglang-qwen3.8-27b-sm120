#!/usr/bin/env python3
"""Fail-closed importer for C2/C3 concurrent campaign raw runs.

The imported document always embeds the raw run, including failed and partial
requests.  ``accepted`` means that every required observation is supported;
missing evidence is never synthesized from client timing or response headers.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
import re
import sys
from typing import Any

# When invoked as ``python3 bench/c2_c3_importer.py``, Python puts ``bench/``
# on sys.path rather than the repository root.  Keep the package import
# explicit and make the documented file-path entrypoint work as well as
# ``python3 -m bench.c2_c3_importer``.
if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bench.c2_c3_runner import SCHEMA as RAW_SCHEMA

SCHEMA = "qwen38.c2-c3-import"
VERSION = 1
SCHEDULER_SCHEMA = "qwen38.server-scheduler-event"
PLACEHOLDERS = ("unresolved", "unknown", "placeholder", "todo", "tbd")
IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
REVISION = re.compile(r"^[0-9a-f]{40}$")
REPO = pathlib.Path(__file__).resolve().parents[1]


def _locked_image_digest() -> str:
    lock = json.loads((REPO / "source.lock.json").read_text())
    return lock["runtime_variants"]["c2-c3-evidence-overlay"]["image_digest"]


EXPECTED_IDENTITIES = {
    # Read from the evidence-overlay lock, rather than pinning its parent image.
    # A rebuilt overlay becomes eligible only after the supervisor updates the lock.
    "image_digest": _locked_image_digest(),
    "source_revision": "5f55db35e926d50676f75b812640ea2410b0fe0e",
    "model_snapshot": "/data/models/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/319f741cce68d7914884900c138a1fbb70a42f30",
    "draft_model_snapshot": "/data/models/models--incoai--Qwen3.8-27B-DFlash2/snapshots/dedf8df68adfb1afeaf7b7480c0a0243108177b4",
    "recipe_identity": "source.lock.json#/runtime_variants/current-cookbook-qwen38-27b",
}
PROFILE_RUNTIME = {
    "c2": {"max_running_requests": 2, "max_mamba_cache_size": 8, "planned_port": 11447},
    "c3": {"max_running_requests": 3, "max_mamba_cache_size": 12, "planned_port": 11448},
}
REQUIRED_SERVER_ARGS = {
    "context_length": 262144,
    "tp_size": 1,
    "kv_cache_dtype": "fp8_e4m3",
    "attention_backend": "flashinfer",
    "chunked_prefill_size": 2048,
    "mamba_ssm_dtype": "float32",
    "mem_fraction_static": 0.85,
    "mamba_radix_cache_strategy": "extra_buffer_lazy",
    "speculative_algorithm": "DFLASH",
    "speculative_num_draft_tokens": 8,
}
EVIDENCE_SOURCES = {"server_info", "server_log", "runtime_introspection"}


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _time(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def _timestamp(value: Any) -> float | None:
    return _time(value.get("utc")) if isinstance(value, dict) else None


def _placeholder(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        return not lowered or any(word in lowered for word in PLACEHOLDERS)
    if isinstance(value, dict):
        return not value or any(_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return not value or any(_placeholder(item) for item in value)
    return value is None


def _command_flags(command: Any) -> tuple[dict[str, str | bool], list[str]]:
    """Parse the independently observed container command, fail-closing ambiguity."""
    if not isinstance(command, list) or not command or not all(isinstance(x, str) and x for x in command):
        return {}, ["launch provenance container_command is missing or malformed"]
    values: dict[str, str | bool] = {}
    errors: list[str] = []
    index = 0
    while index < len(command):
        item = command[index]
        if not item.startswith("--"):
            index += 1
            continue
        flag, separator, inline = item.partition("=")
        if separator:
            value: str | bool = inline
        elif index + 1 < len(command) and not command[index + 1].startswith("--"):
            value = command[index + 1]
            index += 1
        else:
            value = True
        if flag in values:
            errors.append(f"duplicate launch argument: {flag}")
        else:
            values[flag] = value
        index += 1
    return values, errors


def _positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _validate_memory_pools(value: Any, expected_slots: int) -> list[str]:
    if not isinstance(value, dict):
        return ["memory_pools evidence is missing or malformed"]
    errors: list[str] = []
    if value.get("source") not in EVIDENCE_SOURCES:
        errors.append("memory_pools source is missing, placeholder, or unsupported")
    expected = {
        "kv_cache": ("fp8_e4m3", None),
        "mamba_state_cache": ("float32", expected_slots),
        "dflash_intermediate": (None, 8),
    }
    for name, (dtype, slots) in expected.items():
        pool = value.get(name)
        if not isinstance(pool, dict) or not _positive_int(pool.get("bytes")):
            errors.append(f"memory_pools.{name} requires positive measured bytes")
            continue
        if dtype is not None and pool.get("dtype") != dtype:
            errors.append(f"memory_pools.{name} has wrong dtype")
        slot_key = "slots" if name == "mamba_state_cache" else "states"
        if slots is not None and pool.get(slot_key) != slots:
            errors.append(f"memory_pools.{name} has wrong {slot_key}")
    return errors


def _validate_cuda_graphs(value: Any, concurrency: int) -> list[str]:
    if not isinstance(value, dict):
        return ["cuda_graphs evidence is missing or malformed"]
    errors: list[str] = []
    if value.get("source") not in EVIDENCE_SOURCES:
        errors.append("cuda_graphs source is missing, placeholder, or unsupported")
    if value.get("enabled") is not True:
        errors.append("CUDA graphs must be explicitly observed enabled")
    batches = value.get("captured_batch_sizes")
    if (not isinstance(batches, list) or not batches or
            not all(_positive_int(item) for item in batches) or len(set(batches)) != len(batches)):
        errors.append("CUDA graph batch sizes are missing or malformed")
    elif 1 not in batches or concurrency not in batches:
        errors.append("CUDA graphs do not cover batch 1 and profile concurrency")
    if not _positive_int(value.get("memory_bytes")):
        errors.append("CUDA graph measured memory_bytes must be positive")
    return errors


def _usage_counts(request: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    usage = request.get("usage")
    if not isinstance(usage, dict):
        return {"prompt_tokens": None, "completion_tokens": None,
                "reasoning_tokens": None, "emitted_reasoning_tokens": None,
                "visible_tokens": None}, ["missing final server usage"]
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    details = usage.get("completion_tokens_details", {})
    reasoning = usage.get("reasoning_tokens")
    if reasoning is None and isinstance(details, dict):
        reasoning = details.get("reasoning_tokens", 0)
    if reasoning is None:
        reasoning = 0
    for name, value in (("prompt_tokens", prompt), ("completion_tokens", completion),
                        ("reasoning_tokens", reasoning)):
        if type(value) is not int or value < 0:
            errors.append(f"invalid server usage {name}")
    # SGLang's usage reasoning_tokens is a re-tokenization of the rendered
    # reasoning text.  It is not necessarily a subset of generated model-token
    # IDs and can therefore exceed completion_tokens.  Classify the instrumented
    # emitted IDs by their SSE delta channel instead of subtracting unlike units.
    visible = emitted_reasoning = 0
    for event in request.get("events", []):
        parsed = event.get("parsed") if isinstance(event, dict) else None
        if not isinstance(parsed, dict):
            continue
        ids = parsed.get("token_ids")
        if not isinstance(ids, list) or not ids:
            continue
        content = reasoning_content = ""
        choices = parsed.get("choices", [])
        if isinstance(choices, list):
            for choice in choices:
                delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
                if isinstance(delta, dict):
                    content += delta.get("content", "") if isinstance(delta.get("content", ""), str) else ""
                    reasoning_content += (delta.get("reasoning_content", "")
                                          if isinstance(delta.get("reasoning_content", ""), str) else "")
        if content and reasoning_content:
            errors.append("cannot classify emitted tokens from mixed content/reasoning delta")
        elif content:
            visible += len(ids)
        elif reasoning_content:
            emitted_reasoning += len(ids)
    if type(completion) is int and visible + emitted_reasoning != completion:
        errors.append("emitted content/reasoning token count differs from completion usage")
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "reasoning_tokens": reasoning,
            "reasoning_tokens_basis": "server_usage_rendered_text_tokenization",
            "emitted_reasoning_tokens": emitted_reasoning,
            "visible_tokens": visible,
            "visible_tokens_basis": "server_token_ids_classified_by_sse_delta_channel"}, errors


def _token_itl(request: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Accept only explicit, one-timestamp-per-token server evidence."""
    timestamps: list[float] = []
    token_count = 0
    seen_payload = False
    malformed = False
    for event in request.get("events", []):
        parsed = event.get("parsed") if isinstance(event, dict) else None
        content = reasoning = ""
        if isinstance(parsed, dict):
            for choice in parsed.get("choices", []) if isinstance(parsed.get("choices", []), list) else []:
                delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
                if isinstance(delta, dict):
                    content += delta.get("content", "") if isinstance(delta.get("content", ""), str) else ""
                    reasoning += delta.get("reasoning_content", "") if isinstance(delta.get("reasoning_content", ""), str) else ""
        if not (content or reasoning):
            continue
        seen_payload = True
        ids = parsed.get("token_ids") if isinstance(parsed, dict) else None
        times = parsed.get("token_timestamps_s") if isinstance(parsed, dict) else None
        if (not isinstance(ids, list) or not ids or not isinstance(times, list) or len(ids) != len(times)
                or not all(type(token) is int for token in ids)
                or not all(_number(value) for value in times)):
            malformed = True
            continue
        token_count += len(ids)
        timestamps.extend(float(value) for value in times)
    if not seen_payload or malformed or token_count == 0 or len(timestamps) != token_count:
        reason = "server did not provide one timestamp per emitted token"
        return {"status": "unavailable", "basis": reason, "itl_ms": None,
                "max_itl_ms": None, "token_timestamps_s": []}, ["unsupported token-level ITL evidence"]
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        return {"status": "unavailable", "basis": "token timestamps are not strictly ordered",
                "itl_ms": None, "max_itl_ms": None, "token_timestamps_s": timestamps}, ["malformed token timestamps"]
    gaps = [(right - left) * 1000 for left, right in zip(timestamps, timestamps[1:])]
    return {"status": "available", "basis": "server token_ids paired one-to-one with server token_timestamps_s",
            "itl_ms": (sum(gaps) / len(gaps)) if gaps else 0.0,
            "max_itl_ms": max(gaps) if gaps else 0.0, "token_timestamps_s": timestamps}, []


def _request_metrics(request: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    counts, count_errors = _usage_counts(request); errors.extend(count_errors)
    itl, itl_errors = _token_itl(request); errors.extend(itl_errors)
    times = request.get("timestamps", {})
    submission, first, completion = (_timestamp(times.get(name)) for name in ("submission", "first_token", "completion"))
    if None in (submission, first, completion) or not submission <= first <= completion:
        errors.append("missing or unordered request timing")
        ttft = duration = after_first = None
    else:
        ttft = first - submission
        duration = completion - submission
        after_first = completion - first
    completion_tokens = counts["completion_tokens"]
    metrics = {**counts, "ttft_s": ttft, "wall_duration_s": duration,
        "completion_tok_s_end_to_end": (completion_tokens / duration
            if type(completion_tokens) is int and duration and duration > 0 else None),
        "completion_tok_s_after_first": ((completion_tokens - 1) / after_first
            if type(completion_tokens) is int and completion_tokens > 1 and after_first and after_first > 0 else None),
        "itl": itl}
    forced = request.get("forced_output") is True
    exact = type(completion_tokens) is int and completion_tokens == request.get("requested_output_tokens")
    expected_finish = request.get("finish_reason") == "length"
    metrics["forced_output_valid"] = forced and exact and expected_finish
    if forced and not metrics["forced_output_valid"]:
        errors.append("forced output requires exact completion count and length finish")
    return metrics, errors


def _server_events(raw: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    events: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, wrapper in enumerate(raw.get("scheduler_evidence", [])):
        event = wrapper.get("event") if isinstance(wrapper, dict) else None
        if not isinstance(event, dict) or "malformed_line" in event:
            errors.append(f"malformed scheduler evidence at index {index}")
            continue
        if event.get("schema") != SCHEDULER_SCHEMA or event.get("source") != "server_scheduler":
            errors.append(f"unsupported scheduler evidence at index {index}")
            continue
        if _time(event.get("timestamp")) is None:
            errors.append(f"malformed scheduler timestamp at index {index}")
        if event.get("event") not in ("queued", "admitted", "started", "completed", "failed"):
            errors.append(f"malformed scheduler event at index {index}")
        if type(event.get("running")) is not int or event["running"] < 0 or type(event.get("queued")) is not int or event["queued"] < 0:
            errors.append(f"malformed scheduler state at index {index}")
        if not isinstance(event.get("client_request_id"), str) or not event["client_request_id"]:
            errors.append(f"scheduler event lacks request correlation at index {index}")
        if not isinstance(event.get("server_process_id"), str) or not event["server_process_id"]:
            errors.append(f"scheduler event lacks process identity at index {index}")
        events.append(event)
    active: set[str] = set()
    queued: set[str] = set()
    for event in sorted(events, key=lambda item: _time(item.get("timestamp")) or float("inf")):
        rid, kind = event.get("client_request_id"), event.get("event")
        if kind == "queued":
            if rid in active or rid in queued:
                errors.append(f"contradictory scheduler queue transition for {rid}")
            queued.add(rid)
        elif kind == "admitted":
            if rid in active:
                errors.append(f"duplicate scheduler admission for {rid}")
            queued.discard(rid); active.add(rid)
        elif kind == "started":
            if rid not in active:
                errors.append(f"scheduler start precedes admission for {rid}")
        elif kind in ("completed", "failed"):
            if rid not in active:
                errors.append(f"server terminal event lacks active request {rid}")
            active.discard(rid); queued.discard(rid)
        if type(event.get("running")) is int and event["running"] != len(active):
            errors.append(f"contradictory server running count for {rid}")
        if type(event.get("queued")) is int and event["queued"] != len(queued):
            errors.append(f"contradictory server queue count for {rid}")
    if active or queued:
        errors.append("scheduler evidence ends with active or queued requests")
    return events, errors


def _validate_server(raw: dict[str, Any]) -> list[str]:
    server = raw.get("server")
    if not isinstance(server, dict):
        return ["missing server evidence"]
    errors: list[str] = []
    identities = server.get("identities")
    if not isinstance(identities, dict):
        errors.append("immutable server identities are missing")
    else:
        required = ("image_digest", "source_revision", "model_snapshot", "draft_model_snapshot",
                    "recipe_identity", "hardware_identity")
        for key in required:
            if key not in identities or _placeholder(identities.get(key)):
                errors.append(f"invalid server identity: {key}")
        if not IMAGE_DIGEST.fullmatch(identities.get("image_digest", "")):
            errors.append("malformed immutable image digest")
        if not REVISION.fullmatch(identities.get("source_revision", "")):
            errors.append("malformed SGLang source revision")
        for key, expected in EXPECTED_IDENTITIES.items():
            if identities.get(key) != expected:
                errors.append(f"server identity differs from locked campaign input: {key}")
    args = server.get("observed_server_args")
    if not isinstance(args, dict) or _placeholder(args):
        errors.append("observed_server_args must be a non-placeholder /server_info mapping")
    else:
        expected_profile = PROFILE_RUNTIME.get(raw.get("profile"))
        expected_args = dict(REQUIRED_SERVER_ARGS)
        if expected_profile is not None:
            expected_args.update({
                "max_running_requests": expected_profile["max_running_requests"],
                "max_mamba_cache_size": expected_profile["max_mamba_cache_size"],
            })
        for field, expected in expected_args.items():
            if args.get(field) != expected:
                errors.append(f"observed server field {field} must equal {expected}")
        for path_field in ("model_path", "speculative_draft_model_path"):
            if path_field not in args or _placeholder(args.get(path_field)):
                errors.append(f"observed server field {path_field} requires a resolved path")
        expected_revisions = {
            "model_path": "319f741cce68d7914884900c138a1fbb70a42f30",
            "speculative_draft_model_path": "dedf8df68adfb1afeaf7b7480c0a0243108177b4",
        }
        for path_field, revision in expected_revisions.items():
            if not args.get(path_field, "").endswith("/snapshots/" + revision):
                errors.append(f"observed server field {path_field} differs from locked snapshot")
    provenance = server.get("launch_provenance")
    if not isinstance(provenance, dict) or provenance.get("source") != "docker_inspect":
        errors.append("independent docker launch provenance is missing")
    else:
        if provenance.get("image_digest") != EXPECTED_IDENTITIES["image_digest"]:
            errors.append("launch provenance image differs from evidence-overlay lock")
        if provenance.get("source_revision") != EXPECTED_IDENTITIES["source_revision"]:
            errors.append("launch provenance source revision differs from lock")
        command_flags, command_errors = _command_flags(provenance.get("container_command"))
        errors.extend(command_errors)
        if "sglang.launch_server" not in provenance.get("container_command", []):
            errors.append("launch provenance is not the SGLang server command")
        if provenance.get("container_name") != "qwen3.8-27b-sglang":
            errors.append("launch provenance does not name the production service")
        launch_fields = {
            "--model-path": "model_path", "--speculative-draft-model-path": "speculative_draft_model_path",
            "--context-length": "context_length", "--tp-size": "tp_size",
            "--kv-cache-dtype": "kv_cache_dtype", "--attention-backend": "attention_backend",
            "--chunked-prefill-size": "chunked_prefill_size", "--mamba-ssm-dtype": "mamba_ssm_dtype",
            "--mem-fraction-static": "mem_fraction_static",
            "--mamba-radix-cache-strategy": "mamba_radix_cache_strategy",
            "--speculative-algorithm": "speculative_algorithm",
            "--speculative-num-draft-tokens": "speculative_num_draft_tokens",
            "--max-running-requests": "max_running_requests",
            "--max-mamba-cache-size": "max_mamba_cache_size",
        }
        for flag, field in launch_fields.items():
            expected = args.get(field) if isinstance(args, dict) else None
            if expected is None or command_flags.get(flag) != str(expected):
                errors.append(f"launch argument {flag} differs from observed server field {field}")
        if "--mamba-full-memory-ratio" in command_flags:
            errors.append("initial C2/C3 launch must omit --mamba-full-memory-ratio")
        if "--disable-cuda-graph" in command_flags:
            errors.append("initial C2/C3 launch must not disable CUDA graphs")
        if not isinstance(provenance.get("artifact_reference"), str) or _placeholder(provenance.get("artifact_reference")):
            errors.append("launch provenance artifact reference is missing")
    capacity = server.get("resolved_capacity")
    if (not isinstance(capacity, dict) or type(capacity.get("max_running_requests")) is not int
            or capacity["max_running_requests"] <= 0 or _placeholder(capacity)):
        errors.append("resolved capacity is missing or malformed")
    else:
        profile_runtime = PROFILE_RUNTIME.get(raw.get("profile"))
        expected_concurrency = profile_runtime["max_running_requests"] if profile_runtime else None
        if expected_concurrency is None or capacity["max_running_requests"] != expected_concurrency:
            errors.append("resolved concurrency does not match C2/C3 profile")
        if capacity.get("context_length") != 262144:
            errors.append("resolved native context must be 262144")
        if capacity.get("tp_size") != 1:
            errors.append("resolved tensor parallel size must be 1")
        if profile_runtime and capacity.get("max_mamba_cache_size") != profile_runtime["max_mamba_cache_size"]:
            errors.append("resolved Mamba cache size does not match C2/C3 profile")
    limits = server.get("campaign_request_limits")
    if not isinstance(limits, dict) or limits.get("max_output_tokens") != 131072:
        errors.append("campaign_request_limits.max_output_tokens must be 131072 metadata")
    launch = server.get("launch_metadata")
    if not isinstance(launch, dict):
        errors.append("launch_metadata is missing")
    else:
        profile_runtime = PROFILE_RUNTIME.get(raw.get("profile"))
        if profile_runtime and launch.get("planned_port") != profile_runtime["planned_port"]:
            errors.append("launch planned_port does not match C2/C3 profile")
        observed = launch.get("observed_endpoint")
        if not isinstance(observed, str) or not observed.startswith(("http://", "https://")):
            errors.append("launch observed_endpoint is missing")
    expected_slots = PROFILE_RUNTIME.get(raw.get("profile"), {}).get("max_mamba_cache_size", -1)
    errors.extend(_validate_memory_pools(server.get("memory_pools"), expected_slots))
    concurrency = PROFILE_RUNTIME.get(raw.get("profile"), {}).get("max_running_requests", -1)
    errors.extend(_validate_cuda_graphs(server.get("cuda_graphs"), concurrency))
    process = server.get("process")
    if not isinstance(process, dict) or _placeholder(process):
        errors.append("process/restart evidence is missing or placeholder")
    else:
        if type(process.get("restart_count")) is not int or process["restart_count"] < 0:
            errors.append("invalid restart count")
        elif process["restart_count"] != 0:
            errors.append("server restart observed during run")
        samples = process.get("samples")
        if not isinstance(samples, list) or len(samples) < 2:
            errors.append("continuous process evidence requires at least two samples")
        else:
            identities = {sample.get("server_process_id") for sample in samples if isinstance(sample, dict)}
            if None in identities or len(identities) != 1:
                errors.append("server process identity changed or is missing")
            stamps = [_time(sample.get("timestamp")) for sample in samples if isinstance(sample, dict)]
            interval = process.get("sampling_interval_s")
            if (not _number(interval) or interval <= 0 or None in stamps or
                    any(right <= left or right - left > max(.25, interval * 2.5)
                        for left, right in zip(stamps, stamps[1:]))):
                errors.append("process evidence is not continuous and ordered")
    return errors


def _validate_interval(raw: dict[str, Any], events: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    requests = raw.get("requests", [])
    arrivals = [_timestamp(request.get("timestamps", {}).get("arrival")) for request in requests]
    completions = [_timestamp(request.get("timestamps", {}).get("completion")) for request in requests]
    if not arrivals or None in arrivals or None in completions:
        return {}, ["request interval is missing"]
    start, end = min(arrivals), max(completions)
    spread = max(arrivals) - min(arrivals)
    if spread > .1:
        errors.append("concurrent arrivals are not aligned within 100 ms")
    telemetry = raw.get("gpu_telemetry")
    valid_gpu: list[dict[str, Any]] = []
    if not isinstance(telemetry, list) or len(telemetry) < 2:
        errors.append("continuous GPU telemetry requires at least two samples")
    else:
        for index, sample in enumerate(telemetry):
            stamp = _timestamp(sample)
            if (not isinstance(sample, dict) or stamp is None or sample.get("source") != "nvidia-smi"
                    or "error" in sample or any(not _number(sample.get(key)) for key in
                    ("total_vram_bytes", "free_vram_bytes", "gpu_utilization_pct", "power_w", "temperature_c"))):
                errors.append(f"malformed GPU telemetry sample at index {index}")
            else:
                valid_gpu.append(sample)
        if valid_gpu:
            stamps = [_timestamp(sample) for sample in valid_gpu]
            interval = raw.get("collector", {}).get("gpu_interval_s")
            if not _number(interval) or interval <= 0:
                errors.append("invalid GPU sampling interval")
            else:
                if stamps[0] > start or stamps[-1] < end:
                    errors.append("GPU telemetry does not cover the request interval")
                if any(right <= left or right - left > max(.25, interval * 2.5)
                       for left, right in zip(stamps, stamps[1:])):
                    errors.append("GPU telemetry has an interval gap")
            for sample in valid_gpu:
                if sample["total_vram_bytes"] <= 0 or not 0 <= sample["free_vram_bytes"] <= sample["total_vram_bytes"]:
                    errors.append("invalid GPU VRAM telemetry")
                if not 0 <= sample["gpu_utilization_pct"] <= 100:
                    errors.append("invalid GPU utilization telemetry")
    process_samples = raw.get("server", {}).get("process", {}).get("samples", [])
    process_stamps = [_time(sample.get("timestamp")) for sample in process_samples if isinstance(sample, dict)]
    if process_stamps and (None in process_stamps or process_stamps[0] > start or process_stamps[-1] < end):
        errors.append("process/restart evidence does not cover the request interval")
    process_ids = {sample.get("server_process_id") for sample in process_samples if isinstance(sample, dict)}
    scheduler_process_ids = {event.get("server_process_id") for event in events}
    if events and process_ids != scheduler_process_ids:
        errors.append("scheduler and process evidence identify different server processes")
    event_times = [_time(event.get("timestamp")) for event in events]
    if events and (None in event_times or min(event_times) < start - 1 or max(event_times) > end + 1):
        errors.append("scheduler evidence is misaligned with request interval")
    min_free = min((sample["free_vram_bytes"] / sample["total_vram_bytes"] for sample in valid_gpu), default=None)
    if min_free is None or min_free < .05:
        errors.append("five-percent free-VRAM gate failed or unresolved")
    return {"start": _utc(start), "end": _utc(end), "arrival_spread_s": spread,
            "minimum_free_vram_fraction": min_free}, errors


def _utc(epoch: float) -> str:
    return dt.datetime.fromtimestamp(epoch, dt.timezone.utc).isoformat()


def validate_and_import(raw: Any) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(raw, dict):
        return {"schema": SCHEMA, "version": VERSION, "accepted": False,
                "errors": ["raw run must be an object"], "raw_run": raw}
    if raw.get("schema") != RAW_SCHEMA or raw.get("version") != 1:
        errors.append("unsupported raw run schema/version")
    for key in ("run_id", "profile", "stage", "requests", "server", "scheduler_evidence", "gpu_telemetry"):
        if key not in raw:
            errors.append(f"missing {key}")
    requests = raw.get("requests")
    if not isinstance(requests, list) or not requests:
        requests = []
        errors.append("requests must be a non-empty list")
    if raw.get("request_count") != len(requests):
        errors.append("request_count does not match retained requests")
    ids = [request.get("client_request_id") for request in requests if isinstance(request, dict)]
    if len(ids) != len(requests) or any(not isinstance(rid, str) or not rid for rid in ids) or len(set(ids)) != len(ids):
        errors.append("request IDs must be unique and non-empty")
    errors.extend(_validate_server(raw))
    scheduler_events, scheduler_errors = _server_events(raw); errors.extend(scheduler_errors)
    interval, interval_errors = _validate_interval(raw, scheduler_events); errors.extend(interval_errors)
    server_ids = {event.get("client_request_id") for event in scheduler_events}
    if set(ids) != server_ids:
        errors.append("scheduler evidence does not correlate every retained request")
    by_request = {rid: [] for rid in ids}
    for event in scheduler_events:
        if event.get("client_request_id") in by_request:
            by_request[event["client_request_id"]].append(event)
    capacity = raw.get("server", {}).get("resolved_capacity", {}).get("max_running_requests")
    admissions = []
    for rid, rows in by_request.items():
        admitted = next((_time(row.get("timestamp")) for row in rows if row.get("event") == "admitted"), None)
        if admitted is not None:
            admissions.append((admitted, rid))
    admission_rank = {rid: index for index, (_, rid) in enumerate(sorted(admissions))}
    request_imports = []
    for request in requests:
        rid = request.get("client_request_id")
        request_errors: list[str] = []
        if not isinstance(request.get("raw_sse"), list) or not isinstance(request.get("events"), list):
            request_errors.append("raw SSE/event retention is missing")
        for timestamp_name in ("arrival", "submission", "headers", "first_event", "first_token", "completion"):
            value = request.get("timestamps", {}).get(timestamp_name)
            if timestamp_name == "headers" and request.get("failure_bucket") in ("timeout", "restart", "disconnect", "oom"):
                continue
            if _timestamp(value) is None:
                request_errors.append(f"missing {timestamp_name} timestamp")
        metrics, metric_errors = _request_metrics(request); request_errors.extend(metric_errors)
        event_names = [event.get("event") for event in by_request.get(rid, [])]
        if "admitted" not in event_names or "started" not in event_names:
            request_errors.append("missing genuine server admission/start evidence")
        event_times = {name: next((_time(event.get("timestamp")) for event in by_request.get(rid, [])
                                  if event.get("event") == name), None)
                       for name in ("queued", "admitted", "started", "completed", "failed")}
        arrival_time = _timestamp(request.get("timestamps", {}).get("arrival"))
        completion_time = _timestamp(request.get("timestamps", {}).get("completion"))
        terminal_server_time = event_times["completed"] if event_times["completed"] is not None else event_times["failed"]
        if (arrival_time is not None and completion_time is not None and event_times["admitted"] is not None
                and event_times["started"] is not None and not
                (arrival_time <= event_times["admitted"] <= event_times["started"] <= completion_time)):
            request_errors.append("server admission/start timing is not aligned with client interval")
        if terminal_server_time is not None and completion_time is not None and terminal_server_time > completion_time:
            request_errors.append("server terminal event follows client completion")
        if event_times["queued"] is not None and event_times["admitted"] is not None and event_times["queued"] > event_times["admitted"]:
            request_errors.append("server queue event follows admission")
        queue_delay = (event_times["admitted"] - arrival_time
                       if event_times["admitted"] is not None and arrival_time is not None else None)
        metrics["scheduler"] = {"admission_timestamp": _utc(event_times["admitted"]) if event_times["admitted"] is not None else None,
            "start_timestamp": _utc(event_times["started"]) if event_times["started"] is not None else None,
            "queue_delay_s": queue_delay,
            "wave": (admission_rank[rid] // capacity + 1
                     if rid in admission_rank and type(capacity) is int and capacity > 0 else None)}
        if request.get("failure_bucket") is not None or request.get("error") is not None:
            request_errors.append(f"request failed: {request.get('failure_bucket') or 'unclassified'}")
        if request.get("http_status") != 200:
            request_errors.append("request did not return HTTP 200")
        if request.get("finish_reason") not in ("stop", "length"):
            request_errors.append("request is incomplete")
        target = request.get("expected_prompt_tokens")
        if target is not None and metrics["prompt_tokens"] != target:
            request_errors.append("server prompt-token count differs from expected exact count")
        completion_count = metrics["completion_tokens"]
        if request.get("failure_bucket") is not None:
            outcome_bucket = request["failure_bucket"]
        elif request.get("http_status") != 200:
            outcome_bucket = "http_error"
        elif request.get("finish_reason") not in ("stop", "length"):
            outcome_bucket = "incomplete"
        elif (request.get("forced_output") is True and type(completion_count) is int and
              completion_count < request.get("requested_output_tokens", 0)):
            outcome_bucket = "clamped"
        else:
            outcome_bucket = "complete"
        request_imports.append({"client_request_id": rid, "accepted": not request_errors,
                                "errors": request_errors, "outcome_bucket": outcome_bucket, "metrics": metrics,
                                "server_scheduler_events": by_request.get(rid, []), "raw_request": request})
        errors.extend(f"request {rid}: {error}" for error in request_errors)
    max_running = max((event.get("running", -1) for event in scheduler_events), default=-1)
    max_queued = max((event.get("queued", -1) for event in scheduler_events), default=-1)
    expected_peak = min(len(requests), capacity) if type(capacity) is int else None
    if max_running != expected_peak:
        errors.append("observed server occupancy does not reach the expected exact occupancy")
    if len(requests) > (capacity if type(capacity) is int else len(requests)) and max_queued < 1:
        errors.append("excess simultaneous arrivals lack server queue evidence")
    completion_sum = sum(item["metrics"]["completion_tokens"] for item in request_imports
                         if type(item["metrics"]["completion_tokens"]) is int)
    interval_seconds = (_time(interval.get("end")) - _time(interval.get("start"))) if interval else None
    aggregate = {"completion_tokens": completion_sum, "wall_duration_s": interval_seconds,
                 "completion_tok_s": completion_sum / interval_seconds
                    if interval_seconds and interval_seconds > 0 else None,
                 "observed_max_running": max_running, "observed_max_queued": max_queued,
                 "observed_waves": max((item["metrics"]["scheduler"]["wave"] or 0
                                        for item in request_imports), default=0),
                 "expected_waves": (math.ceil(len(requests) / capacity)
                                    if type(capacity) is int and capacity > 0 else None)}
    strict = raw.get("boundary_proof")
    if strict is not None:
        expected = strict.get("expected_server_prompt_tokens") if isinstance(strict, dict) else None
        if type(expected) is not int or expected <= 0 or any(item["metrics"]["prompt_tokens"] != expected for item in request_imports):
            errors.append("optional exact-boundary proof gate is unresolved")
    return {"schema": SCHEMA, "version": VERSION, "run_id": raw.get("run_id"),
            "profile": raw.get("profile"), "stage": raw.get("stage"), "accepted": not errors,
            "errors": errors, "interval": interval, "aggregate_metrics": aggregate,
            "requests": request_imports, "raw_run": raw}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)
    imported = validate_and_import(json.loads(args.input.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(imported, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"accepted": imported["accepted"], "errors": imported["errors"]}, indent=2))
    return 0 if imported["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
