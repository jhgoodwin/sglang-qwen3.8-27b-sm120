#!/usr/bin/env python3
"""Fail-closed two-request near-full-context BF16 KV probe.

This is an operational smoke probe, not a C2/C3 qualification importer.  It
tokenizes each exact chat message immediately before a barrier-released
request, retains complete SSE evidence, and validates the server's final
usage against the requested 250K + 6K budget.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bench.c2_c3_campaign import ExactPromptBuilder

PROMPT_TOKENS = 250_000
MAX_TOKENS = 6_000
CONCURRENCY = 2
DEFAULT_BASE_URL = "http://127.0.0.1:11436"
EXPECTED_MODEL_REVISION = "319f741cce68d7914884900c138a1fbb70a42f30"
EXPECTED_SOURCE_REVISION = "5f55db35e926d50676f75b812640ea2410b0fe0e"


def _stamp() -> dict[str, Any]:
    return {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "monotonic_s": time.monotonic()}


def _find(info: Any, names: tuple[str, ...]) -> Any:
    """Find a scalar in nested server_info, without guessing missing values."""
    if isinstance(info, dict):
        for name in names:
            if name in info:
                return info[name]
        for value in info.values():
            found = _find(value, names)
            if found is not None:
                return found
    elif isinstance(info, list):
        for value in info:
            found = _find(value, names)
            if found is not None:
                return found
    return None


def _all_values(info: Any, names: tuple[str, ...]) -> list[Any]:
    found: list[Any] = []
    if isinstance(info, dict):
        for key, value in info.items():
            if key in names: found.append(value)
            found.extend(_all_values(value, names))
    elif isinstance(info, list):
        for value in info: found.extend(_all_values(value, names))
    return found


def capacity_preflight(info: Any, prompt_tokens: int = PROMPT_TOKENS,
                       max_tokens: int = MAX_TOKENS, concurrency: int = CONCURRENCY,
                       expected_model_revision: str = EXPECTED_MODEL_REVISION,
                       expected_source_revision: str = EXPECTED_SOURCE_REVISION) -> dict[str, Any]:
    """Return canonical evidence or raise; absent/conflicting facts fail closed."""
    if prompt_tokens <= 0 or max_tokens <= 0 or concurrency != 2:
        raise ValueError("probe requires positive token values and concurrency exactly 2")
    if not isinstance(info, dict): raise ValueError("unavailable canonical server_info object")
    model_path, version = info.get("model_path"), info.get("version")
    # SGLang's reported dev version exposes the nine-character git suffix.
    expected_short = expected_source_revision[:9]
    if not isinstance(model_path, str) or not model_path.rstrip("/").endswith(expected_model_revision):
        raise ValueError(f"unavailable/conflicting model identity: {model_path!r}")
    if not isinstance(version, str) or expected_short not in version:
        raise ValueError(f"unavailable/conflicting runtime version: {version!r}")
    context, dtype = info.get("context_length"), info.get("kv_cache_dtype")
    running, capacity = info.get("max_running_requests"), info.get("max_total_num_tokens")
    if not isinstance(context, int): raise ValueError(f"unavailable canonical context_length: {context!r}")
    if dtype is None: raise ValueError("unavailable canonical KV dtype")
    if not isinstance(running, int): raise ValueError(f"unavailable canonical max_running_requests: {running!r}")
    if not isinstance(capacity, int): raise ValueError(f"unavailable canonical token capacity: {capacity!r}")
    # Internal state is useful corroboration when present, but cannot override
    # canonical top-level fields. Any disagreement is a fail-closed conflict.
    states = info.get("internal_states")
    if isinstance(states, list):
        for state in states:
            if not isinstance(state, dict): continue
            for key, value in (("context_length", context), ("max_running_requests", running),
                               ("max_total_num_tokens", capacity)):
                if key in state and state[key] != value:
                    raise ValueError(f"conflicting internal_states {key}: {state[key]!r} != {value!r}")
    required = concurrency * (prompt_tokens + max_tokens)
    if not isinstance(context, int) or context < prompt_tokens + max_tokens:
        raise ValueError(f"unavailable/insufficient context_length evidence: {context!r}")
    if str(dtype).lower() not in {"bf16", "bfloat16", "torch.bfloat16"}:
        raise ValueError(f"BF16 KV evidence missing or mismatched: {dtype!r}")
    if not isinstance(running, int) or running < concurrency:
        raise ValueError(f"unavailable max_running_requests evidence: {running!r}")
    if not isinstance(capacity, int) or capacity < required:
        raise ValueError(f"unavailable/insufficient token_capacity evidence: {capacity!r} < {required}")
    return {"model_revision": expected_model_revision, "source_revision": expected_source_revision,
            "context_length": context, "kv_cache_dtype": dtype,
            "max_running_requests": running, "token_capacity": capacity,
            "required_token_capacity": required}


def build_request(model: str, prompt: str, request_id: str, max_tokens: int = MAX_TOKENS,
                  reasoning_effort: str = "medium") -> dict[str, Any]:
    return {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True}, "reasoning_effort": reasoning_effort}


def _event(payload: str) -> Any:
    if payload.strip() == "[DONE]":
        return "[DONE]"
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return {"malformed_payload": payload}


def _stream(spec: dict[str, Any], result: dict[str, Any], barrier: threading.Barrier,
            builder: ExactPromptBuilder, prompt_tokens: int, max_tokens: int, timeout: float,
            reasoning_effort: str = "medium") -> None:
    rid = result["request_id"]
    try:
        result["token_proof"] = builder.prove(result["prompt"], prompt_tokens)
        barrier.wait(timeout=timeout)
        body = build_request(spec["model"], result["prompt"], rid, max_tokens, reasoning_effort)
        result["request"] = body
        result["timestamps"]["submission"] = _stamp()
        req = urllib.request.Request(spec["base_url"].rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(body, ensure_ascii=False).encode(),
                                     headers={"accept": "text/event-stream", "content-type": "application/json",
                                              "X-Request-ID": rid})
        event_lines: list[str] = []
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result["http_status"] = response.status
            result["headers"] = dict(response.headers.items())
            while True:
                line = response.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace")
                result["raw_sse"].append(decoded)
                if decoded in ("\n", "\r\n"):
                    if event_lines:
                        parsed = _event("\n".join(event_lines)); result["events"].append({"timestamp": _stamp(), "event": parsed}); event_lines = []
                elif decoded.startswith("data:"):
                    event_lines.append(decoded[5:].strip())
        if event_lines:
            result["events"].append({"timestamp": _stamp(), "event": _event("\n".join(event_lines))})
        for event_record in result["events"]:
            event = event_record["event"]
            if not isinstance(event, dict):
                continue
            usage = event.get("usage")
            if isinstance(usage, dict): result["usage"] = usage
            for choice in event.get("choices", []):
                if isinstance(choice, dict) and choice.get("finish_reason"):
                    result["finish_reason"] = choice["finish_reason"]
                delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
                if isinstance(delta, dict) and (delta.get("content") or delta.get("reasoning_content")) and "first_token" not in result["timestamps"]:
                    result["timestamps"]["first_token"] = event_record["timestamp"]
        result["timestamps"]["completion"] = _stamp()
    except Exception as exc:
        if "request" not in result:
            try: barrier.abort()
            except Exception: pass
        if isinstance(exc, urllib.error.HTTPError):
            body = exc.read().decode("utf-8", errors="replace")
            result["http_status"] = exc.code; result["error_body"] = body
        result["error"] = f"{type(exc).__name__}: {exc}"
        text = result["error"].lower() + " " + result.get("error_body", "").lower()
        result["failure_class"] = ("oom" if "oom" in text or "out of memory" in text else
                                    "timeout" if "timeout" in text else
                                    "http_error" if result.get("http_status", 0) >= 400 else "disconnect")
        result["timestamps"]["completion"] = _stamp()


def validate_result(result: dict[str, Any], prompt_tokens: int = PROMPT_TOKENS,
                    max_tokens: int = MAX_TOKENS) -> list[str]:
    usage = result.get("usage") or {}
    expected = {"prompt_tokens": prompt_tokens, "completion_tokens": max_tokens,
                "total_tokens": prompt_tokens + max_tokens}
    errors = []
    if result.get("error"): errors.append(result["error"])
    for key, value in expected.items():
        if usage.get(key) != value: errors.append(f"usage {key}={usage.get(key)!r}, expected {value}")
    if result.get("finish_reason") != "length":
        errors.append(f"finish_reason={result.get('finish_reason')!r}, expected 'length'")
    return errors


def run_probe(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    started = _stamp(); output: dict[str, Any] = {"schema": "qwen38.bf16-context-probe", "version": 1,
        "started": started, "config": {"prompt_tokens": args.prompt_tokens, "max_tokens": args.max_tokens,
        "concurrency": args.concurrency, "base_url": args.base_url, "model": args.model}, "results": []}
    try:
        info = None
        for endpoint in ("/server_info", "/get_server_info"):
            status, _, body = _json_request(args.base_url.rstrip("/") + endpoint, timeout=args.timeout)
            if status == 200: info = body; output["server_info"] = {"endpoint": endpoint, "body": body}; break
        output["preflight"] = capacity_preflight(info, args.prompt_tokens, args.max_tokens, args.concurrency)
    except Exception as exc:
        output["status"] = "fail"; output["failure"] = str(exc); output["ended"] = _stamp(); pathlib.Path(args.output).write_text(json.dumps(output, indent=2) + "\n"); return output, 1
    try:
        prompts = [ExactPromptBuilder(args.base_url, args.model, reasoning_effort="medium", timeout=args.timeout).build(args.prompt_tokens, namespace=f"bf16-probe-{i}")[0] for i in range(args.concurrency)]
    except Exception as exc:
        output["status"] = "fail"; output["failure"] = f"prompt build: {type(exc).__name__}: {exc}"; output["ended"] = _stamp(); pathlib.Path(args.output).write_text(json.dumps(output, indent=2) + "\n"); return output, 1
    barrier = threading.Barrier(args.concurrency, timeout=args.timeout)
    results = [{"request_id": f"bf16-context-{i}-{hashlib.sha256(p.encode()).hexdigest()[:12]}", "prompt": p,
                "raw_sse": [], "events": [], "timestamps": {}, "usage": None} for i, p in enumerate(prompts)]
    threads = [threading.Thread(target=_stream, args=({"base_url": args.base_url, "model": args.model}, r, barrier, ExactPromptBuilder(args.base_url, args.model, reasoning_effort="medium", timeout=args.timeout), args.prompt_tokens, args.max_tokens, args.timeout, "medium")) for r in results]
    for thread in threads: thread.start()
    for thread in threads: thread.join(args.timeout + 5)
    output["results"] = results
    intervals = [(r["timestamps"].get("submission", {}).get("monotonic_s"), r["timestamps"].get("completion", {}).get("monotonic_s")) for r in results]
    first_tokens = [r["timestamps"].get("first_token", {}).get("monotonic_s") for r in results]
    overlap = len(intervals) == 2 and all(v is not None for pair in intervals for v in pair) and all(v is not None for v in first_tokens) and max(first_tokens) < min(b for _, b in intervals)
    output["concurrency_evidence"] = {"intervals": intervals, "first_tokens": first_tokens, "generation_overlap": overlap, "basis": "client-observed first-token/completion timestamps; scheduler residency is not claimed"}
    errors = [f"{r['request_id']}: {e}" for r in results for e in validate_result(r, args.prompt_tokens, args.max_tokens)]
    if any(thread.is_alive() for thread in threads): errors.append("request thread remained alive after join")
    if not overlap: errors.append("requests did not overlap")
    output["status"] = "pass" if not errors else "fail"; output["failure_reasons"] = errors; output["ended"] = _stamp()
    pathlib.Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    return output, 0 if not errors else 1


def _json_request(url: str, timeout: float) -> tuple[int, str, Any]:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"accept": "application/json"}), timeout=timeout) as response:
            return response.status, response.headers.get("content-type", ""), json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("content-type", ""), {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL); parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--output", type=pathlib.Path, required=True); parser.add_argument("--prompt-tokens", type=int, default=PROMPT_TOKENS)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS); parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args(argv); _, code = run_probe(args); return code


if __name__ == "__main__": raise SystemExit(main())
