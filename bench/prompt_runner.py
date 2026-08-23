#!/usr/bin/env python3
"""Run a directory of coding prompts against an OpenAI-compatible endpoint.

Prompt files are read as bytes and decoded only for JSON transport; their
contents are never interpreted or executed.  The output is an audit-friendly
JSON document containing the request, response, timing, and server metadata.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import time
import urllib.error
import urllib.request
from typing import Any


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def json_request(url: str, body: dict[str, Any] | None = None, timeout: float = 1800) -> tuple[int, str, Any]:
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
    headers = {"accept": "application/json"}
    if body is not None:
        headers["content-type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if "text/event-stream" in response.headers.get("content-type", ""):
                lines: list[str] = []
                first_token_time = None
                while True:
                    line = response.readline()
                    if not line:
                        break
                    decoded = line.decode("utf-8", errors="replace")
                    lines.append(decoded)
                    if first_token_time is None and decoded.startswith("data:"):
                        try:
                            event = json.loads(decoded[5:].strip())
                            delta = (event.get("choices") or [{}])[0].get("delta") or {}
                            if delta.get("content") or delta.get("reasoning_content"):
                                first_token_time = time.monotonic()
                        except (ValueError, IndexError, AttributeError):
                            pass
                raw = "".join(lines).encode()
            else:
                raw = response.read()
            status = response.status
            content_type = response.headers.get("content-type", "")
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = exc.code
        content_type = exc.headers.get("content-type", "")
    text = raw.decode("utf-8", errors="replace")
    if "text/event-stream" in content_type or text.lstrip().startswith("data:"):
        parsed = parse_sse(text)
        if 'first_token_time' in locals():
            parsed["first_token_time"] = first_token_time
        return status, content_type, parsed
    try:
        return status, content_type, json.loads(text) if text else {}
    except json.JSONDecodeError:
        return status, content_type, {"raw": text}


def parse_sse(text: str) -> dict[str, Any]:
    events: list[Any] = []
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload and payload != "[DONE]":
            try:
                events.append(json.loads(payload))
            except json.JSONDecodeError:
                events.append({"raw": payload})
    content: list[str] = []
    reasoning: list[str] = []
    finish_reason = None
    usage = None
    for event in events:
        usage = event.get("usage") or usage
        for choice in event.get("choices", []):
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning.append(delta["reasoning_content"])
    return {"events": events, "content": "".join(content),
            "reasoning_content": "".join(reasoning),
            "finish_reason": finish_reason, "usage": usage}


def usage_fields(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        return {}
    usage = response.get("usage")
    if not isinstance(usage, dict):
        usage = response.get("meta_info") if isinstance(response.get("meta_info"), dict) else {}
    return {name: usage.get(name) for name in
            ("prompt_tokens", "completion_tokens", "total_tokens",
             "reasoning_tokens", "prompt_token_count", "completion_token_count")
            if name in usage}


def speculative_fields(value: Any) -> dict[str, Any]:
    """Retain server fields whose names identify speculative decoding stats."""
    found: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if any(word in key.lower() for word in ("spec", "accept")):
                found[key] = item
            elif isinstance(item, (dict, list)):
                nested = speculative_fields(item)
                if nested:
                    found[key] = nested
    elif isinstance(value, list):
        for item in value:
            nested = speculative_fields(item)
            if nested:
                found.update(nested)
    return found


def build_request_body(model: str, prompt: str, max_tokens: int, stream: bool,
                       reasoning_effort: str | None = None) -> dict[str, Any]:
    """Build the exact request body used by both live and dry-run modes."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    return body


def extract_chat(response: Any, streamed: bool) -> tuple[str, str, Any, Any, dict[str, Any]]:
    if streamed:
        return (response.get("content", ""), response.get("reasoning_content", ""),
                response.get("finish_reason"), response.get("usage"), response)
    choices = response.get("choices", []) if isinstance(response, dict) else []
    message = choices[0].get("message", {}) if choices else {}
    return (message.get("content", "") or "", message.get("reasoning_content", "") or "",
            choices[0].get("finish_reason") if choices else None,
            response.get("usage") if isinstance(response, dict) else None, response)


def run_prompt(base_url: str, model: str, prompt_path: pathlib.Path, max_tokens: int,
               timeout: float, stream: bool, reasoning_effort: str | None = None) -> dict[str, Any]:
    raw = prompt_path.read_bytes()
    prompt = raw.decode("utf-8")
    request_body = build_request_body(model, prompt, max_tokens, stream, reasoning_effort)
    started = time.time()
    started_mono = time.monotonic()
    first_token = None
    try:
        status, content_type, response = json_request(
            base_url.rstrip("/") + "/v1/chat/completions", request_body, timeout)
        ended = time.time()
        ended_mono = time.monotonic()
        if stream and isinstance(response, dict):
            # json_request has already parsed the complete SSE body.  This is
            # a conservative TTFT proxy; exact TTFT is supplied by live mode
            # only when the transport records an event timestamp.
            events = response.get("events", [])
            first_token = response.get("first_token_time")
        content, reasoning, finish, usage, raw_response = extract_chat(response, stream)
        fields = usage_fields({"usage": usage})
        completion = fields.get("completion_tokens") or fields.get("completion_token_count")
        elapsed = ended_mono - started_mono
        post_first = (ended_mono - first_token) if first_token is not None else None
        metrics_unavailable = []
        if completion is None:
            metrics_unavailable.append("completion_token_usage_missing")
        if first_token is None:
            metrics_unavailable.append("first_token_time_missing")
        result = {"prompt_file": prompt_path.name, "prompt_sha256": hashlib.sha256(raw).hexdigest(),
                "request_start": dt.datetime.fromtimestamp(started, dt.timezone.utc).isoformat(),
                "request_end": dt.datetime.fromtimestamp(ended, dt.timezone.utc).isoformat(),
                "http_status": status, "content_type": content_type, "request": request_body,
                "finish_reason": finish, "usage": fields, "wall_duration_s": elapsed,
                "ttft_s": (first_token - started_mono) if first_token is not None else None,
                "completion_tok_s_after_first": ((completion - 1) / post_first
                    if completion is not None and completion > 1 and post_first and post_first > 0 else None),
                "completion_tok_s_end_to_end": (completion / elapsed) if completion and elapsed > 0 else None,
                "metrics_unavailable": metrics_unavailable,
                "content": content, "reasoning_content": reasoning,
                "speculative_stats": {**speculative_fields(raw_response), **speculative_fields(usage)},
                "response": raw_response}
        if status >= 400:
            result["error"] = f"HTTP {status}"
        result["completion_valid"] = status < 400 and finish == "stop"
        if status < 400 and finish != "stop":
            result["incomplete"] = f"finish_reason={finish!r}"
        return result
    except (OSError, ValueError, urllib.error.URLError) as exc:
        ended = time.time()
        ended_mono = time.monotonic()
        return {"prompt_file": prompt_path.name, "prompt_sha256": hashlib.sha256(raw).hexdigest(),
                "request_start": dt.datetime.fromtimestamp(started, dt.timezone.utc).isoformat(),
                "request_end": dt.datetime.fromtimestamp(ended, dt.timezone.utc).isoformat(),
                "http_status": None, "request": request_body, "error": str(exc),
                "wall_duration_s": ended_mono - started_mono, "content": "", "reasoning_content": "",
                "metrics_unavailable": ["request_failed"], "speculative_stats": {}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-dir", type=pathlib.Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:11436")
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--reasoning-effort", default=None,
                        help="Optional explicit reasoning control; omitted by default")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    paths = sorted(p for p in args.prompt_dir.iterdir() if p.is_file() and p.suffix == ".txt")
    metadata: dict[str, Any] = {"schema": "qwen38.coding_prompt_run", "version": 1,
        "run_start": utc_now(), "base_url": args.base_url, "model": args.model,
        "max_tokens": args.max_tokens, "stream": not args.no_stream,
        "prompt_dir": str(args.prompt_dir), "prompt_count": len(paths)}
    server_info = None
    if not args.dry_run:
        for endpoint in ("/server_info", "/get_server_info"):
            try:
                status, _, body = json_request(args.base_url.rstrip("/") + endpoint, timeout=args.timeout)
                if status == 200:
                    server_info = {"endpoint": endpoint, "status": status, "body": body}
                    break
            except OSError:
                pass
    metadata["server_info"] = server_info
    if args.dry_run:
        results = []
        for p in paths:
            raw = p.read_bytes()
            results.append({"prompt_file": p.name,
                            "prompt_sha256": hashlib.sha256(raw).hexdigest(),
                            "request": build_request_body(
                                args.model, raw.decode("utf-8"), args.max_tokens,
                                not args.no_stream, args.reasoning_effort)})
    else:
        results = [run_prompt(args.base_url, args.model, p, args.max_tokens, args.timeout,
                              not args.no_stream, args.reasoning_effort)
                   for p in paths]
    metadata["run_end"] = utc_now()
    output = {"metadata": metadata, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "prompt_count": len(paths),
                      "errors": sum("error" in r for r in results),
                      "incomplete": sum(r.get("completion_valid") is False
                                        and "error" not in r for r in results)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
