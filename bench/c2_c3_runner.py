#!/usr/bin/env python3
"""Concurrent streaming client for the queued C2/C3 campaign.

This module deliberately records raw observations rather than deciding whether
they prove a campaign result.  ``c2_c3_importer.py`` performs the fail-closed
validation.  Server scheduler events are an independent input: client timing
and response headers are never promoted to admission or queue evidence.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable

SCHEMA = "qwen38.c2-c3-concurrent-run"
VERSION = 1
SPEC_SCHEMA = "qwen38.c2-c3-run-spec"


def _utc(epoch: float | None = None) -> str:
    value = dt.datetime.now(dt.timezone.utc) if epoch is None else dt.datetime.fromtimestamp(epoch, dt.timezone.utc)
    return value.isoformat()


def _stamp() -> dict[str, Any]:
    return {"utc": _utc(), "monotonic_s": time.monotonic()}


def build_body(spec: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": spec["model"],
        "messages": [{"role": "user", "content": request["prompt"]}],
        "max_tokens": request["max_tokens"],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if request.get("reasoning_effort") is not None:
        body["reasoning_effort"] = request["reasoning_effort"]
    if request.get("ignore_eos") is True:
        body["ignore_eos"] = True
    return body


def validate_spec(spec: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(spec, dict):
        return ["spec must be an object"]
    if spec.get("schema") != SPEC_SCHEMA or spec.get("version") != 1:
        errors.append("unsupported spec schema/version")
    for key in ("run_id", "profile", "stage", "base_url", "model"):
        if not isinstance(spec.get(key), str) or not spec[key].strip():
            errors.append(f"invalid {key}")
    requests = spec.get("requests")
    if not isinstance(requests, list) or not requests:
        return errors + ["requests must be a non-empty list"]
    seen: set[str] = set()
    for index, request in enumerate(requests):
        prefix = f"requests[{index}]"
        if not isinstance(request, dict):
            errors.append(f"{prefix} must be an object")
            continue
        rid = request.get("client_request_id")
        if not isinstance(rid, str) or not rid.strip() or rid in seen:
            errors.append(f"{prefix} client_request_id must be unique and non-empty")
        else:
            seen.add(rid)
        if not isinstance(request.get("prompt"), str):
            errors.append(f"{prefix} prompt must be a string")
        if type(request.get("max_tokens")) is not int or request["max_tokens"] <= 0:
            errors.append(f"{prefix} max_tokens must be positive")
        if request.get("forced_output") not in (True, False):
            errors.append(f"{prefix} forced_output must be boolean")
        if request.get("forced_output") is True and request.get("ignore_eos") is not True:
            errors.append(f"{prefix} forced output requires ignore_eos")
        if (request.get("expected_prompt_tokens") is not None and
                (type(request["expected_prompt_tokens"]) is not int or request["expected_prompt_tokens"] <= 0)):
            errors.append(f"{prefix} expected_prompt_tokens must be positive when present")
    boundary = spec.get("boundary_proof")
    if boundary is not None and (not isinstance(boundary, dict) or
            type(boundary.get("expected_server_prompt_tokens")) is not int or
            boundary["expected_server_prompt_tokens"] <= 0):
        errors.append("boundary_proof requires expected_server_prompt_tokens")
    return errors


def _event_parts(event: Any) -> tuple[str, str, Any, Any]:
    content: list[str] = []
    reasoning: list[str] = []
    finish = None
    usage = None
    if isinstance(event, dict):
        usage = event.get("usage")
        for choice in event.get("choices", []) if isinstance(event.get("choices", []), list) else []:
            if not isinstance(choice, dict):
                continue
            finish = choice.get("finish_reason") or finish
            delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
            if isinstance(delta.get("content"), str):
                content.append(delta["content"])
            if isinstance(delta.get("reasoning_content"), str):
                reasoning.append(delta["reasoning_content"])
    return "".join(content), "".join(reasoning), finish, usage


def _parse_sse_payload(raw: str) -> Any:
    payload = raw.strip()
    if payload == "[DONE]":
        return "[DONE]"
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return {"malformed_payload": payload}


def _record_sse(response: Any, result: dict[str, Any]) -> None:
    """Read one SSE response incrementally, retaining every line and event."""
    event_lines: list[str] = []
    try:
        while True:
            raw_line = response.readline()
            if not raw_line:
                break
            decoded = raw_line.decode("utf-8", errors="replace")
            result["raw_sse"].append({**_stamp(), "line": decoded})
            if decoded in ("\n", "\r\n"):
                if event_lines:
                    _append_event(event_lines, result)
                    event_lines = []
            elif decoded.startswith("data:"):
                event_lines.append(decoded[5:].lstrip().rstrip("\r\n"))
    finally:
        # A timeout/disconnect may interrupt an SSE block before its blank
        # delimiter.  The already-read data remains evidence and must not be
        # discarded merely because the next readline failed.
        if event_lines:
            _append_event(event_lines, result)


def _append_event(lines: list[str], result: dict[str, Any]) -> None:
    stamp = _stamp()
    raw = "\n".join(lines)
    parsed = _parse_sse_payload(raw)
    event = {**stamp, "raw_data": raw, "parsed": parsed}
    result["events"].append(event)
    if parsed == "[DONE]":
        result["done_event"] = stamp
        return
    content, reasoning, finish, usage = _event_parts(parsed)
    result["content"] += content
    result["reasoning_content"] += reasoning
    if finish is not None:
        result["finish_reason"] = finish
    if isinstance(usage, dict):
        result["usage"] = usage
    if (content or reasoning) and result["timestamps"]["first_token"] is None:
        result["timestamps"]["first_token"] = stamp


def _failure_bucket(exc: BaseException, result: dict[str, Any]) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if isinstance(exc, TimeoutError) or "timed out" in text or "timeout" in text:
        return "timeout"
    if "out of memory" in text or "oom" in text:
        return "oom"
    if "restart" in text or "connection reset" in text or "remote end closed" in text:
        return "restart"
    if isinstance(exc, urllib.error.HTTPError) or (isinstance(result.get("http_status"), int) and result["http_status"] >= 400):
        return "http_error"
    return "disconnect"


def _server_failure_bucket(result: dict[str, Any], fallback: str) -> str:
    evidence = (result.get("raw_error_body", "") + " " +
                json.dumps(result.get("events", []), ensure_ascii=False)).lower()
    if "out of memory" in evidence or "cuda oom" in evidence:
        return "oom"
    if "server restart" in evidence or "process restart" in evidence:
        return "restart"
    return fallback


def stream_request(spec: dict[str, Any], request_spec: dict[str, Any], result: dict[str, Any], timeout: float) -> None:
    body = build_body(spec, request_spec)
    result["request"] = body
    endpoint = spec["base_url"].rstrip("/") + "/v1/chat/completions"
    payload = json.dumps(body, ensure_ascii=False).encode()
    headers = {"accept": "text/event-stream", "content-type": "application/json",
               "x-request-id": request_spec["client_request_id"]}
    result["timestamps"]["submission"] = _stamp()
    transport = urllib.request.Request(endpoint, data=payload, headers=headers)
    response = None
    try:
        response = urllib.request.urlopen(transport, timeout=timeout)
        result["http_status"] = response.status
        result["response_headers"] = dict(response.headers.items())
        result["timestamps"]["headers"] = _stamp()
        _record_sse(response, result)
    except urllib.error.HTTPError as exc:
        response = exc
        result["http_status"] = exc.code
        result["response_headers"] = dict(exc.headers.items()) if exc.headers else {}
        result["timestamps"]["headers"] = _stamp()
        try:
            if "text/event-stream" in result["response_headers"].get("Content-Type", ""):
                _record_sse(exc, result)
            else:
                body_bytes = exc.read()
                result["raw_error_body"] = body_bytes.decode("utf-8", errors="replace")
        except Exception as read_exc:  # preserve both the HTTP and partial-read failures
            result["secondary_error"] = {"type": type(read_exc).__name__, "message": str(read_exc)}
        result["failure_bucket"] = _server_failure_bucket(result, "http_error")
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    except Exception as exc:
        result["failure_bucket"] = _failure_bucket(exc, result)
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass


def _empty_result(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "client_request_id": request["client_request_id"],
        "forced_output": request["forced_output"], "requested_output_tokens": request["max_tokens"],
        "expected_prompt_tokens": request.get("expected_prompt_tokens"),
        "timestamps": {"arrival": None, "submission": None, "headers": None,
                       "first_event": None, "first_token": None, "completion": None},
        "http_status": None, "response_headers": {}, "raw_sse": [], "events": [],
        "content": "", "reasoning_content": "", "finish_reason": None, "usage": None,
        "failure_bucket": None, "error": None,
    }


def _worker(spec: dict[str, Any], request: dict[str, Any], result: dict[str, Any], gate: threading.Barrier,
            timeout: float, transport: Callable[[dict[str, Any], dict[str, Any], dict[str, Any], float], None]) -> None:
    try:
        gate.wait()
        result["timestamps"]["arrival"] = _stamp()
        transport(spec, request, result, timeout)
    except Exception as exc:
        result["failure_bucket"] = _failure_bucket(exc, result)
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if result["events"]:
            result["timestamps"]["first_event"] = {
                key: result["events"][0][key] for key in ("utc", "monotonic_s")}
        result["timestamps"]["completion"] = _stamp()


class JsonlTail:
    """Continuously retain appended server scheduler JSONL, including malformed lines."""
    def __init__(self, path: pathlib.Path, interval: float = .05):
        self.path, self.interval = path, interval
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        offset = 0
        while not self._stop.is_set():
            offset = self._read(offset)
            self._stop.wait(self.interval)
        self._read(offset)

    def _read(self, offset: int) -> int:
        try:
            with self.path.open() as handle:
                handle.seek(offset)
                for line in handle:
                    raw = line.rstrip("\n")
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        parsed = {"malformed_line": raw}
                    self.samples.append({"collector_timestamp": _stamp(), "event": parsed})
                return handle.tell()
        except FileNotFoundError:
            return offset

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        if self._thread:
            self._thread.join(max(1.0, self.interval * 3))
        return self.samples


class GpuSampler:
    FIELDS = ("total_vram_bytes", "free_vram_bytes", "gpu_utilization_pct", "power_w", "temperature_c")

    def __init__(self, gpu: str, interval: float, command: Callable[..., Any] = subprocess.run):
        self.gpu, self.interval, self.command = gpu, interval, command
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def _sample(self) -> None:
        stamp = _stamp()
        query = "memory.total,memory.free,utilization.gpu,power.draw,temperature.gpu"
        try:
            completed = self.command(["nvidia-smi", "--id=" + self.gpu, "--query-gpu=" + query,
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                     timeout=max(2.0, self.interval))
            raw = completed.stdout.strip()
            values = [float(value.strip()) for value in raw.split(",")]
            if completed.returncode != 0 or len(values) != 5 or not all(math.isfinite(x) for x in values):
                raise ValueError(completed.stderr.strip() or f"malformed nvidia-smi output: {raw!r}")
            self.samples.append({**stamp, "source": "nvidia-smi", "gpu": self.gpu,
                "total_vram_bytes": int(values[0] * 1024 * 1024),
                "free_vram_bytes": int(values[1] * 1024 * 1024),
                "gpu_utilization_pct": values[2], "power_w": values[3],
                "temperature_c": values[4], "raw": raw})
        except Exception as exc:
            self.samples.append({**stamp, "source": "nvidia-smi", "gpu": self.gpu,
                                 "error": {"type": type(exc).__name__, "message": str(exc)}})

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        if self._thread:
            self._thread.join(max(1.0, self.interval * 2))
        self._sample()
        return self.samples


class ProcessSampler:
    """Continuously prove that one server PID/start identity spans the run."""
    def __init__(self, pid: int, interval: float):
        self.pid, self.interval = pid, interval
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def _sample(self) -> None:
        stamp = _stamp()
        try:
            stat = pathlib.Path(f"/proc/{self.pid}/stat").read_text()
            # Field 2 (comm) is parenthesized and may itself contain spaces;
            # parse after its final ')' so field 22 remains unambiguous.
            tail = stat[stat.rfind(")") + 2:].split()
            start_ticks = tail[19]
            self.samples.append({"timestamp": stamp["utc"], "monotonic_s": stamp["monotonic_s"],
                                 "server_process_id": f"pid:{self.pid}:start_ticks:{start_ticks}"})
        except Exception as exc:
            self.samples.append({"timestamp": stamp["utc"], "monotonic_s": stamp["monotonic_s"],
                                 "server_process_id": None,
                                 "error": {"type": type(exc).__name__, "message": str(exc)}})

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        if self._thread:
            self._thread.join(max(1.0, self.interval * 2))
        self._sample()
        return self.samples


def run_concurrent(spec: dict[str, Any], server_evidence: dict[str, Any], scheduler_path: pathlib.Path,
                   gpu: str = "0", timeout: float = 1800, sample_interval: float = 1.0,
                   transport: Callable[[dict[str, Any], dict[str, Any], dict[str, Any], float], None] = stream_request,
                   gpu_sampler: Any | None = None, scheduler_tail: Any | None = None,
                   process_sampler: Any | None = None, server_pid: int | None = None) -> dict[str, Any]:
    errors = validate_spec(spec)
    if errors:
        raise ValueError("; ".join(errors))
    results = [_empty_result(request) for request in spec["requests"]]
    gate = threading.Barrier(len(results) + 1)
    gpu_monitor = gpu_sampler or GpuSampler(gpu, sample_interval)
    scheduler_monitor = scheduler_tail or JsonlTail(scheduler_path, min(sample_interval, .1))
    if process_sampler is None and server_pid is None:
        raise ValueError("server_pid or process_sampler is required for continuous restart evidence")
    process_monitor = process_sampler or ProcessSampler(server_pid, sample_interval)
    run_start = _stamp()
    gpu_monitor.start(); scheduler_monitor.start(); process_monitor.start()
    threads = [threading.Thread(target=_worker,
        args=(spec, request, result, gate, timeout, transport), daemon=True)
        for request, result in zip(spec["requests"], results)]
    for thread in threads:
        thread.start()
    gate.wait()
    release = _stamp()
    for thread in threads:
        thread.join()
    run_end = _stamp()
    scheduler_evidence = scheduler_monitor.stop()
    gpu_telemetry = gpu_monitor.stop()
    process_samples = process_monitor.stop()
    server_evidence = dict(server_evidence)
    process_ids = {sample.get("server_process_id") for sample in process_samples}
    server_evidence["process"] = {"samples": process_samples,
        "restart_count": max(0, len(process_ids - {None}) - 1),
        "sampling_interval_s": sample_interval}
    return {"schema": SCHEMA, "version": VERSION, "run_id": spec["run_id"],
        "profile": spec["profile"], "stage": spec["stage"], "repetition": spec.get("repetition"),
        "timestamps": {"collector_start": run_start, "barrier_release": release, "run_end": run_end},
        "request_count": len(results), "requests": results, "server": server_evidence,
        "scheduler_evidence": scheduler_evidence, "gpu_telemetry": gpu_telemetry,
        "boundary_proof": spec.get("boundary_proof"),
        "collector": {"gpu_interval_s": sample_interval, "scheduler_source": str(scheduler_path),
                      "scheduler_source_contract": "server-generated request-correlated JSONL"}}


def dry_run(spec: dict[str, Any]) -> dict[str, Any]:
    errors = validate_spec(spec)
    if errors:
        raise ValueError("; ".join(errors))
    return {"schema": SCHEMA, "version": VERSION, "mode": "dry-run", "run_id": spec["run_id"],
            "profile": spec["profile"], "stage": spec["stage"], "barrier_parties": len(spec["requests"]),
            "requests": [{"client_request_id": request["client_request_id"],
                          "headers": {"x-request-id": request["client_request_id"]},
                          "body": build_body(spec, request)} for request in spec["requests"]]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("dry-run", "run"):
        command = sub.add_parser(name)
        command.add_argument("--spec", required=True, type=pathlib.Path)
        command.add_argument("--output", required=True, type=pathlib.Path)
    live = sub.choices["run"]
    live.add_argument("--server-evidence", required=True, type=pathlib.Path)
    live.add_argument("--scheduler-events", required=True, type=pathlib.Path)
    live.add_argument("--gpu", default="0")
    live.add_argument("--server-pid", required=True, type=int)
    live.add_argument("--timeout", type=float, default=1800)
    live.add_argument("--sample-interval", type=float, default=1.0)
    args = parser.parse_args(argv)
    spec = json.loads(args.spec.read_text())
    if args.command == "dry-run":
        document = dry_run(spec)
    else:
        server = json.loads(args.server_evidence.read_text())
        if args.sample_interval <= 0 or args.timeout <= 0:
            parser.error("--sample-interval and --timeout must be positive")
        document = run_concurrent(spec, server, args.scheduler_events, args.gpu,
                                  args.timeout, args.sample_interval, server_pid=args.server_pid)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
