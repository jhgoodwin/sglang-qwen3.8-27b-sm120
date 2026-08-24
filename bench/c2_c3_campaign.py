#!/usr/bin/env python3
"""Fail-closed driver for the queued native-context C2/C3 campaign.

The driver owns request construction and durable orchestration; container
lifecycle remains with the supervisor.  ``plan`` never contacts a server.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from bench import c2_c3_importer, c2_c3_runner

MANIFEST = pathlib.Path(__file__).with_name("c2-c3-native-context-campaign.json")
TARGETS = {"B-near-native-prefill": (261120, 1024),
           "C-max-output-decode": (1024, 131072),
           "D-combined-boundary-safe": (130048, 131072),
           "E-four-arrival-queue": (130048, 131072)}
STAGES = ("A-boot-admission", "B-near-native-prefill", "C-max-output-decode",
          "D-combined-boundary-safe", "E-four-arrival-queue")
COLD_BOOT_STAGES = ("A-boot-admission", "B-near-native-prefill")
RESTART_EXIT = 75


def post_json(url: str, body: dict[str, Any], timeout: float = 30) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode())
        if not isinstance(payload, dict):
            raise ValueError("endpoint response must be an object")
        return response.status, payload


def token_count(response: dict[str, Any]) -> int:
    """Accept the pinned server's token response, but never infer a count."""
    for key in ("count", "prompt_tokens", "num_tokens"):
        if type(response.get(key)) is int:
            return response[key]
    for key in ("tokens", "input_ids", "token_ids"):
        value = response.get(key)
        if isinstance(value, list):
            return len(value)
    usage = response.get("usage")
    if isinstance(usage, dict) and type(usage.get("prompt_tokens")) is int:
        return usage["prompt_tokens"]
    raise ValueError("tokenize response has no explicit token count")


class ExactPromptBuilder:
    """Construct exact chat prompt counts using bounded feedback search.

    Character length is only a search coordinate.  Every accepted prompt is
    proved by tokenizing the exact messages sent to generation.  A local scan
    around the binary estimate handles tokenizer merges and non-monotonic
    points without trusting monotonicity as proof.
    """
    def __init__(self, endpoint: str, model: str, reasoning_effort: str = "medium",
                 max_calls: int = 240, timeout: float = 30,
                 tokenize: Callable[[str, dict[str, Any], float], tuple[int, dict[str, Any]]] = post_json,
                 artifact_dir: pathlib.Path | None = None):
        self.endpoint = endpoint.rstrip("/") + "/v1/tokenize"
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_calls = max_calls
        self.timeout = timeout
        self.tokenize = tokenize
        self.artifact_dir = artifact_dir
        self.calls: list[dict[str, Any]] = []
        self._operation_call_start = 0

    def _messages(self, text: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": text}]

    def _probe(self, text: str) -> int:
        if len(self.calls) - self._operation_call_start >= self.max_calls:
            raise RuntimeError(f"tokenize call bound {self.max_calls} exceeded")
        messages = self._messages(text)
        body = {"model": self.model, "messages": messages,
                "reasoning_effort": self.reasoning_effort}
        try:
            status, response = self.tokenize(self.endpoint, body, self.timeout)
        except Exception as exc:
            self.calls.append({"messages_sha256": _messages_hash(messages), "error": f"{type(exc).__name__}: {exc}"})
            raise
        count = token_count(response)
        record: dict[str, Any] = {"messages_sha256": _messages_hash(messages), "message_chars": len(text),
                                  "status": status, "token_count": count}
        if self.artifact_dir is not None:
            response_dir = self.artifact_dir / "tokenize-responses"
            response_dir.mkdir(parents=True, exist_ok=True)
            response_path = response_dir / f"{len(self.calls):04d}.json"
            response_path.write_text(json.dumps(_tokenize_response_summary(response), indent=2) + "\n")
            record["response_artifact"] = str(response_path)
        else:
            record["response"] = response
        self.calls.append(record)
        return count

    def build(self, target: int, namespace: str = "default") -> tuple[str, dict[str, Any]]:
        if type(target) is not int or target <= 0:
            raise ValueError("target token count must be positive")
        # The server's chat template remains the source of truth.  The coarse
        # unit is intentionally multi-token; correction uses independent
        # suffix candidates and never assumes one repetition equals one token.
        call_start = len(self.calls)
        self._operation_call_start = call_start
        nonce = hashlib.sha256(namespace.encode()).hexdigest()
        unit = f" qwen38-{nonce[:16]}-filler"
        # Put the nonce first so separate requests stop sharing a content
        # prefix immediately after the unavoidable chat-template tokens.
        base = f"{nonce}\nReturn the requested result."
        counts: dict[int, int] = {}
        def count_chars(n: int) -> int:
            n = max(0, n)
            if n not in counts:
                counts[n] = self._probe(base + unit * n)
            return counts[n]
        lo, hi = 0, 1
        while count_chars(hi) < target and hi < 1_000_000:
            lo, hi = hi, hi * 2
        if count_chars(hi) < target:
            raise RuntimeError("tokenizer cannot reach requested prompt count")
        # Binary search provides a useful center; exact proof is the local scan.
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if count_chars(mid) < target:
                lo = mid
            else:
                hi = mid
        center = lo
        # Search a bounded correction alphabet against the real tokenizer. It
        # covers residue gaps caused by multi-token units and merge boundaries.
        suffixes = ["", " ", "a", "b", "c", "x", "y", "z", "0", "1", ".", "!", "?", "\n"]
        suffixes += [a + b for a in suffixes[1:] for b in suffixes[1:]]
        suffixes += [a + b + c for a in (" a", " b", " c", " x", " y", " z")
                     for b in ("a", "b", "c", "x", "y", "z") for c in ("a", "b", "c", "x", "y", "z")]
        for chars in range(max(0, center - 2), center + 3):
            prefix = base + unit * chars
            for suffix in suffixes[:40]:
                text = prefix + suffix
                if self._probe(text) == target:
                    proof = {"target": target, "observed": target,
                             "messages_sha256": _messages_hash(self._messages(text)),
                             "namespace": namespace,
                             "algorithm": "nonce-separated-coarse-plus-tokenized-suffix"}
                    return text, proof
        raise RuntimeError(f"failed closed: no exact prompt count {target} after {len(self.calls)} calls")

    def prove(self, text: str, target: int) -> dict[str, Any]:
        """Re-tokenize the exact generation messages immediately before use."""
        self._operation_call_start = len(self.calls)
        observed = self._probe(text)
        if observed != target:
            raise RuntimeError(f"prompt drift: expected {target}, observed {observed}")
        return {"target": target, "observed": observed,
                "messages_sha256": _messages_hash(self._messages(text)),
                "call": self.calls[-1]}


def _messages_hash(messages: list[dict[str, str]]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _tokenize_response_summary(response: dict[str, Any]) -> dict[str, Any]:
    """Retain proof without copying hundreds of thousands of token IDs."""
    summary: dict[str, Any] = {"token_count": token_count(response)}
    for key in ("tokens", "input_ids", "token_ids"):
        value = response.get(key)
        if isinstance(value, list):
            encoded = json.dumps(value, separators=(",", ":")).encode()
            summary.update({"token_field": key, "token_ids_sha256": hashlib.sha256(encoded).hexdigest()})
            break
    for key in ("count", "prompt_tokens", "num_tokens", "usage"):
        if key in response and key not in summary:
            summary[key] = response[key]
    return summary


def request_shape(stage: str, profile: str, repetition: int, prompts: list[str], manifest: dict[str, Any], model: str | None = None) -> dict[str, Any]:
    spec: dict[str, Any] = {"schema": c2_c3_runner.SPEC_SCHEMA, "version": 1,
        "run_id": f"{profile}-{stage}-r{repetition}", "profile": profile,
        "stage": stage, "base_url": "", "model": model or manifest["runtime"]["model"],
        "requests": []}
    runtime = next(item for item in manifest["profiles"] if item["id"] == profile)
    input_tokens, output_tokens = TARGETS.get(stage, (16, 32))
    count = 4 if stage == "E-four-arrival-queue" else runtime["max_running_requests"]
    forced = stage not in ("A-boot-admission",)
    for index in range(count + (1 if stage == "A-boot-admission" else 0)):
        if prompts and len(prompts) != count + (1 if stage == "A-boot-admission" else 0):
            raise ValueError("one distinct prompt is required per concurrent request")
        prompt = prompts[index] if prompts else f"{profile}-{stage}-r{repetition}-q{index}: Reply briefly."
        spec["requests"].append({"client_request_id": f"{spec['run_id']}-{index}", "prompt": prompt,
            "max_tokens": output_tokens if forced else 32, "forced_output": forced,
            "ignore_eos": forced, "expected_prompt_tokens": input_tokens if forced else None,
            "reasoning_effort": "medium"})
    return spec


def free_vram_gate(raw: dict[str, Any], minimum: float = .05) -> None:
    telemetry = raw.get("gpu_telemetry", [])
    fractions = [row["free_vram_bytes"] / row["total_vram_bytes"] for row in telemetry
                 if isinstance(row, dict) and type(row.get("free_vram_bytes")) is int
                 and type(row.get("total_vram_bytes")) is int and row["total_vram_bytes"] > 0]
    if not fractions:
        raise RuntimeError("free VRAM gate failed: no measured telemetry")
    if min(fractions) < minimum:
        raise RuntimeError(f"free VRAM gate failed: minimum {min(fractions):.4f} < {minimum:.4f}")


def preflight_free_vram_gate(gpu: str = "0", minimum: float = .05) -> float:
    """Fail before an expensive request if current host VRAM is below the gate."""
    result = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=memory.free,memory.total",
                             "--format=csv,noheader,nounits"], check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    fields = [part.strip() for part in result.stdout.strip().split(",")]
    if len(fields) != 2:
        raise RuntimeError("free VRAM gate failed: malformed nvidia-smi output")
    try:
        free_mib, total_mib = (int(value) for value in fields)
    except ValueError as error:
        raise RuntimeError("free VRAM gate failed: malformed nvidia-smi output") from error
    if total_mib <= 0:
        raise RuntimeError("free VRAM gate failed: invalid total VRAM")
    fraction = free_mib / total_mib
    if fraction < minimum:
        raise RuntimeError(f"free VRAM gate failed before submission: {fraction:.4f} < {minimum:.4f}")
    return fraction


def short_warmup(base_url: str, model: str, artifact: pathlib.Path, timeout: float = 180) -> str:
    """Issue exactly one bounded stream before PID bootstrap."""
    rid = f"campaign-warmup-{time.time_ns()}-{hashlib.sha256(str(artifact).encode()).hexdigest()[:10]}"
    spec = {"model": model, "messages": [{"role": "user", "content": "Reply with one short word."}],
            "max_tokens": 8, "stream": True, "reasoning_effort": "medium"}
    request = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(spec).encode(), headers={"content-type": "application/json",
                                     "x-request-id": rid}, method="POST")
    rows: list[str] = []
    error: Exception | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            while True:
                line = response.readline()
                if not line:
                    break
                rows.append(line.decode("utf-8", errors="replace"))
    except Exception as exc:
        error = exc
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps({"request_id": rid, "request": spec, "raw_sse": rows,
        "error": None if error is None else f"{type(error).__name__}: {error}"}, indent=2) + "\n")
    if error is not None:
        raise error
    return rid


def _scheduler_has_request(path: pathlib.Path, request_id: str) -> bool:
    if not path.is_file():
        return False
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("client_request_id") == request_id:
            return True
    return False


def plan(manifest: dict[str, Any], profile: str, base_url: str) -> dict[str, Any]:
    runtime = next((p for p in manifest["profiles"] if p["id"] == profile), None)
    if runtime is None:
        raise ValueError("profile must be c2 or c3")
    return {"schema": "qwen38.c2-c3-campaign-plan", "version": 1, "mode": "plan",
            "profile": profile, "base_url": base_url, "network_contacted": False,
            "optional_boundary": {"enabled": False},
            "stages": [{"id": stage, "depends_on": next((x["depends_on"] for x in manifest["stages"] if x["id"] == stage), []),
                        "repetitions": 3, "shape": TARGETS.get(stage, {"admit": runtime["max_running_requests"], "excess": 1}),
                        "lifecycle": ("one measured repetition per fresh boot"
                                      if stage in COLD_BOOT_STAGES else
                                      "shares the final B lifecycle with unique-prefix requests")}
                       for stage in STAGES],
            "next_command": "run after supervisor-owned server boot and evidence collection; exit 75 requests each required restart"}


def _load_state(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema": "qwen38.c2-c3-campaign-state", "version": 2, "accepted": {},
                "failures": [], "lifecycles": [], "prompt_hashes": {}}
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema") != "qwen38.c2-c3-campaign-state":
        raise ValueError("invalid campaign state")
    return value


def _lifecycle(server: dict[str, Any], evidence_path: pathlib.Path,
               scheduler_path: pathlib.Path) -> dict[str, Any]:
    provenance = server.get("launch_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("server evidence lacks launch provenance")
    container_id = provenance.get("container_id")
    if not isinstance(container_id, str) or not container_id:
        raise ValueError("server evidence lacks container identity")
    pid, process_identity = c2_c3_runner.bootstrap_server_pid(scheduler_path)
    return {"container_id": container_id, "process_identity": process_identity,
            "server_pid": pid, "server_evidence_path": str(evidence_path.resolve()),
            "scheduler_events_path": str(scheduler_path.resolve()),
            "launch_artifact": provenance.get("artifact_reference")}


def _require_fresh_lifecycle(state: dict[str, Any], current: dict[str, Any]) -> None:
    if not state.get("lifecycles"):
        return
    previous = state["lifecycles"][-1]
    for field in ("container_id", "process_identity", "server_evidence_path",
                  "scheduler_events_path", "launch_artifact"):
        if current.get(field) == previous.get(field):
            raise RuntimeError(f"cold cell requires fresh {field}")


def _next_cell(state: dict[str, Any]) -> tuple[str, int] | None:
    accepted = state.get("accepted", {})
    for stage in STAGES:
        for repetition in range(1, 4):
            if f"{stage}/r{repetition}" not in accepted:
                return stage, repetition
    return None


def _attempt_number(state: dict[str, Any], key: str) -> int:
    """Return an append-only attempt number for an unaccepted cell."""
    return 1 + sum(1 for failure in state.get("failures", [])
                   if isinstance(failure, dict) and failure.get("key") == key)


def _attempt_name(base: str, attempt: int) -> str:
    if type(attempt) is not int or attempt <= 0:
        raise ValueError("attempt must be positive")
    return base if attempt == 1 else f"{base}-attempt{attempt}"


def _accepted_reimport(state: dict[str, Any], key: str) -> tuple[pathlib.Path, dict[str, Any], dict[str, Any]] | None:
    """Revalidate a preserved raw run before issuing any duplicate GPU work."""
    for failure in reversed(state.get("failures", [])):
        if not isinstance(failure, dict) or failure.get("key") != key:
            continue
        artifact = failure.get("artifact")
        if not isinstance(artifact, str):
            continue
        path = pathlib.Path(artifact)
        if not path.is_file():
            continue
        try:
            wrapper = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        raw = wrapper.get("raw") if isinstance(wrapper, dict) else None
        if not isinstance(raw, dict):
            continue
        imported = c2_c3_importer.validate_and_import(raw)
        if imported.get("accepted") is True:
            return path, raw, imported
    return None


def _write_state(path: pathlib.Path, state: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest", type=pathlib.Path, default=MANIFEST)
    common.add_argument("--profile", required=True, choices=("c2", "c3"))
    common.add_argument("--base-url", default=None)
    common.add_argument("--artifact-root", type=pathlib.Path, required=True)
    sub.add_parser("plan", parents=[common])
    run = sub.add_parser("run", parents=[common])
    run.add_argument("--server-evidence", type=pathlib.Path, required=True)
    run.add_argument("--scheduler-events", type=pathlib.Path, required=True)
    run.add_argument("--server-pid", default="auto")
    run.add_argument("--sample-interval", type=float, default=1.0)
    run.add_argument("--gpu", default="0")
    run.add_argument("--timeout", type=float, default=None,
                     help="per-cell stream timeout; defaults to 1800s, 2100s for C/D, 2700s for E")
    args = parser.parse_args(argv)
    if args.base_url is None:
        args.base_url = f"http://127.0.0.1:{11447 if args.profile == 'c2' else 11448}"
    manifest = json.loads(args.manifest.read_text())
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    if args.command == "plan":
        document = plan(manifest, args.profile, args.base_url)
        (args.artifact_root / "plan.json").write_text(json.dumps(document, indent=2) + "\n")
        return 0
    # Live execution intentionally requires evidence and a supervisor-owned
    # process.  The complete cell loop is delegated to the runner below.
    if not args.server_evidence.is_file() or not args.scheduler_events.is_file():
        raise SystemExit("BLOCKED: collect server evidence and scheduler JSONL before run")
    state_path = args.artifact_root / "state.json"
    state = _load_state(state_path)
    state.setdefault("lifecycles", [])
    state.setdefault("prompt_hashes", {})
    state["version"] = 2
    if state.get("profile") not in (None, args.profile):
        raise SystemExit("BLOCKED: artifact root belongs to another profile")
    state["profile"] = args.profile
    server = json.loads(args.server_evidence.read_text())
    observed_args = server.get("observed_server_args", {})
    model = observed_args.get("model_path")
    if not isinstance(model, str) or not model:
        raise SystemExit("BLOCKED: server evidence lacks observed model_path")
    if not args.server_evidence.is_file() or not args.scheduler_events.is_file():
        raise SystemExit("BLOCKED: evidence paths disappeared")

    # A ready-for-restart state must reject the old server before even issuing
    # another warmup.  Evidence paths are append-only provenance, not aliases.
    provenance = server.get("launch_provenance", {})
    static_lifecycle = {"container_id": provenance.get("container_id"),
        "server_evidence_path": str(args.server_evidence.resolve()),
        "scheduler_events_path": str(args.scheduler_events.resolve()),
        "launch_artifact": provenance.get("artifact_reference")}
    if state.get("status") == "ready_for_restart":
        try:
            _require_fresh_lifecycle(state, static_lifecycle)
        except RuntimeError as error:
            raise SystemExit(f"BLOCKED: {error}") from error

    same_lifecycle = bool(state["lifecycles"] and all(
        static_lifecycle.get(field) == state["lifecycles"][-1].get(field)
        for field in static_lifecycle))
    if same_lifecycle:
        lifecycle = state["lifecycles"][-1]
        pid, identity = c2_c3_runner.bootstrap_server_pid(args.scheduler_events)
        if identity != lifecycle.get("process_identity"):
            raise SystemExit("BLOCKED: scheduler identity changed within recorded lifecycle")
        lifecycle["server_pid"] = pid
    else:
        warmup_artifact = args.artifact_root / "warmup" / f"boot-{len(state['lifecycles']) + 1:02d}-short-stream.json"
        try:
            warmup_id = short_warmup(args.base_url, model, warmup_artifact, timeout=180)
        except Exception as error:
            state["failures"].append({"phase": "warmup", "artifact": str(warmup_artifact),
                                      "error": f"{type(error).__name__}: {error}"})
            _write_state(state_path, state)
            raise SystemExit(f"BLOCKED: warmup failed; partial artifact preserved at {warmup_artifact}") from error
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                if not _scheduler_has_request(args.scheduler_events, warmup_id):
                    raise ValueError("warmup scheduler event not observed")
                c2_c3_runner.bootstrap_server_pid(args.scheduler_events)
                break
            except ValueError:
                time.sleep(.5)
        else:
            raise SystemExit("BLOCKED: warmup completed but scheduler identity was not observed")
        lifecycle = _lifecycle(server, args.server_evidence, args.scheduler_events)
        try:
            _require_fresh_lifecycle(state, lifecycle)
        except RuntimeError as error:
            raise SystemExit(f"BLOCKED: {error}") from error
        lifecycle["warmup_artifact"] = str(warmup_artifact)
        lifecycle["warmup_request_id"] = warmup_id
        state["lifecycles"].append(lifecycle)
        state["status"] = "running"
        state.pop("next_command", None)
        _write_state(state_path, state)

    while (cell := _next_cell(state)) is not None:
            stage, repetition = cell
            key = f"{stage}/r{repetition}"
            attempt = _attempt_number(state, key)
            artifact_name = _attempt_name(f"{args.profile}-{stage}-r{repetition}", attempt)
            repetition_name = _attempt_name(f"r{repetition}", attempt)
            dependencies = next(x["depends_on"] for x in manifest["stages"] if x["id"] == stage)
            if not all(f"{dependency}/r{rep}" in state["accepted"]
                       for dependency in dependencies for rep in range(1, 4)):
                raise SystemExit(f"BLOCKED: dependency for {stage} has not completed accepted repetitions")
            if stage in COLD_BOOT_STAGES and lifecycle.get("measured_cell"):
                raise SystemExit(f"BLOCKED: {key} requires a fresh boot; current lifecycle already measured "
                                 f"{lifecycle['measured_cell']}")
            reimport = _accepted_reimport(state, key)
            if reimport is not None:
                source_path, raw, imported = reimport
                reimport_path = args.artifact_root / "reimports" / f"{artifact_name}-reimport.json"
                reimport_path.parent.mkdir(parents=True, exist_ok=True)
                raw_bytes = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
                reimport_path.write_text(json.dumps({"source_artifact": str(source_path),
                    "source_raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                    "imported": imported}, indent=2) + "\n")
                state["accepted"][key] = str(reimport_path)
                for index, request in enumerate(raw.get("requests", [])):
                    body = request.get("request") if isinstance(request, dict) else None
                    messages = body.get("messages") if isinstance(body, dict) else None
                    if isinstance(messages, list):
                        state["prompt_hashes"][_messages_hash(messages)] = f"{key}/q{index}"
                lifecycle["measured_cell"] = key if stage in COLD_BOOT_STAGES else lifecycle.get("measured_cell")
                lifecycle.setdefault("post_cold_lifecycle_rule",
                                     "C/D/E may share the final B lifecycle; prompts remain unique")
                following = _next_cell(state)
                if stage in COLD_BOOT_STAGES and following is not None and following[0] in COLD_BOOT_STAGES:
                    state["status"] = "ready_for_restart"
                    state["next_command"] = (f"stop the supervisor-owned server; start a fresh {args.profile} server "
                        f"with new evidence and scheduler paths; rerun this command for {following[0]}/r{following[1]}")
                _write_state(state_path, state)
                print(json.dumps({"status": state.get("status"), "accepted": key,
                                  "reimported_from": str(source_path),
                                  "next_cell": None if following is None else f"{following[0]}/r{following[1]}"}))
                if state.get("status") == "ready_for_restart":
                    return RESTART_EXIT
                continue
            try:
                preflight_fraction = preflight_free_vram_gate(args.gpu)
            except (OSError, subprocess.SubprocessError, RuntimeError) as error:
                state["failures"].append({"key": key, "phase": "preflight-vram",
                                          "error": f"{type(error).__name__}: {error}"})
                _write_state(state_path, state)
                raise SystemExit(f"BLOCKED: {error}") from error

            runtime = next(item for item in manifest["profiles"] if item["id"] == args.profile)
            request_count = (4 if stage == "E-four-arrival-queue" else
                             runtime["max_running_requests"] + (1 if stage == "A-boot-admission" else 0))
            input_tokens = TARGETS.get(stage, (None, None))[0]
            prompts: list[str] = []
            proofs: list[dict[str, Any]] = []
            builder = ExactPromptBuilder(args.base_url, model,
                                         artifact_dir=args.artifact_root / "tokenize" / stage /
                                         repetition_name)
            prompt_dir = args.artifact_root / "prompts" / stage / f"r{repetition}"
            prompt_dir.mkdir(parents=True, exist_ok=True)
            proof_path = args.artifact_root / "proofs" / stage / f"{repetition_name}.json"
            proof_path.parent.mkdir(parents=True, exist_ok=True)
            cache_variant = "cold-unique-prefix" if stage in COLD_BOOT_STAGES else "unique-prefix"
            try:
                for index in range(request_count):
                    namespace = f"{args.profile}/{stage}/r{repetition}/q{index}"
                    prompt_path = prompt_dir / f"q{index}.txt"
                    if input_tokens is None:
                        prompt = f"{hashlib.sha256(namespace.encode()).hexdigest()}\nReply briefly."
                    elif prompt_path.is_file():
                        prompt = prompt_path.read_text()
                    else:
                        prompt, build_proof = builder.build(input_tokens, namespace=namespace)
                        proofs.append({"phase": "build", **build_proof})
                        prompt_path.write_text(prompt)
                    prompt_hash = _messages_hash(builder._messages(prompt))
                    owner = state["prompt_hashes"].get(prompt_hash)
                    if owner is not None and owner != f"{key}/q{index}":
                        raise RuntimeError(f"prompt hash reused by {owner} and {key}/q{index}")
                    if prompt_hash in {_messages_hash(builder._messages(value)) for value in prompts}:
                        raise RuntimeError(f"concurrent prompts are not distinct for {key}")
                    prompts.append(prompt)

                # Re-tokenize every exact generation message after construction
                # and immediately before writing/submitting the measured batch.
                if input_tokens is not None:
                    for index, prompt in enumerate(prompts):
                        proofs.append({"phase": "pre-submit", "request_index": index,
                                       **builder.prove(prompt, input_tokens)})
            except Exception as error:
                proof_path.write_text(json.dumps({"cell": key, "cache_variant": cache_variant,
                    "status": "failed", "error": f"{type(error).__name__}: {error}",
                    "tokenize_calls": builder.calls, "proofs": proofs}, indent=2) + "\n")
                state["failures"].append({"key": key, "phase": "exact-prompt-proof",
                                          "artifact": str(proof_path),
                                          "error": f"{type(error).__name__}: {error}"})
                _write_state(state_path, state)
                raise SystemExit(f"BLOCKED: exact prompt proof failed; artifact preserved at {proof_path}") from error
            proof_path.write_text(json.dumps({"cell": key, "cache_variant": cache_variant,
                "distinct_messages_sha256": [_messages_hash(builder._messages(p)) for p in prompts],
                "tokenize_calls": builder.calls, "proofs": proofs}, indent=2) + "\n")
            specs = request_shape(stage, args.profile, repetition, prompts, manifest, model=model)
            specs["base_url"] = args.base_url
            specs["cache_variant"] = cache_variant
            specs["lifecycle"] = {key: lifecycle.get(key) for key in
                                  ("container_id", "process_identity", "server_evidence_path",
                                   "scheduler_events_path", "warmup_artifact")}
            specs["preflight_free_vram_fraction"] = preflight_fraction
            spec_path = args.artifact_root / "specs" / f"{artifact_name}.json"
            spec_path.parent.mkdir(parents=True, exist_ok=True)
            spec_path.write_text(json.dumps(specs, indent=2) + "\n")
            output = args.artifact_root / "raw" / f"{artifact_name}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            raw = None
            try:
                pid = c2_c3_runner.bootstrap_server_pid(args.scheduler_events)[0] if args.server_pid == "auto" else int(args.server_pid)
                timeout = args.timeout or (2700 if stage == "E-four-arrival-queue" else (2100 if stage in ("C-max-output-decode", "D-combined-boundary-safe") else 1800))
                raw = c2_c3_runner.run_concurrent(specs, server, args.scheduler_events, server_pid=pid,
                                                   gpu=args.gpu, timeout=timeout,
                                                   sample_interval=args.sample_interval)
                free_vram_gate(raw)
                imported = c2_c3_importer.validate_and_import(raw)
                output.write_text(json.dumps({"raw": raw, "imported": imported}, indent=2) + "\n")
                if not imported.get("accepted"):
                    state["failures"].append({"key": key, "artifact": str(output), "errors": imported.get("errors", [])})
                    _write_state(state_path, state)
                    raise SystemExit(f"BLOCKED: importer rejected {key}; artifact preserved at {output}")
                state["accepted"][key] = str(output)
                for index, prompt in enumerate(prompts):
                    state["prompt_hashes"][_messages_hash(builder._messages(prompt))] = f"{key}/q{index}"
                lifecycle["measured_cell"] = key if stage in COLD_BOOT_STAGES else lifecycle.get("measured_cell")
                lifecycle.setdefault("post_cold_lifecycle_rule", "C/D/E may share the final B lifecycle; prompts remain unique")
                _write_state(state_path, state)
            except Exception as exc:
                if isinstance(raw, dict):
                    output.write_text(json.dumps({"raw": raw, "error": f"{type(exc).__name__}: {exc}"}, indent=2) + "\n")
                state["failures"].append({"key": key, "error": f"{type(exc).__name__}: {exc}"})
                _write_state(state_path, state)
                raise
            following = _next_cell(state)
            if stage in COLD_BOOT_STAGES and following is not None and following[0] in COLD_BOOT_STAGES:
                state["status"] = "ready_for_restart"
                state["next_command"] = (f"stop the supervisor-owned server; start a fresh {args.profile} server with new "
                    "evidence and scheduler paths; rerun this command for " + f"{following[0]}/r{following[1]}")
                _write_state(state_path, state)
                print(json.dumps({"status": state["status"], "accepted": key,
                                  "next_cell": f"{following[0]}/r{following[1]}",
                                  "next_command": state["next_command"]}))
                return RESTART_EXIT
    state["status"] = "complete"
    state["next_command"] = "start the other profile, or import the C2/C3 comparison if both are complete"
    _write_state(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
