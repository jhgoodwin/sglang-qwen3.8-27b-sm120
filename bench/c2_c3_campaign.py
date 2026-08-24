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

    def _messages(self, text: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": text}]

    def _probe(self, text: str) -> int:
        if len(self.calls) >= self.max_calls:
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
            response_path.write_text(json.dumps(response, indent=2) + "\n")
            record["response_artifact"] = str(response_path)
        else:
            record["response"] = response
        self.calls.append(record)
        return count

    def build(self, target: int) -> tuple[str, dict[str, Any]]:
        if type(target) is not int or target <= 0:
            raise ValueError("target token count must be positive")
        # The server's chat template remains the source of truth.  The coarse
        # unit is intentionally multi-token; correction uses independent
        # suffix candidates and never assumes one repetition equals one token.
        unit = " qwen38-campaign-filler"
        base = "Return the requested result."
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
                             "calls": list(self.calls), "algorithm": "bounded-coarse-plus-tokenized-suffix"}
                    return text, proof
        raise RuntimeError(f"failed closed: no exact prompt count {target} after {len(self.calls)} calls")

    def prove(self, text: str, target: int) -> dict[str, Any]:
        """Re-tokenize the exact generation messages immediately before use."""
        observed = self._probe(text)
        if observed != target:
            raise RuntimeError(f"prompt drift: expected {target}, observed {observed}")
        return {"target": target, "observed": observed,
                "messages_sha256": _messages_hash(self._messages(text)),
                "call": self.calls[-1]}


def _messages_hash(messages: list[dict[str, str]]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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
        prompt = prompts[index % len(prompts)] if prompts else "Reply briefly."
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
    if fractions and min(fractions) < minimum:
        raise RuntimeError(f"free VRAM gate failed: minimum {min(fractions):.4f} < {minimum:.4f}")


def short_warmup(base_url: str, model: str, artifact: pathlib.Path, timeout: float = 180) -> str:
    """Issue exactly one bounded stream before PID bootstrap."""
    rid = f"campaign-warmup-{time.time_ns()}-{hashlib.sha256(str(artifact).encode()).hexdigest()[:10]}"
    spec = {"model": model, "messages": [{"role": "user", "content": "Reply with one short word."}],
            "max_tokens": 8, "stream": True, "reasoning_effort": "medium"}
    request = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(spec).encode(), headers={"content-type": "application/json",
                                     "x-request-id": rid}, method="POST")
    rows: list[str] = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        while True:
            line = response.readline()
            if not line:
                break
            rows.append(line.decode("utf-8", errors="replace"))
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps({"request_id": rid, "request": spec, "raw_sse": rows}, indent=2) + "\n")
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
                        "repetitions": 3, "shape": TARGETS.get(stage, {"admit": runtime["max_running_requests"], "excess": 1})}
                       for stage in STAGES],
            "next_command": "run after supervisor-owned server boot, evidence collection, and PID bootstrap"}


def _load_state(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema": "qwen38.c2-c3-campaign-state", "version": 1, "accepted": {}, "failures": []}
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema") != "qwen38.c2-c3-campaign-state":
        raise ValueError("invalid campaign state")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest", type=pathlib.Path, default=MANIFEST)
    common.add_argument("--profile", required=True, choices=("c2", "c3"))
    common.add_argument("--base-url", default="http://127.0.0.1:11447")
    common.add_argument("--artifact-root", type=pathlib.Path, required=True)
    sub.add_parser("plan", parents=[common])
    run = sub.add_parser("run", parents=[common])
    run.add_argument("--server-evidence", type=pathlib.Path, required=True)
    run.add_argument("--scheduler-events", type=pathlib.Path, required=True)
    run.add_argument("--server-pid", default="auto")
    run.add_argument("--sample-interval", type=float, default=1.0)
    run.add_argument("--timeout", type=float, default=None,
                     help="per-cell stream timeout; defaults to 1800s, 2100s for C/D, 2700s for E")
    args = parser.parse_args(argv)
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
    server = json.loads(args.server_evidence.read_text())
    observed_args = server.get("observed_server_args", {})
    model = observed_args.get("model_path")
    if not isinstance(model, str) or not model:
        raise SystemExit("BLOCKED: server evidence lacks observed model_path")
    if not args.server_evidence.is_file() or not args.scheduler_events.is_file():
        raise SystemExit("BLOCKED: evidence paths disappeared")
    warmup_artifact = args.artifact_root / "warmup" / "short-stream.json"
    if not state.get("warmup_accepted"):
        warmup_id = short_warmup(args.base_url, model, warmup_artifact, timeout=180)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                if not _scheduler_has_request(args.scheduler_events, warmup_id):
                    raise ValueError("warmup scheduler event not observed")
                c2_c3_runner.bootstrap_server_pid(args.scheduler_events)
                state["warmup_accepted"] = str(warmup_artifact)
                state_path.write_text(json.dumps(state, indent=2) + "\n")
                break
            except ValueError:
                time.sleep(.5)
        if not state.get("warmup_accepted"):
            raise SystemExit("BLOCKED: warmup completed but scheduler identity was not observed")
    # Prompt generation is performed per shape, and all tokenize calls are
    # retained before any generation is submitted.
    builder = ExactPromptBuilder(args.base_url, model, artifact_dir=args.artifact_root)
    prompts_by_target: dict[int, str] = {}
    proofs: list[dict[str, Any]] = []
    prompt_dir = args.artifact_root / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for target in sorted({value[0] for value in TARGETS.values()}):
        prompt_path = prompt_dir / f"{target}.txt"
        if prompt_path.is_file():
            prompts_by_target[target] = prompt_path.read_text()
            proof = {"target": target, "observed": "persisted", "messages_sha256": _messages_hash(builder._messages(prompts_by_target[target]))}
        else:
            prompts_by_target[target], proof = builder.build(target)
            prompt_path.write_text(prompts_by_target[target])
        proofs.append(proof)
    (args.artifact_root / "prompt-tokenize-proofs.json").write_text(
        json.dumps({"calls": builder.calls, "proofs": proofs}, indent=2) + "\n")
    accepted = state.get("accepted", {})
    for stage in STAGES:
        if stage != STAGES[0] and not all(f"{dependency}/r{rep}" in accepted
                                          for dependency in next(x["depends_on"] for x in manifest["stages"] if x["id"] == stage)
                                          for rep in range(1, 4)):
            raise SystemExit(f"BLOCKED: dependency for {stage} has not completed accepted repetitions")
        for repetition in range(1, 4):
            key = f"{stage}/r{repetition}"
            if key in state.get("accepted", {}):
                continue
            input_tokens = TARGETS.get(stage, (None, None))[0]
            if input_tokens is not None:
                proof = builder.prove(prompts_by_target[input_tokens], input_tokens)
                proofs.append(proof)
                (args.artifact_root / "prompt-tokenize-proofs.json").write_text(
                    json.dumps({"calls": builder.calls, "proofs": proofs}, indent=2) + "\n")
            specs = request_shape(stage, args.profile, repetition,
                                  [prompts_by_target.get(input_tokens, "Reply briefly.")], manifest, model=model)
            specs["base_url"] = args.base_url
            spec_path = args.artifact_root / "specs" / f"{args.profile}-{stage}-r{repetition}.json"
            spec_path.parent.mkdir(parents=True, exist_ok=True)
            spec_path.write_text(json.dumps(specs, indent=2) + "\n")
            output = args.artifact_root / "raw" / f"{args.profile}-{stage}-r{repetition}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            raw = None
            try:
                pid = c2_c3_runner.bootstrap_server_pid(args.scheduler_events)[0] if args.server_pid == "auto" else int(args.server_pid)
                timeout = args.timeout or (2700 if stage == "E-four-arrival-queue" else (2100 if stage in ("C-max-output-decode", "D-combined-boundary-safe") else 1800))
                raw = c2_c3_runner.run_concurrent(specs, server, args.scheduler_events, server_pid=pid,
                                                   timeout=timeout, sample_interval=args.sample_interval)
                free_vram_gate(raw)
                imported = c2_c3_importer.validate_and_import(raw)
                output.write_text(json.dumps({"raw": raw, "imported": imported}, indent=2) + "\n")
                if not imported.get("accepted"):
                    state["failures"].append({"key": key, "artifact": str(output), "errors": imported.get("errors", [])})
                    state_path.write_text(json.dumps(state, indent=2) + "\n")
                    raise SystemExit(f"BLOCKED: importer rejected {key}; artifact preserved at {output}")
                state["accepted"][key] = str(output)
                accepted = state["accepted"]
                state_path.write_text(json.dumps(state, indent=2) + "\n")
            except Exception as exc:
                if isinstance(raw, dict):
                    output.write_text(json.dumps({"raw": raw, "error": f"{type(exc).__name__}: {exc}"}, indent=2) + "\n")
                state["failures"].append({"key": key, "error": f"{type(exc).__name__}: {exc}"})
                state_path.write_text(json.dumps(state, indent=2) + "\n")
                raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
