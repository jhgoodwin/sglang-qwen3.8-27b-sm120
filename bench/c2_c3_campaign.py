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
                 tokenize: Callable[[str, dict[str, Any], float], tuple[int, dict[str, Any]]] = post_json):
        self.endpoint = endpoint.rstrip("/") + "/v1/tokenize"
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_calls = max_calls
        self.timeout = timeout
        self.tokenize = tokenize
        self.calls: list[dict[str, Any]] = []

    def _messages(self, text: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": text}]

    def _probe(self, text: str) -> int:
        if len(self.calls) >= self.max_calls:
            raise RuntimeError(f"tokenize call bound {self.max_calls} exceeded")
        body = {"model": self.model, "messages": self._messages(text),
                "reasoning_effort": self.reasoning_effort}
        try:
            status, response = self.tokenize(self.endpoint, body, self.timeout)
        except Exception as exc:
            self.calls.append({"request": body, "error": f"{type(exc).__name__}: {exc}"})
            raise
        count = token_count(response)
        self.calls.append({"request": body, "status": status, "response": response,
                           "token_count": count})
        return count

    def build(self, target: int) -> tuple[str, dict[str, Any]]:
        if type(target) is not int or target <= 0:
            raise ValueError("target token count must be positive")
        # Deliberately simple, deterministic UTF-8 text.  The server's chat
        # template remains the source of truth for all overhead.
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
        center = hi
        radius = min(4096, max(64, (hi - lo) * 2))
        candidates = [center + delta for delta in range(-radius, radius + 1)]
        # Prefer nearby values and then deterministic pseudo-random offsets to
        # catch local merge irregularities while retaining a hard call bound.
        candidates += [center + ((i * 7919) % (radius * 2 + 1)) - radius for i in range(64)]
        for chars in candidates:
            if chars < 0:
                continue
            if count_chars(chars) == target:
                text = base + unit * chars
                proof = {"target": target, "observed": target, "messages": self._messages(text),
                         "calls": list(self.calls), "algorithm": "bounded-binary-local-correction"}
                return text, proof
        raise RuntimeError(f"failed closed: no exact prompt count {target} after {len(self.calls)} calls")

    def prove(self, text: str, target: int) -> dict[str, Any]:
        """Re-tokenize the exact generation messages immediately before use."""
        observed = self._probe(text)
        if observed != target:
            raise RuntimeError(f"prompt drift: expected {target}, observed {observed}")
        return {"target": target, "observed": observed, "messages": self._messages(text),
                "call": self.calls[-1]}


def request_shape(stage: str, profile: str, repetition: int, prompts: list[str], manifest: dict[str, Any]) -> dict[str, Any]:
    spec: dict[str, Any] = {"schema": c2_c3_runner.SPEC_SCHEMA, "version": 1,
        "run_id": f"{profile}-{stage}-r{repetition}", "profile": profile,
        "stage": stage, "base_url": "", "model": manifest["runtime"]["model"].split("#")[-1],
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
    # Prompt generation is performed per shape, and all tokenize calls are
    # retained before any generation is submitted.
    builder = ExactPromptBuilder(args.base_url, manifest["runtime"]["model"])
    prompts_by_target: dict[int, str] = {}
    proofs: list[dict[str, Any]] = []
    for target in sorted({value[0] for value in TARGETS.values()}):
        prompts_by_target[target], proof = builder.build(target)
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
                                  [prompts_by_target.get(input_tokens, "Reply briefly.")], manifest)
            specs["base_url"] = args.base_url
            spec_path = args.artifact_root / "specs" / f"{args.profile}-{stage}-r{repetition}.json"
            spec_path.parent.mkdir(parents=True, exist_ok=True)
            spec_path.write_text(json.dumps(specs, indent=2) + "\n")
            output = args.artifact_root / "raw" / f"{args.profile}-{stage}-r{repetition}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            raw = None
            try:
                server = json.loads(args.server_evidence.read_text())
                pid = c2_c3_runner.bootstrap_server_pid(args.scheduler_events)[0] if args.server_pid == "auto" else int(args.server_pid)
                raw = c2_c3_runner.run_concurrent(specs, server, args.scheduler_events, server_pid=pid)
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
