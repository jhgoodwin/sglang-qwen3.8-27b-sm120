#!/usr/bin/env python3
"""Join flattened /server_info facts to independent immutable launch evidence."""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import urllib.request
from typing import Any

from bench.c2_c3_importer import EXPECTED_IDENTITIES, PROFILE_RUNTIME, REQUIRED_SERVER_ARGS

REPO = pathlib.Path(__file__).resolve().parents[1]
PROVENANCE_SCHEMA = "qwen38.c2-c3-launch-provenance"
OBSERVED_FIELDS = tuple(REQUIRED_SERVER_ARGS) + (
    "model_path", "speculative_draft_model_path", "max_running_requests", "max_mamba_cache_size",
)


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive measured integer")
    return value


def _runtime_block(response: dict[str, Any]) -> dict[str, Any]:
    states = response.get("internal_states")
    if not isinstance(states, list) or len(states) != 1 or not isinstance(states[0], dict):
        raise ValueError("/server_info must contain exactly one TP1 scheduler internal state")
    block = states[0].get("c2c3_evidence")
    if not isinstance(block, dict):
        raise ValueError("/server_info lacks internal_states[0].c2c3_evidence")
    return block


def _profile_document(profile: str) -> dict[str, Any]:
    value = json.loads((REPO / "profiles.json").read_text()).get("profiles", {}).get(profile)
    if not isinstance(value, dict):
        raise ValueError(f"profiles.json lacks {profile}")
    return value


def _expected_overlay() -> tuple[str, str]:
    overlay = json.loads((REPO / "source.lock.json").read_text())["runtime_variants"]["c2-c3-evidence-overlay"]
    return overlay["image_digest"], overlay["sglang_revision"]


def _command_flags(command: Any) -> dict[str, str | bool]:
    if not isinstance(command, list) or not command or not all(isinstance(x, str) and x for x in command):
        raise ValueError("container command must be a non-empty string list")
    result: dict[str, str | bool] = {}
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
        if flag in result:
            raise ValueError(f"duplicate container command flag: {flag}")
        result[flag] = value
        index += 1
    return result


def _mount_host_path(mounts: Any, container_path: str) -> str:
    candidates: list[tuple[int, str]] = []
    for mount in mounts if isinstance(mounts, list) else []:
        if not isinstance(mount, dict):
            continue
        source, destination = mount.get("Source"), mount.get("Destination")
        if (isinstance(source, str) and isinstance(destination, str) and
                (container_path == destination or container_path.startswith(destination.rstrip("/") + "/"))):
            suffix = container_path[len(destination):].lstrip("/")
            candidates.append((len(destination), str(pathlib.PurePosixPath(source) / suffix)))
    if not candidates:
        raise ValueError(f"no Docker mount backs observed path {container_path}")
    return max(candidates)[1]


def build_launch_provenance(inspect: dict[str, Any], *, profile: str,
                            hardware_identity: str, raw_reference: str) -> dict[str, Any]:
    """Validate and normalize one raw docker-container-inspect object."""
    if profile not in PROFILE_RUNTIME:
        raise ValueError("profile must be c2 or c3")
    if not isinstance(hardware_identity, str) or not hardware_identity.startswith("GPU-"):
        raise ValueError("hardware identity must be an observed NVIDIA GPU UUID")
    expected_digest, expected_revision = _expected_overlay()
    if not isinstance(raw_reference, str) or not raw_reference:
        raise ValueError("raw Docker inspect reference is required")
    if inspect.get("Name") != "/qwen3.8-27b-sglang":
        raise ValueError("Docker container name differs from production service name")
    if not isinstance(inspect.get("Id"), str) or not inspect["Id"]:
        raise ValueError("Docker container identity is missing")
    config = inspect.get("Config") if isinstance(inspect.get("Config"), dict) else {}
    configured_ref, image_digest = config.get("Image"), inspect.get("Image")
    labels, command = config.get("Labels") or {}, config.get("Cmd")
    if image_digest != expected_digest or not isinstance(configured_ref, str) or not configured_ref.endswith("@" + expected_digest):
        raise ValueError("Docker image does not match locked evidence overlay")
    if _profile_document(profile).get("image") != configured_ref:
        raise ValueError("Docker image does not match runnable profile")
    if labels.get("org.opencontainers.image.revision") != expected_revision:
        raise ValueError("Docker image source-revision label differs from lock")
    flags = _command_flags(command)
    if "--mamba-full-memory-ratio" in flags or "--disable-cuda-graph" in flags:
        raise ValueError("initial profile contains a forbidden launch factor")
    expected_flags = {
        "--context-length": "262144", "--tp-size": "1", "--kv-cache-dtype": "fp8_e4m3",
        "--attention-backend": "flashinfer", "--chunked-prefill-size": "2048",
        "--mamba-ssm-dtype": "float32", "--mem-fraction-static": "0.85",
        "--mamba-radix-cache-strategy": "extra_buffer_lazy", "--speculative-algorithm": "DFLASH",
        "--speculative-num-draft-tokens": "8",
        "--max-running-requests": str(PROFILE_RUNTIME[profile]["max_running_requests"]),
        "--max-mamba-cache-size": str(PROFILE_RUNTIME[profile]["max_mamba_cache_size"]),
    }
    for flag, expected in expected_flags.items():
        if flags.get(flag) != expected:
            raise ValueError(f"Docker launch argument {flag} differs from profile")
    paths = {
        "model_snapshot": _mount_host_path(inspect.get("Mounts"), str(flags.get("--model-path", ""))),
        "draft_model_snapshot": _mount_host_path(inspect.get("Mounts"), str(flags.get("--speculative-draft-model-path", ""))),
    }
    for name, host_path in paths.items():
        if host_path != EXPECTED_IDENTITIES[name]:
            raise ValueError(f"Docker mount provenance differs from locked {name}")
    return {
        "schema": PROVENANCE_SCHEMA, "version": 1, "source": "docker_inspect", "profile": profile,
        "container_id": inspect.get("Id"), "container_name": inspect.get("Name", "").lstrip("/"),
        "image_ref": configured_ref, "image_digest": image_digest, "source_revision": expected_revision,
        "container_command": command, "model_snapshot": paths["model_snapshot"],
        "draft_model_snapshot": paths["draft_model_snapshot"], "hardware_identity": hardware_identity,
        "raw_reference": raw_reference,
    }


def capture_provenance(container: str, *, profile: str, output: pathlib.Path,
                       raw_output: pathlib.Path, gpu: str) -> None:
    raw = subprocess.run(["docker", "container", "inspect", container], check=True,
                         stdout=subprocess.PIPE, text=True).stdout
    raw_output.write_text(raw)
    values = json.loads(raw)
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError("docker inspect must return exactly one container")
    uuid = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=uuid", "--format=csv,noheader,nounits"],
                          check=True, stdout=subprocess.PIPE, text=True).stdout.strip()
    result = build_launch_provenance(values[0], profile=profile, hardware_identity=uuid,
                                     raw_reference=str(raw_output))
    output.write_text(json.dumps(result, indent=2) + "\n")


def build_evidence(response: dict[str, Any], *, provenance: dict[str, Any],
                   profile: str, endpoint: str, raw_reference: str) -> dict[str, Any]:
    if profile not in PROFILE_RUNTIME:
        raise ValueError("profile must be c2 or c3")
    if provenance.get("schema") != PROVENANCE_SCHEMA or provenance.get("profile") != profile:
        raise ValueError("launch provenance schema/profile mismatch")
    expected_digest, expected_revision = _expected_overlay()
    locked = {"image_digest": expected_digest, "source_revision": expected_revision,
              "model_snapshot": EXPECTED_IDENTITIES["model_snapshot"],
              "draft_model_snapshot": EXPECTED_IDENTITIES["draft_model_snapshot"]}
    for field, expected in locked.items():
        if provenance.get(field) != expected:
            raise ValueError(f"launch provenance drift: {field}")
    if provenance.get("source") != "docker_inspect" or not provenance.get("raw_reference"):
        raise ValueError("launch provenance lacks independent source/reference")
    if provenance.get("container_name") != "qwen3.8-27b-sglang":
        raise ValueError("launch provenance does not name the production service")
    command_flags = _command_flags(provenance.get("container_command"))
    if "sglang.launch_server" not in provenance.get("container_command", []):
        raise ValueError("launch provenance is not the SGLang server command")
    if "--mamba-full-memory-ratio" in command_flags or "--disable-cuda-graph" in command_flags:
        raise ValueError("initial profile contains a forbidden launch factor")

    observed_args: dict[str, Any] = {}
    for field in OBSERVED_FIELDS:
        if field not in response:
            raise ValueError(f"/server_info lacks flattened field {field}")
        observed_args[field] = response[field]
    expected_args = dict(REQUIRED_SERVER_ARGS)
    expected_args.update({"max_running_requests": PROFILE_RUNTIME[profile]["max_running_requests"],
                          "max_mamba_cache_size": PROFILE_RUNTIME[profile]["max_mamba_cache_size"]})
    for field, expected in expected_args.items():
        if observed_args.get(field) != expected:
            raise ValueError(f"observed /server_info field {field} differs from profile")
    launch_fields = {
        "--model-path": "model_path", "--speculative-draft-model-path": "speculative_draft_model_path",
        "--context-length": "context_length", "--tp-size": "tp_size", "--kv-cache-dtype": "kv_cache_dtype",
        "--attention-backend": "attention_backend", "--chunked-prefill-size": "chunked_prefill_size",
        "--mamba-ssm-dtype": "mamba_ssm_dtype", "--mem-fraction-static": "mem_fraction_static",
        "--mamba-radix-cache-strategy": "mamba_radix_cache_strategy",
        "--speculative-algorithm": "speculative_algorithm",
        "--speculative-num-draft-tokens": "speculative_num_draft_tokens",
        "--max-running-requests": "max_running_requests", "--max-mamba-cache-size": "max_mamba_cache_size",
    }
    for flag, field in launch_fields.items():
        if command_flags.get(flag) != str(observed_args[field]):
            raise ValueError(f"launch argument {flag} differs from observed /server_info {field}")
    for field, revision in (("model_path", "319f741cce68d7914884900c138a1fbb70a42f30"),
                            ("speculative_draft_model_path", "dedf8df68adfb1afeaf7b7480c0a0243108177b4")):
        if not isinstance(observed_args[field], str) or not observed_args[field].endswith("/snapshots/" + revision):
            raise ValueError(f"observed /server_info field {field} differs from locked snapshot")

    observed = _runtime_block(response)
    capacity = observed.get("resolved_capacity")
    if not isinstance(capacity, dict):
        raise ValueError("resolved_capacity must be observed runtime evidence")
    for name, value in (("max_running_requests", PROFILE_RUNTIME[profile]["max_running_requests"]),
                        ("context_length", 262144), ("tp_size", 1),
                        ("max_mamba_cache_size", PROFILE_RUNTIME[profile]["max_mamba_cache_size"])):
        if capacity.get(name) != value:
            raise ValueError(f"observed resolved capacity {name} differs from profile")
    pools, graphs = observed.get("memory_pools"), observed.get("cuda_graphs")
    if not isinstance(pools, dict) or not isinstance(graphs, dict):
        raise ValueError("memory_pools and cuda_graphs must be observed")
    for name in ("kv_cache", "mamba_state_cache", "dflash_intermediate"):
        pool = pools.get(name)
        if not isinstance(pool, dict):
            raise ValueError(f"missing observed pool {name}")
        _positive_int(pool.get("bytes"), f"memory_pools.{name}.bytes")
    if pools["kv_cache"].get("dtype") != "fp8_e4m3":
        raise ValueError("observed KV dtype differs from profile")
    if (pools["mamba_state_cache"].get("dtype") != "float32" or
            pools["mamba_state_cache"].get("slots") != PROFILE_RUNTIME[profile]["max_mamba_cache_size"]):
        raise ValueError("observed Mamba pool differs from profile")
    if pools["dflash_intermediate"].get("states") != 8:
        raise ValueError("DFlash evidence must be tied to eight configured states")
    batches = graphs.get("captured_batch_sizes")
    if (graphs.get("enabled") is not True or not isinstance(batches, list) or 1 not in batches or
            PROFILE_RUNTIME[profile]["max_running_requests"] not in batches):
        raise ValueError("CUDA graph captures do not cover profile occupancy")
    _positive_int(graphs.get("memory_bytes"), "cuda_graphs.memory_bytes")

    identities = {"image_digest": provenance["image_digest"], "source_revision": provenance["source_revision"],
                  "model_snapshot": provenance["model_snapshot"],
                  "draft_model_snapshot": provenance["draft_model_snapshot"],
                  "recipe_identity": EXPECTED_IDENTITIES["recipe_identity"],
                  "hardware_identity": provenance.get("hardware_identity")}
    if not isinstance(identities["hardware_identity"], str) or not identities["hardware_identity"].startswith("GPU-"):
        raise ValueError("launch provenance lacks observed GPU UUID")
    return {
        "identities": identities, "observed_server_args": observed_args,
        "resolved_capacity": {key: capacity[key] for key in ("max_running_requests", "context_length",
            "max_total_num_tokens", "tp_size", "max_mamba_cache_size") if key in capacity},
        "campaign_request_limits": {"max_output_tokens": 131072},
        "launch_metadata": {"planned_port": PROFILE_RUNTIME[profile]["planned_port"], "observed_endpoint": endpoint},
        "launch_provenance": {**provenance, "artifact_reference": provenance.get("artifact_reference")},
        "memory_pools": pools, "cuda_graphs": graphs, "raw_server_info_reference": raw_reference,
    }


def collect(url: str, *, profile: str, provenance_path: pathlib.Path,
            output: pathlib.Path, raw_output: pathlib.Path) -> None:
    endpoint = url.rstrip("/") + "/server_info"
    with urllib.request.urlopen(endpoint, timeout=10) as response:
        raw = response.read().decode("utf-8")
    raw_output.write_text(raw + "\n")
    provenance = json.loads(provenance_path.read_text())
    provenance["artifact_reference"] = str(provenance_path)
    evidence = build_evidence(json.loads(raw), provenance=provenance, profile=profile,
                              endpoint=url, raw_reference=str(raw_output))
    output.write_text(json.dumps(evidence, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture-provenance")
    capture.add_argument("--container", required=True); capture.add_argument("--profile", choices=sorted(PROFILE_RUNTIME), required=True)
    capture.add_argument("--gpu", default="0"); capture.add_argument("--output", type=pathlib.Path, required=True)
    capture.add_argument("--raw-output", type=pathlib.Path, required=True)
    runtime = sub.add_parser("collect")
    runtime.add_argument("--url", required=True); runtime.add_argument("--profile", choices=sorted(PROFILE_RUNTIME), required=True)
    runtime.add_argument("--launch-provenance", type=pathlib.Path, required=True)
    runtime.add_argument("--output", type=pathlib.Path, required=True); runtime.add_argument("--raw-output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "capture-provenance":
        capture_provenance(args.container, profile=args.profile, output=args.output, raw_output=args.raw_output, gpu=args.gpu)
    else:
        collect(args.url, profile=args.profile, provenance_path=args.launch_provenance,
                output=args.output, raw_output=args.raw_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
