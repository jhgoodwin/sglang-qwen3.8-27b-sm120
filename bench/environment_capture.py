#!/usr/bin/env python3
"""Capture reproducible host and immutable-input identity without side effects."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any, Callable

Command = Callable[[list[str]], tuple[int, str, str]]


def _run(argv: list[str]) -> tuple[int, str, str]:
    try:
        p = subprocess.run(argv, text=True, capture_output=True, timeout=15, check=False)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except (FileNotFoundError, OSError) as exc:
        return 127, "", str(exc)
    except subprocess.TimeoutExpired as exc:
        return 124, (exc.stdout or "").strip() if isinstance(exc.stdout, str) else "", "timeout"


def command_result(argv: list[str], runner: Command = _run) -> dict[str, Any]:
    code, out, err = runner(argv)
    if code == 0:
        status = "available"
    elif code == 127:
        status = "unavailable"
    else:
        status = "error"
    result: dict[str, Any] = {"status": status, "command": argv, "returncode": code}
    if out:
        result["stdout"] = out
    if err:
        result["stderr"] = err[:1000]
    return result


def _commands(runner: Command) -> dict[str, Any]:
    specs = {
        "gpu_inventory": ["nvidia-smi", "--query-gpu=name,memory.total,vbios_version,clocks.current.graphics,clocks.current.memory,persistence_mode,ecc.mode.current,power.limit,temperature.gpu", "--format=csv,noheader,nounits"],
        "topology": ["nvidia-smi", "topo", "-m"],
        "p2p": ["nvidia-smi", "topo", "-p2p", "r"],
        "gpu_processes": ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"],
        "gpu_display": ["nvidia-smi", "--query-gpu=index,display_active,display_mode", "--format=csv,noheader"],
        "driver": ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        "pci_tree": ["lspci", "-tv"],
        "numa": ["numactl", "-H"],
        "cpu": ["lscpu"],
        "ram": ["free", "-b"],
        "docker": ["docker", "version", "--format", "{{json .}}"],
        "nvidia_container_toolkit": ["nvidia-container-cli", "--version"],
        "kernel": ["uname", "-a"],
        "ports": ["ss", "-ltnp"],
    }
    return {name: command_result(argv, runner) for name, argv in specs.items()}


def _disk(path: Path) -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(path)
        return {"status": "available", "path": str(path), "bytes_total": usage.total, "bytes_free": usage.free, "bytes_used": usage.used}
    except OSError as exc:
        return {"status": "error", "path": str(path), "error": str(exc)}


def _snapshot(path: Path, repo: str | None, revision: str | None, full_hash: bool) -> dict[str, Any]:
    if not path.exists() or not path.is_dir():
        return {"status": "unavailable", "reason": "explicit snapshot path does not exist", "path": str(path)}
    if path.parent.name != "snapshots":
        raise ValueError("--hf-snapshot-path must be a canonical Hugging Face .../snapshots/<revision> directory")
    actual_revision = path.name
    if revision != actual_revision:
        raise ValueError(f"--hf-revision {revision!r} does not match snapshot directory revision {actual_revision!r}")
    files = sorted(p for p in path.rglob("*") if p.is_file())
    entries = []
    total = 0
    for p in files:
        size = p.stat().st_size
        total += size
        item: dict[str, Any] = {"path": str(p.relative_to(path)), "bytes": size}
        # HF cache snapshots normally symlink files to blobs whose basename is the ETag.
        if p.is_symlink():
            item["etag"] = p.resolve().name
        else:
            item["identity"] = "unresolved_without_full_hash"
        if full_hash:
            h = hashlib.sha256()
            with p.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(chunk)
            item["sha256"] = h.hexdigest()
        entries.append(item)
    shard_files = [p for p in files if p.name.endswith(".safetensors") and not p.name.endswith(".index.json")]
    indexes = [p for p in files if p.name.endswith(".safetensors.index.json")]
    index_info: dict[str, Any] = {"status": "absent"}
    if indexes:
        try:
            payload = json.loads(indexes[0].read_text())
            refs = sorted(set(payload.get("weight_map", {}).values()))
            actual_names = sorted(p.name for p in shard_files)
            index_info = {"status": "valid" if refs == actual_names else "error", "path": str(indexes[0].relative_to(path)), "weight_map_entries": len(payload.get("weight_map", {})), "referenced_shards": refs, "actual_shards": actual_names}
            if index_info["status"] == "error":
                raise ValueError("safetensors index shard references do not match snapshot files")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("safetensors index"):
                raise
            index_info = {"status": "error", "error": str(exc)}
    known = sum(1 for item in entries if "etag" in item or "sha256" in item)
    return {"status": "available", "repo": repo, "revision": actual_revision, "path": str(path), "file_count": len(files), "total_bytes": total, "weight_shard_count": len(shard_files), "weight_shard_bytes": sum(p.stat().st_size for p in shard_files), "safetensors_index": index_info, "identity_coverage": {"known_files": known, "unresolved_files": len(files) - known, "status": "complete" if known == len(files) else "partial"}, "files": entries, "full_hash": full_hash}


def _compatibility() -> dict[str, Any]:
    packages = {"sglang": "sglang", "flashinfer": "flashinfer", "pytorch": "torch", "sgl-kernel": "sgl-kernel"}
    result: dict[str, Any] = {}
    for label, package in packages.items():
        try:
            result[label] = {"status": "available", "version": importlib.metadata.version(package)}
        except importlib.metadata.PackageNotFoundError:
            result[label] = {"status": "unavailable", "reason": "package not installed"}
    try:
        import torch
        result["cuda"] = {"status": "available" if torch.version.cuda else "unavailable", "version": torch.version.cuda}
    except (ImportError, AttributeError):
        result["cuda"] = {"status": "unavailable", "reason": "PyTorch unavailable"}
    return result


def _locks(paths: list[Path]) -> dict[str, Any]:
    result = {}
    for path in paths:
        if not path.exists():
            result[str(path)] = {"status": "unavailable", "reason": "lock file not found"}
            continue
        try:
            result[str(path)] = {"status": "available", "values": json.loads(path.read_text())}
        except (OSError, json.JSONDecodeError) as exc:
            result[str(path)] = {"status": "error", "error": str(exc)}
    return result


def capture(output: Path, *, snapshot: Path | None = None, repo: str | None = None, revision: str | None = None, full_hash: bool = False, disk_paths: list[Path] | None = None, lock_paths: list[Path] | None = None, runner: Command = _run) -> dict[str, Any]:
    if full_hash and snapshot is None:
        raise ValueError("--full-hash requires --hf-snapshot-path")
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    data: dict[str, Any] = {
        "schema": "environment.v1",
        "captured_at_utc": now,
        "provenance": {"hostname": socket.gethostname(), "python": platform.python_version(), "cwd": os.getcwd()},
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "commands": _commands(runner),
        "compatibility": _compatibility(),
        "disk": {str(path): _disk(path) for path in (disk_paths or [Path.cwd()])},
        "locks": _locks(lock_paths or [Path("source.lock.json"), Path("stack.lock.json")]),
        "inputs": {"source_lock": "UNRESOLVED until explicitly verified", "image": "UNRESOLVED until explicitly verified"},
    }
    gpu_status = data["commands"]["gpu_inventory"]["status"]
    if gpu_status != "available":
        data["capture_environment_error"] = {"status": "present", "scope": "capture environment", "reason": "NVIDIA driver inventory was unavailable or returned an error; this does not establish a host hardware failure"}
    if snapshot is not None:
        if not repo or not revision:
            raise ValueError("--hf-repo and --hf-revision are required with --hf-snapshot-path")
        data["inputs"]["huggingface_snapshot"] = _snapshot(snapshot, repo, revision, full_hash)
    else:
        data["inputs"]["huggingface_snapshot"] = {"status": "unavailable", "reason": "no explicit completed snapshot path supplied"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    note = output.with_name("source-compatibility.md")
    snap = data["inputs"]["huggingface_snapshot"]
    note.write_text(
        "# Source and compatibility capture\n\n"
        "This note is generated by `bench/environment_capture.py`; it records identity only and never downloads or starts a service.\n\n"
        f"- Capture: `{data['captured_at_utc']}`\n"
        f"- Hugging Face snapshot: `{snap.get('status')}` ({snap.get('repo', 'not supplied')} @ {snap.get('revision', 'not supplied')})\n"
        "- Source lock: `UNRESOLVED` until independently verified.\n"
        "- Container image: `UNRESOLVED` until an immutable digest is supplied and verified.\n"
        "- Missing NVIDIA/Docker tools are capability limitations of the capture environment, not host hardware failures.\n"
        f"- NVIDIA inventory capture environment error: `{data.get('capture_environment_error', {}).get('status', 'none')}`.\n"
        "- Full file hashing is opt-in with `--full-hash`; symlink blob names are recorded as ETags when available.\n"
    )
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("environment.json"))
    parser.add_argument("--hf-snapshot-path", type=Path)
    parser.add_argument("--hf-repo")
    parser.add_argument("--hf-revision")
    parser.add_argument("--full-hash", action="store_true", help="explicitly hash every snapshot file (expensive)")
    parser.add_argument("--disk-path", action="append", type=Path, help="path whose filesystem capacity is recorded (repeatable)")
    parser.add_argument("--lock-path", action="append", type=Path, help="lock JSON path to capture (repeatable)")
    args = parser.parse_args(argv)
    try:
        data = capture(args.output, snapshot=args.hf_snapshot_path, repo=args.hf_repo, revision=args.hf_revision, full_hash=args.full_hash, disk_paths=args.disk_path, lock_paths=args.lock_path)
    except ValueError as exc:
        parser.error(str(exc))
    available = sum(1 for r in data["commands"].values() if r["status"] == "available")
    unavailable = sum(1 for r in data["commands"].values() if r["status"] == "unavailable")
    print(f"captured {args.output}: {available} available, {unavailable} unavailable; missing host capabilities are recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
