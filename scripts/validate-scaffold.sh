#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
command -v python3 >/dev/null || { echo "missing python3" >&2; exit 1; }
python3 - "$repo" <<'PY'
import json, pathlib, re, sys
root = pathlib.Path(sys.argv[1])
for name in ("release.json", "source.lock.json", "stack.lock.json", "cache-schema.json", "profiles.json"):
    json.loads((root / name).read_text())
release = json.loads((root / "release.json").read_text())
assert release["model"] == "RadixArk/Qwen3.8-27B-NVFP4"
assert release["default_profile"] == "production"
assert re.fullmatch(r"v[1-9][0-9]*", release["cache_schema"])
profiles = json.loads((root / "profiles.json").read_text())["profiles"]
for name in ("production", "tp1-bf16-safe", "tp1-bf16-production", "tp2-bf16-safe", "tp2-bf16-production", "replica0", "replica1", "tp1-bf16-dspark-candidate", "tp2-bf16-dspark-candidate", "tp1-bf16-eagle-candidate", "tp2-bf16-eagle-candidate", "tp1-bf16-dflash-candidate"):
    assert name in profiles
assert "--speculative-algorithm" not in json.dumps(profiles["tp1-bf16-safe"])
for name in ("tp1-bf16-dspark-candidate", "tp2-bf16-dspark-candidate"):
    candidate = profiles[name]
    assert candidate["status"] == "unqualified_candidate"
    assert candidate["draft_model"] == "RadixArk/Qwen3.8-27B-DSpark"
    assert candidate["draft_model_dir_env"] == "DRAFT_MODEL_DIR"
    assert "--speculative-algorithm" in candidate["extra_args"]
    assert "--speculative-draft-model-path" in candidate["extra_args"]
assert profiles["tp1-bf16-dspark-candidate"]["port"] != profiles["tp2-bf16-dspark-candidate"]["port"]
for name in ("tp1-bf16-eagle-candidate", "tp2-bf16-eagle-candidate"):
    candidate = profiles[name]
    assert candidate["status"] == "unqualified_candidate"
    assert candidate["alias"] == name.replace("-eagle-candidate", "-safe")
    assert candidate["extra_args"][:2] == ["--speculative-algorithm", "EAGLE"]
    assert candidate["extra_args"][2:] == ["--speculative-num-steps", "3", "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", "4"] + (["--disable-custom-all-reduce"] if name.startswith("tp2-") else [])
assert profiles["tp1-bf16-eagle-candidate"]["port"] != profiles["tp2-bf16-eagle-candidate"]["port"]
dflash = profiles["tp1-bf16-dflash-candidate"]
assert dflash["status"] == "unqualified_candidate"
assert dflash["alias"] == "tp1-bf16-safe"
assert dflash["draft_model"] == "incoai/Qwen3.8-27B-DFlash2"
assert dflash["draft_model_dir_env"] == "DRAFT_MODEL_DIR"
assert dflash["extra_args"] == ["--speculative-algorithm", "DFLASH", "--speculative-draft-model-path", "/models/Qwen3.8-27B-DFlash2", "--speculative-num-draft-tokens", "8"]
current = {
    "tp1-nvfp4-cookbook-no-spec": (11444, "extra_buffer", "2.55", False),
    "tp1-nvfp4-dflash-cookbook-default": (11445, "extra_buffer", "6.63", True),
    "tp1-nvfp4-dflash-cookbook-lazy": (11446, "extra_buffer_lazy", "6.12", True),
}
for name, (port, strategy, ratio, speculative) in current.items():
    candidate = profiles[name]
    assert candidate["status"] == "unqualified_current_cookbook_candidate"
    assert candidate["model_repository"] == "RadixArk/Qwen3.8-27B-NVFP4"
    assert candidate["port"] == port and candidate["tp"] == 1 and candidate["gpus"] == "0"
    assert candidate["mamba_cache_strategy"] == strategy
    assert str(candidate["mamba_full_memory_ratio"]) == ratio
    assert ["--mamba-full-memory-ratio", ratio] == [candidate["extra_args"][candidate["extra_args"].index("--mamba-full-memory-ratio")], candidate["extra_args"][candidate["extra_args"].index("--mamba-full-memory-ratio") + 1]]
    max_idx = candidate["extra_args"].index("--max-running-requests")
    assert candidate["extra_args"][max_idx:max_idx + 2] == ["--max-running-requests", "1"]
    assert ("--speculative-algorithm" in candidate["extra_args"]) is speculative
assert json.loads((root / "source.lock.json").read_text())["runtime_variants"]["current-cookbook-qwen38-27b"]["recipe_main_revision"] == "d1af3c89233c475fc1bf11939d86787e6cddd58c"
assert "127.0.0.1" in (root / "RUN.md").read_text()
expected_defaults = {
    "production": ("0", 11436, "qwen3.8-27b-sglang"),
    "tp1-bf16-safe": ("0", 11436, "sglang-qwen38-27b-tp1-bf16-safe"),
    "tp1-bf16-production": ("0", 11436, "qwen3.8-27b-sglang"),
    "tp2-bf16-safe": ("0,1", 11436, "sglang-qwen38-27b-tp2-bf16-safe"),
    "tp2-bf16-production": ("0,1", 11436, "qwen3.8-27b-sglang"),
    "replica0": ("0", 11436, "sglang-qwen38-27b-replica0"),
    "replica1": ("1", 11437, "sglang-qwen38-27b-replica1"),
}
def resolve(name, trail=()):
    assert name not in trail, f"profile alias cycle: {' -> '.join(trail + (name,))}"
    profile = profiles[name]
    parent = profile.get("alias")
    if not parent:
        return dict(profile)
    resolved = resolve(parent, trail + (name,))
    resolved.update({key: value for key, value in profile.items() if key != "alias"})
    return resolved

for name, (gpus, port, container_name) in expected_defaults.items():
    resolved = resolve(name)
    assert resolved["gpus"] == gpus
    assert resolved["port"] == port
    assert resolved["container_name"] == container_name
serve = (root / "serve.sh").read_text()
production = resolve("production")
assert production["status"] == "qualified_native_context_c2"
assert production["default"] is True and production["evidence"] is False
assert "--max-running-requests" in production["extra_args"]
assert production["extra_args"][production["extra_args"].index("--max-running-requests") + 1] == "2"
assert production["extra_args"][production["extra_args"].index("--max-mamba-cache-size") + 1] == "8"
assert "profile=${PROFILE:-production}" in serve
assert "[[ \"$profile\" == production ]] && cache_profile=c2" in serve
assert "replica1" in serve
assert "container_port" in serve and "--port" in serve
assert "DRAFT_MODEL_DIR is required" in serve
assert "draft_model_container_path" in serve
assert "draft_model_mount_target" in serve
assert "tp1-bf16-eagle-candidate" in serve and "tp2-bf16-eagle-candidate" in serve
assert "--speculative-algorithm EAGLE" in serve
assert "--disable-custom-all-reduce" in serve
assert "tp1-bf16-dflash-candidate" in serve
print("scaffold static contract valid; runtime qualification: not run")
PY
bash -n "$repo/serve.sh"
test -x "$repo/serve.sh" || { echo "serve.sh must be executable" >&2; exit 1; }
