#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/model" "$tmp/hf" "$tmp/bin"
mkdir -p "$tmp/draft"
printf '%s\n' '#!/bin/sh' 'printf "GPU 0\\nGPU 1\\n"' > "$tmp/bin/nvidia-smi"
printf '%s\n' '#!/bin/sh' 'printf "%s\\n" "$*" > "$CAPTURE"' > "$tmp/bin/docker"
chmod +x "$tmp/bin/nvidia-smi" "$tmp/bin/docker"
valid="registry.example/qwen:tested@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

run_profile() {
  local profile=$1 expected_gpu=$2 expected_port=$3 expected_name=$4 expected_tp=$5
  CAPTURE="$tmp/$profile.args" env PATH="$tmp/bin:$PATH" MODEL_DIR="$tmp/model" \
    HF_CACHE_HUB="$tmp/hf" IMAGE="$valid" CACHE_DIR="$tmp/cache-$profile" \
    PROFILE="$profile" CAPTURE="$tmp/$profile.args" "$repo/serve.sh"
  grep -Fq -- "--name $expected_name" "$tmp/$profile.args"
  grep -Fq -- "--gpus device=$expected_gpu" "$tmp/$profile.args"
  grep -Fq -- "-p 127.0.0.1:$expected_port:8000" "$tmp/$profile.args"
  grep -Fq -- "--tp-size $expected_tp" "$tmp/$profile.args"
  grep -Fq -- '--port 8000' "$tmp/$profile.args"
  grep -Fq -- '--shm-size=16g' "$tmp/$profile.args"
  grep -Fq -- '--ulimit memlock=-1' "$tmp/$profile.args"
  grep -Fq -- "$tmp/model:/models/Qwen3.8-27B:ro" "$tmp/$profile.args"
  grep -Fq -- "$tmp/hf:/hf-cache:ro" "$tmp/$profile.args"
  grep -Fq -- '/cache/torch' "$tmp/$profile.args"
  grep -Fq -- '/cache/triton' "$tmp/$profile.args"
  grep -Fq -- '/cache/flashinfer' "$tmp/$profile.args"
  grep -Fq -- 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' "$tmp/$profile.args"
  ! grep -Fq -- '--speculative-' "$tmp/$profile.args"
}

run_profile tp1-bf16-safe 0 11436 sglang-qwen38-27b-tp1-bf16-safe 1
run_profile tp1-bf16-production 0 11436 sglang-qwen38-27b-tp1-bf16-production 1
run_profile tp2-bf16-safe 0,1 11436 sglang-qwen38-27b-tp2-bf16-safe 2
run_profile tp2-bf16-production 0,1 11436 sglang-qwen38-27b-tp2-bf16-production 2
run_profile replica0 0 11436 sglang-qwen38-27b-replica0 1
run_profile replica1 1 11437 sglang-qwen38-27b-replica1 1
echo "all named profile docker defaults passed"

run_candidate() {
  local profile=$1 expected_gpu=$2 expected_port=$3 expected_name=$4 expected_tp=$5
  CAPTURE="$tmp/$profile.args" env PATH="$tmp/bin:$PATH" MODEL_DIR="$tmp/model" HF_CACHE_HUB="$tmp/hf" \
    DRAFT_MODEL_DIR="$tmp/draft" IMAGE="$valid" CACHE_DIR="$tmp/cache-$profile" \
    PROFILE="$profile" CAPTURE="$tmp/$profile.args" "$repo/serve.sh"
  grep -Fq -- "--name $expected_name" "$tmp/$profile.args"
  grep -Fq -- "--gpus device=$expected_gpu" "$tmp/$profile.args"
  grep -Fq -- "-p 127.0.0.1:$expected_port:8000" "$tmp/$profile.args"
  grep -Fq -- "--tp-size $expected_tp" "$tmp/$profile.args"
  grep -Fq -- "-v $tmp/draft:/models/Qwen3.8-27B-DSpark:ro" "$tmp/$profile.args"
  grep -Fq -- '--speculative-algorithm DSPARK' "$tmp/$profile.args"
  grep -Fq -- '--speculative-draft-model-path /models/Qwen3.8-27B-DSpark' "$tmp/$profile.args"
  grep -Fq -- '--speculative-draft-attention-backend flashinfer' "$tmp/$profile.args"
  grep -Fq -- '--mamba-radix-cache-strategy extra_buffer_lazy' "$tmp/$profile.args"
}
run_candidate tp1-bf16-dspark-candidate 0 11438 sglang-qwen38-27b-tp1-bf16-dspark-candidate 1
run_candidate tp2-bf16-dspark-candidate 0,1 11439 sglang-qwen38-27b-tp2-bf16-dspark-candidate 2
if env PATH="$tmp/bin:$PATH" MODEL_DIR="$tmp/model" HF_CACHE_HUB="$tmp/hf" IMAGE="$valid" CACHE_DIR="$tmp/cache-missing-draft" PROFILE=tp1-bf16-dspark-candidate "$repo/serve.sh" 2>"$tmp/missing-draft"; then
  echo "candidate without DRAFT_MODEL_DIR unexpectedly accepted" >&2; exit 1
fi
grep -q 'DRAFT_MODEL_DIR is required' "$tmp/missing-draft"
echo "DSpark candidate args, isolated defaults, and draft requirement passed"

CAPTURE="$tmp/override.args" env PATH="$tmp/bin:$PATH" MODEL_DIR="$tmp/model" HF_CACHE_HUB="$tmp/hf" \
  IMAGE="$valid" CACHE_DIR="$tmp/cache-override" PROFILE=replica1 PORT=12345 CONTAINER_NAME=custom-qwen \
  GPU_LIST=0 TP_SIZE=1 CAPTURE="$tmp/override.args" "$repo/serve.sh"
grep -Fq -- '--name custom-qwen' "$tmp/override.args"
grep -Fq -- '-p 127.0.0.1:12345:8000' "$tmp/override.args"
grep -Fq -- '--gpus device=0' "$tmp/override.args"
echo "explicit port/name/GPU overrides passed"

for bad in "registry.example/qwen:tested@sha256:0123456789abcdef" "registry.example/qwen:tested@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdeg"; do
  if env PATH="$tmp/bin:$PATH" MODEL_DIR="$tmp/model" HF_CACHE_HUB="$tmp/hf" IMAGE="$bad" CACHE_DIR="$tmp/cache-bad" "$repo/serve.sh" 2>"$tmp/error"; then echo "invalid digest unexpectedly accepted: $bad" >&2; exit 1; fi
  grep -q 'immutable image@sha256:digest' "$tmp/error"
done
echo "launcher digest validation valid/invalid cases passed"

printf '%s\n' '#!/bin/sh' 'printf "LISTEN 0 0 127.0.0.1:11437\\n"' > "$tmp/bin/ss"
chmod +x "$tmp/bin/ss"
if env PATH="$tmp/bin:$PATH" MODEL_DIR="$tmp/model" HF_CACHE_HUB="$tmp/hf" IMAGE="$valid" CACHE_DIR="$tmp/cache-collision" PROFILE=replica1 "$repo/serve.sh" 2>"$tmp/collision"; then
  echo "port collision unexpectedly accepted" >&2; exit 1
fi
grep -q 'host port 11437 is already listening' "$tmp/collision"
echo "port collision refusal passed"
