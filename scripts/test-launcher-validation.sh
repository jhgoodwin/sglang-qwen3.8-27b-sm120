#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/model" "$tmp/hf" "$tmp/bin"
mkdir -p "$tmp/draft"
printf '%s\n' '#!/bin/sh' 'printf "GPU 0\\nGPU 1\\n"' > "$tmp/bin/nvidia-smi"
printf '%s\n' '#!/bin/sh' 'printf "%s\\n" "$*" > "$CAPTURE"' 'printf "%s\\n" "$@" >> "$CAPTURE"' > "$tmp/bin/docker"
chmod +x "$tmp/bin/nvidia-smi" "$tmp/bin/docker"
valid="registry.example/qwen:tested@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

run_profile() {
  local profile=$1 expected_gpu=$2 expected_port=$3 expected_name=$4 expected_tp=$5
  CAPTURE="$tmp/$profile.args" env PATH="$tmp/bin:$PATH" MODEL_DIR="$tmp/model" \
    HF_CACHE_HUB="$tmp/hf" IMAGE="$valid" CACHE_DIR="$tmp/cache-$profile" \
    PROFILE="$profile" CAPTURE="$tmp/$profile.args" "$repo/serve.sh"
  grep -Fq -- "--name $expected_name" "$tmp/$profile.args"
  grep -Fq -- "--gpus \"device=$expected_gpu\"" "$tmp/$profile.args"
  grep -Fxq -- "\"device=$expected_gpu\"" "$tmp/$profile.args"
  grep -Fq -- "-p 127.0.0.1:$expected_port:8000" "$tmp/$profile.args"
  grep -Fq -- "--tp-size $expected_tp" "$tmp/$profile.args"
  grep -Fq -- '--port 8000' "$tmp/$profile.args"
  grep -Fq -- '--host 0.0.0.0' "$tmp/$profile.args"
  grep -Fq -- '--shm-size=16g' "$tmp/$profile.args"
  grep -Fq -- '--ulimit memlock=-1' "$tmp/$profile.args"
  grep -Fq -- "$tmp/model:/models/Qwen3.8-27B:ro" "$tmp/$profile.args"
  grep -Fq -- "$tmp/hf:/hf-cache:ro" "$tmp/$profile.args"
  grep -Fq -- '/cache/torch' "$tmp/$profile.args"
  grep -Fq -- '/cache/triton' "$tmp/$profile.args"
  grep -Fq -- '/cache/flashinfer' "$tmp/$profile.args"
  grep -Fq -- 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' "$tmp/$profile.args"
  grep -Fq -- '--mamba-radix-cache-strategy extra_buffer_lazy' "$tmp/$profile.args"
  ! grep -Fq -- '--speculative-' "$tmp/$profile.args"
}

run_profile tp1-bf16-safe 0 11436 sglang-qwen38-27b-tp1-bf16-safe 1
run_profile tp1-bf16-production 0 11436 sglang-qwen38-27b-tp1-bf16-production 1
run_profile tp2-bf16-safe 0,1 11436 sglang-qwen38-27b-tp2-bf16-safe 2
run_profile tp2-bf16-production 0,1 11436 sglang-qwen38-27b-tp2-bf16-production 2
run_profile replica0 0 11436 sglang-qwen38-27b-replica0 1
run_profile replica1 1 11437 sglang-qwen38-27b-replica1 1
echo "all named profile docker defaults passed"

# Canonical Hugging Face snapshots must mount the repository root so their
# ../../blobs/* symlinks remain valid inside the container.
snapshot="$tmp/hf-repo/snapshots/revision-abc"
mkdir -p "$snapshot" "$tmp/hf-repo/blobs"
CAPTURE="$tmp/snapshot.args" env PATH="$tmp/bin:$PATH" MODEL_DIR="$snapshot" \
  HF_CACHE_HUB="$tmp/hf" IMAGE="$valid" CACHE_DIR="$tmp/cache-snapshot" \
  PROFILE=tp1-bf16-safe CAPTURE="$tmp/snapshot.args" "$repo/serve.sh"
grep -Fq -- "-v $tmp/hf-repo:/models/Qwen3.8-27B-cache:ro" "$tmp/snapshot.args"
grep -Fq -- '--model-path /models/Qwen3.8-27B-cache/snapshots/revision-abc' "$tmp/snapshot.args"
echo "Hugging Face snapshot symlink-preserving mount passed"

run_candidate() {
  local profile=$1 expected_gpu=$2 expected_port=$3 expected_name=$4 expected_tp=$5
  CAPTURE="$tmp/$profile.args" env PATH="$tmp/bin:$PATH" MODEL_DIR="$tmp/model" HF_CACHE_HUB="$tmp/hf" \
    DRAFT_MODEL_DIR="$tmp/draft" IMAGE="$valid" CACHE_DIR="$tmp/cache-$profile" \
    PROFILE="$profile" CAPTURE="$tmp/$profile.args" "$repo/serve.sh"
  grep -Fq -- "--name $expected_name" "$tmp/$profile.args"
  grep -Fq -- "--gpus \"device=$expected_gpu\"" "$tmp/$profile.args"
  grep -Fxq -- "\"device=$expected_gpu\"" "$tmp/$profile.args"
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

run_nvfp4_dspark_candidate() {
  CAPTURE="$tmp/tp1-nvfp4-dspark-candidate.args" env PATH="$tmp/bin:$PATH" MODEL_DIR="$tmp/model" HF_CACHE_HUB="$tmp/hf" \
    DRAFT_MODEL_DIR="$tmp/draft" IMAGE="$valid" CACHE_DIR="$tmp/cache-nvfp4-dspark" \
    PROFILE=tp1-nvfp4-dspark-candidate CAPTURE="$tmp/tp1-nvfp4-dspark-candidate.args" "$repo/serve.sh"
  grep -Fq -- '--name sglang-qwen38-27b-tp1-nvfp4-dspark-candidate' "$tmp/tp1-nvfp4-dspark-candidate.args"
  grep -Fq -- '--gpus "device=0"' "$tmp/tp1-nvfp4-dspark-candidate.args"
  grep -Fxq -- '"device=0"' "$tmp/tp1-nvfp4-dspark-candidate.args"
  grep -Fq -- '-p 127.0.0.1:11443:8000' "$tmp/tp1-nvfp4-dspark-candidate.args"
  grep -Fq -- '--tp-size 1' "$tmp/tp1-nvfp4-dspark-candidate.args"
  grep -Fq -- '-v '"$tmp"'/draft:/models/Qwen3.8-27B-DSpark:ro' "$tmp/tp1-nvfp4-dspark-candidate.args"
  grep -Fq -- '--model-path /models/Qwen3.8-27B' "$tmp/tp1-nvfp4-dspark-candidate.args"
  grep -Fq -- '--speculative-algorithm DSPARK' "$tmp/tp1-nvfp4-dspark-candidate.args"
  grep -Fq -- '--speculative-draft-model-path /models/Qwen3.8-27B-DSpark' "$tmp/tp1-nvfp4-dspark-candidate.args"
  grep -Fq -- '--speculative-draft-attention-backend flashinfer' "$tmp/tp1-nvfp4-dspark-candidate.args"
  grep -Fq -- '--speculative-dspark-block-size 7' "$tmp/tp1-nvfp4-dspark-candidate.args"
  grep -Fq -- '--speculative-draft-model-quantization unquant' "$tmp/tp1-nvfp4-dspark-candidate.args"
  grep -Fq -- '--attention-backend flashinfer' "$tmp/tp1-nvfp4-dspark-candidate.args"
  grep -Fq -- '--mamba-ssm-dtype float32' "$tmp/tp1-nvfp4-dspark-candidate.args"
  grep -Fq -- '--mamba-radix-cache-strategy extra_buffer_lazy' "$tmp/tp1-nvfp4-dspark-candidate.args"
  [[ "$(grep -o -- '--kv-cache-dtype fp8_e4m3' "$tmp/tp1-nvfp4-dspark-candidate.args" | wc -l)" -eq 1 ]]
}
run_nvfp4_dspark_candidate
if env PATH="$tmp/bin:$PATH" MODEL_DIR="$tmp/model" HF_CACHE_HUB="$tmp/hf" IMAGE="$valid" CACHE_DIR="$tmp/cache-missing-nvfp4-dspark" PROFILE=tp1-nvfp4-dspark-candidate "$repo/serve.sh" 2>"$tmp/missing-nvfp4-dspark"; then
  echo "NVFP4 DSpark candidate without DRAFT_MODEL_DIR unexpectedly accepted" >&2; exit 1
fi
grep -q 'DRAFT_MODEL_DIR is required' "$tmp/missing-nvfp4-dspark"
echo "NVFP4 DSpark candidate args and draft requirement passed"

draft_snapshot="$tmp/draft-repo/snapshots/revision-draft"
mkdir -p "$draft_snapshot" "$tmp/draft-repo/blobs"
CAPTURE="$tmp/draft-snapshot.args" env PATH="$tmp/bin:$PATH" MODEL_DIR="$tmp/model" HF_CACHE_HUB="$tmp/hf" \
  DRAFT_MODEL_DIR="$draft_snapshot" IMAGE="$valid" CACHE_DIR="$tmp/cache-draft-snapshot" \
  PROFILE=tp1-bf16-dspark-candidate CAPTURE="$tmp/draft-snapshot.args" "$repo/serve.sh"
grep -Fq -- "-v $tmp/draft-repo:/models/Qwen3.8-27B-DSpark-cache:ro" "$tmp/draft-snapshot.args"
grep -Fq -- '--speculative-draft-model-path /models/Qwen3.8-27B-DSpark-cache/snapshots/revision-draft' "$tmp/draft-snapshot.args"
echo "DSpark snapshot symlink-preserving mount passed"

run_dflash_candidate() {
  local profile=$1
  CAPTURE="$tmp/$profile.args" env PATH="$tmp/bin:$PATH" MODEL_DIR="$tmp/model" HF_CACHE_HUB="$tmp/hf" \
    DRAFT_MODEL_DIR="$tmp/draft" IMAGE="$valid" CACHE_DIR="$tmp/cache-$profile" \
    PROFILE="$profile" CAPTURE="$tmp/$profile.args" "$repo/serve.sh"
  grep -Fq -- '--name sglang-qwen38-27b-tp1-bf16-dflash-candidate' "$tmp/$profile.args"
  grep -Fq -- '--gpus "device=0"' "$tmp/$profile.args"
  grep -Fxq -- '"device=0"' "$tmp/$profile.args"
  grep -Fq -- '-p 127.0.0.1:11442:8000' "$tmp/$profile.args"
  grep -Fq -- '--tp-size 1' "$tmp/$profile.args"
  grep -Fq -- "-v $tmp/draft:/models/Qwen3.8-27B-DFlash2:ro" "$tmp/$profile.args"
  grep -Fq -- '--speculative-algorithm DFLASH' "$tmp/$profile.args"
  grep -Fq -- '--speculative-draft-model-path /models/Qwen3.8-27B-DFlash2' "$tmp/$profile.args"
  grep -Fq -- '--speculative-num-draft-tokens 8' "$tmp/$profile.args"
  grep -Fq -- '--attention-backend flashinfer' "$tmp/$profile.args"
  grep -Fq -- '--kv-cache-dtype fp8_e4m3' "$tmp/$profile.args"
  grep -Fq -- '--mamba-ssm-dtype float32' "$tmp/$profile.args"
  grep -Fq -- '--mamba-radix-cache-strategy extra_buffer_lazy' "$tmp/$profile.args"
}
run_dflash_candidate tp1-bf16-dflash-candidate
if env PATH="$tmp/bin:$PATH" MODEL_DIR="$tmp/model" HF_CACHE_HUB="$tmp/hf" IMAGE="$valid" CACHE_DIR="$tmp/cache-missing-dflash" PROFILE=tp1-bf16-dflash-candidate "$repo/serve.sh" 2>"$tmp/missing-dflash"; then
  echo "DFlash candidate without DRAFT_MODEL_DIR unexpectedly accepted" >&2; exit 1
fi
grep -q 'DRAFT_MODEL_DIR is required' "$tmp/missing-dflash"
echo "DFlash candidate exact args and draft requirement passed"

dflash_snapshot="$tmp/dflash-repo/snapshots/revision-dflash"
mkdir -p "$dflash_snapshot" "$tmp/dflash-repo/blobs"
CAPTURE="$tmp/dflash-snapshot.args" env PATH="$tmp/bin:$PATH" MODEL_DIR="$tmp/model" HF_CACHE_HUB="$tmp/hf" \
  DRAFT_MODEL_DIR="$dflash_snapshot" IMAGE="$valid" CACHE_DIR="$tmp/cache-dflash-snapshot" \
  PROFILE=tp1-bf16-dflash-candidate CAPTURE="$tmp/dflash-snapshot.args" "$repo/serve.sh"
grep -Fq -- "-v $tmp/dflash-repo:/models/Qwen3.8-27B-DFlash2-cache:ro" "$tmp/dflash-snapshot.args"
grep -Fq -- '--speculative-draft-model-path /models/Qwen3.8-27B-DFlash2-cache/snapshots/revision-dflash' "$tmp/dflash-snapshot.args"
echo "DFlash snapshot symlink-preserving mount passed"

run_eagle_candidate() {
  local profile=$1 expected_gpu=$2 expected_port=$3 expected_name=$4 expected_tp=$5
  CAPTURE="$tmp/$profile.args" env PATH="$tmp/bin:$PATH" MODEL_DIR="$tmp/model" HF_CACHE_HUB="$tmp/hf" \
    IMAGE="$valid" CACHE_DIR="$tmp/cache-$profile" PROFILE="$profile" CAPTURE="$tmp/$profile.args" "$repo/serve.sh"
  grep -Fq -- "--name $expected_name" "$tmp/$profile.args"
  grep -Fq -- "--gpus \"device=$expected_gpu\"" "$tmp/$profile.args"
  grep -Fq -- "-p 127.0.0.1:$expected_port:8000" "$tmp/$profile.args"
  grep -Fq -- "--tp-size $expected_tp" "$tmp/$profile.args"
  grep -Fq -- '--speculative-algorithm EAGLE' "$tmp/$profile.args"
  grep -Fq -- '--speculative-num-steps 3' "$tmp/$profile.args"
  grep -Fq -- '--speculative-eagle-topk 1' "$tmp/$profile.args"
  grep -Fq -- '--speculative-num-draft-tokens 4' "$tmp/$profile.args"
  grep -Fq -- '--mamba-ssm-dtype float32' "$tmp/$profile.args"
  grep -Fq -- '--mamba-radix-cache-strategy extra_buffer_lazy' "$tmp/$profile.args"
  grep -Fq -- '--mem-fraction-static 0.80' "$tmp/$profile.args"
}
run_eagle_candidate tp1-bf16-eagle-candidate 0 11440 sglang-qwen38-27b-tp1-bf16-eagle-candidate 1
! grep -Fq -- '--disable-custom-all-reduce' "$tmp/tp1-bf16-eagle-candidate.args"
run_eagle_candidate tp2-bf16-eagle-candidate 0,1 11441 sglang-qwen38-27b-tp2-bf16-eagle-candidate 2
grep -Fq -- '--disable-custom-all-reduce' "$tmp/tp2-bf16-eagle-candidate.args"
echo "EAGLE candidate args and TP2 custom-all-reduce guard passed"

CAPTURE="$tmp/override.args" env PATH="$tmp/bin:$PATH" MODEL_DIR="$tmp/model" HF_CACHE_HUB="$tmp/hf" \
  IMAGE="$valid" CACHE_DIR="$tmp/cache-override" PROFILE=replica1 PORT=12345 CONTAINER_NAME=custom-qwen \
  GPU_LIST=0 TP_SIZE=1 CAPTURE="$tmp/override.args" "$repo/serve.sh"
grep -Fq -- '--name custom-qwen' "$tmp/override.args"
grep -Fq -- '-p 127.0.0.1:12345:8000' "$tmp/override.args"
grep -Fq -- '--gpus "device=0"' "$tmp/override.args"
grep -Fxq -- '"device=0"' "$tmp/override.args"
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
