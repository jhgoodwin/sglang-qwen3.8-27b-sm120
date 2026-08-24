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
run_profile tp1-bf16-production 0 11436 qwen3.8-27b-sglang 1
run_profile tp2-bf16-safe 0,1 11436 sglang-qwen38-27b-tp2-bf16-safe 2
run_profile tp2-bf16-production 0,1 11436 qwen3.8-27b-sglang 2
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

run_current_cookbook() {
  local profile=$1 expected_port=$2 expected_strategy=$3 expected_ratio=$4 speculative=$5
  CAPTURE="$tmp/$profile.args" env PATH="$tmp/bin:$PATH" MODEL_DIR="$tmp/model" HF_CACHE_HUB="$tmp/hf" \
    DRAFT_MODEL_DIR="$tmp/draft" IMAGE="$valid" CACHE_DIR="$tmp/cache-$profile" \
    PROFILE="$profile" CAPTURE="$tmp/$profile.args" "$repo/serve.sh"
  grep -Fq -- "--name sglang-qwen38-27b-$profile" "$tmp/$profile.args"
  grep -Fq -- "--gpus \"device=0\"" "$tmp/$profile.args"
  grep -Fxq -- '"device=0"' "$tmp/$profile.args"
  grep -Fq -- "-p 127.0.0.1:$expected_port:8000" "$tmp/$profile.args"
  grep -Fq -- '--model-path /models/Qwen3.8-27B' "$tmp/$profile.args"
  grep -Fq -- '--attention-backend flashinfer' "$tmp/$profile.args"
  grep -Fq -- '--chunked-prefill-size 2048' "$tmp/$profile.args"
  grep -Fq -- '--mem-fraction-static 0.85' "$tmp/$profile.args"
  grep -Fq -- "--mamba-radix-cache-strategy $expected_strategy" "$tmp/$profile.args"
  grep -Fq -- "--mamba-full-memory-ratio $expected_ratio" "$tmp/$profile.args"
  grep -Fq -- '--max-running-requests 1' "$tmp/$profile.args"
  grep -Fq -- '--kv-cache-dtype fp8_e4m3' "$tmp/$profile.args"
  if [[ "$speculative" == 1 ]]; then
    grep -Fq -- "-v $tmp/draft:/models/Qwen3.8-27B-DFlash2:ro" "$tmp/$profile.args"
    grep -Fq -- '--speculative-algorithm DFLASH' "$tmp/$profile.args"
    grep -Fq -- '--speculative-draft-model-path /models/Qwen3.8-27B-DFlash2' "$tmp/$profile.args"
    grep -Fq -- '--speculative-num-draft-tokens 8' "$tmp/$profile.args"
  else
    ! grep -Fq -- '--speculative-' "$tmp/$profile.args"
  fi
}
run_current_cookbook tp1-nvfp4-cookbook-no-spec 11444 extra_buffer 2.55 0
run_current_cookbook tp1-nvfp4-dflash-cookbook-default 11445 extra_buffer 6.63 1
run_current_cookbook tp1-nvfp4-dflash-cookbook-lazy 11446 extra_buffer_lazy 6.12 1
echo "current cookbook NVFP4 profile args and isolated defaults passed"

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
echo '#!/bin/sh' > "$tmp/bin/ss"
echo 'exit 0' >> "$tmp/bin/ss"
chmod +x "$tmp/bin/ss"

# Queued C2/C3 profiles use the pinned evidence overlay and reserve an
# operator-owned JSONL target without changing the existing profile contract.
c2_image='qwen38-c2c3-evidence@sha256:cb7a56b3cc39872f43732a58e5adc13361bb24a53d1425ab878d2829a90ac6d0'
run_c2c3() {
  local profile=$1 port=$2 max_running=$3 mamba_size=$4
  local evidence="$tmp/evidence-$profile"
  mkdir -p "$evidence"
  CAPTURE="$tmp/$profile.args" env PATH="$tmp/bin:$PATH" MODEL_DIR="$tmp/model" HF_CACHE_HUB="$tmp/hf" \
    DRAFT_MODEL_DIR="$tmp/draft" EVIDENCE_DIR="$evidence" EVIDENCE_FILE="$profile.jsonl" \
    CACHE_DIR="$tmp/cache-$profile" PROFILE="$profile" CAPTURE="$tmp/$profile.args" "$repo/serve.sh"
  grep -Fq -- "--name qwen3.8-27b-sglang" "$tmp/$profile.args"
  grep -Fq -- "--pid=host" "$tmp/$profile.args"
  grep -Fq -- "-p 127.0.0.1:$port:8000" "$tmp/$profile.args"
  grep -Fq -- "$c2_image" "$tmp/$profile.args"
  grep -Fq -- "-v $evidence:/c2-c3-evidence:rw" "$tmp/$profile.args"
  grep -Fq -- "SGLANG_C2C3_EVIDENCE_PATH=/c2-c3-evidence/$profile.jsonl" "$tmp/$profile.args"
  grep -Fq -- "--max-running-requests $max_running" "$tmp/$profile.args"
  grep -Fq -- "--max-mamba-cache-size $mamba_size" "$tmp/$profile.args"
  grep -Fq -- "--speculative-num-draft-tokens 8" "$tmp/$profile.args"
  grep -Fq -- "--mamba-radix-cache-strategy extra_buffer_lazy" "$tmp/$profile.args"
  grep -Fq -- "-v $tmp/draft:/models/Qwen3.8-27B-DFlash2:ro" "$tmp/$profile.args"
  ! grep -Fq -- '--mamba-full-memory-ratio' "$tmp/$profile.args"
  [[ -s "$evidence/$profile.jsonl" ]] && { echo "evidence target unexpectedly nonempty" >&2; exit 1; }
  [[ "$(stat -c '%a' "$evidence/$profile.jsonl")" == 600 ]] || { echo "evidence target is not mode 0600" >&2; exit 1; }
  if env PATH="$tmp/bin:$PATH" MODEL_DIR="$tmp/model" HF_CACHE_HUB="$tmp/hf" DRAFT_MODEL_DIR="$tmp/draft" \
      EVIDENCE_DIR="$evidence" EVIDENCE_FILE="$profile.jsonl" CACHE_DIR="$tmp/cache-stale-$profile" PROFILE="$profile" \
      "$repo/serve.sh" 2>"$tmp/stale-$profile"; then
    echo "stale evidence target unexpectedly accepted: $profile" >&2; exit 1
  fi
  grep -q 'evidence target already exists' "$tmp/stale-$profile"
}
run_c2c3 c2 11447 2 8
run_c2c3 c3 11448 3 12
auto_evidence="$tmp/evidence-auto"
mkdir -p "$auto_evidence"
CAPTURE="$tmp/c2-auto.args" env PATH="$tmp/bin:$PATH" MODEL_DIR="$tmp/model" HF_CACHE_HUB="$tmp/hf" \
  DRAFT_MODEL_DIR="$tmp/draft" EVIDENCE_DIR="$auto_evidence" CACHE_DIR="$tmp/cache-c2-auto" \
  PROFILE=c2 CAPTURE="$tmp/c2-auto.args" "$repo/serve.sh"
auto_path=$(grep -o -- 'SGLANG_C2C3_EVIDENCE_PATH=/c2-c3-evidence/[^ ]*' "$tmp/c2-auto.args" | head -1)
grep -Fq -- 'SGLANG_C2C3_EVIDENCE_PATH=/c2-c3-evidence/c2-' <<<"$auto_path"
auto_file=${auto_path##*/}
[[ "$auto_file" != *'/'* && ! -s "$auto_evidence/$auto_file" ]]
if env PATH="$tmp/bin:$PATH" MODEL_DIR="$tmp/model" HF_CACHE_HUB="$tmp/hf" DRAFT_MODEL_DIR="$tmp/draft" \
    EVIDENCE_DIR="$auto_evidence" EVIDENCE_FILE=bad/name.jsonl CACHE_DIR="$tmp/cache-slash" PROFILE=c2 "$repo/serve.sh" 2>"$tmp/slash"; then
  echo "slash-containing evidence filename unexpectedly accepted" >&2; exit 1
fi
grep -q 'EVIDENCE_FILE must be a filename' "$tmp/slash"

# A stopped container still reserves its name; ps -a must be used for this
# guard. Replace only the test double, never a real Docker service.
printf '%s\n' '#!/bin/sh' \
  'case "$*" in' \
  "  'ps -a '* ) printf '%s\\n' 'qwen3.8-27b-sglang'; exit 0 ;;" \
  'esac' 'exit 0' > "$tmp/bin/docker"
chmod +x "$tmp/bin/docker"
if env PATH="$tmp/bin:$PATH" MODEL_DIR="$tmp/model" HF_CACHE_HUB="$tmp/hf" DRAFT_MODEL_DIR="$tmp/draft" \
    EVIDENCE_DIR="$tmp/evidence-stopped" CACHE_DIR="$tmp/cache-stopped" PROFILE=c2 "$repo/serve.sh" 2>"$tmp/stopped"; then
  echo "stopped-name collision unexpectedly accepted" >&2; exit 1
fi
grep -q 'container name qwen3.8-27b-sglang is already in use' "$tmp/stopped"
echo "C2/C3 pinned evidence profiles, state pins, PID namespace, and stale-target refusal passed"
python3 - "$repo/serve.sh" <<'PY'
import sys
text = open(sys.argv[1]).read()
assert text.index("model_dir=/data/models/models--RadixArk--Qwen3.8-27B-NVFP4") < text.index("model_mount_src=$model_dir")
assert "docker ps -a --filter" in text
assert 'evidence_file="${profile}-' in text
print("C2/C3 default mount ordering and all-container collision guards passed")
PY

# The evidence image was layered on the immutable no-AVX overlay with patch
# 0001 already present, not directly on the official upstream base.
python3 - "$repo/source.lock.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    lock = json.load(handle)
variant = lock["runtime_variants"]["c2-c3-evidence-overlay"]
official = "lmsysorg/sglang@sha256:616a3e97f45191af975896cfa644279096cb31bd408a071c2e99ca7209c3cafe"
parent = "sglang-qwen38-27b-sm120@sha256:d3346cea82545d982b7ec169f1f0f6f47834b0c4a70ec693e954a8d66111cb8d"
parent_id = "sha256:d3346cea82545d982b7ec169f1f0f6f47834b0c4a70ec693e954a8d66111cb8d"
alias = "qwen38-noavx-base:c2c3-build"
assert variant["upstream_base_image"] == official
assert variant["parent_overlay_image"] == parent
assert variant["temporary_build_alias"] == alias
assert variant["temporary_build_alias_verified_image_id"] == parent_id
assert variant["buildkit_resolved_parent"] == f"{alias}@{parent_id}"
assert variant["build_args"] == {"BASE_IMAGE": alias, "BASE_HAS_NOAVX": "1"}
assert variant["temporary_build_alias_status"] == "removed_after_build"
assert variant["patch_application"] == {
    "patches/sglang/0001-noavx-disable-nixl-ep-import.patch": "inherited_from_parent_overlay_and_verified_before_skip",
    "patches/sglang/0002-c2c3-server-evidence.patch": "applied_by_this_overlay_build",
}
assert variant["status"] == "built_no_gpu_import_verified"
print("C2/C3 evidence overlay parent and patch provenance passed")
PY

python3 - "$repo/profiles.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    profiles = json.load(handle)["profiles"]
image = "qwen38-c2c3-evidence@sha256:cb7a56b3cc39872f43732a58e5adc13361bb24a53d1425ab878d2829a90ac6d0"
for name, port, running, cache in (("c2", 11447, "2", "8"), ("c3", 11448, "3", "12")):
    profile = profiles[name]
    args = profile["extra_args"]
    assert profile["image"] == image
    assert profile["context_length"] == 262144
    assert profile["port"] == port
    assert profile["container_name"] == "qwen3.8-27b-sglang"
    assert args[args.index("--max-running-requests") + 1] == running
    assert args[args.index("--max-mamba-cache-size") + 1] == cache
    assert "--mamba-full-memory-ratio" not in args
print("C2/C3 declarative profiles match runnable state pins passed")
PY
