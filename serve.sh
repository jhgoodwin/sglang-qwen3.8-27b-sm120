#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
profile=${PROFILE:-tp1-bf16-safe}
model_dir=${MODEL_DIR:-/data/models/Qwen3.8-27B}
hf_cache=${HF_CACHE_HUB:-/data/models}
image=${IMAGE:-UNRESOLVED@sha256:REPLACE_AFTER_IMAGE_VERIFICATION}
gpus=${GPU_LIST:-}
tp=${TP_SIZE:-}
context=${CONTEXT_LENGTH:-}
port=${PORT:-}
container_port=${CONTAINER_PORT:-8000}
name=${CONTAINER_NAME:-}
cache_dir=${CACHE_DIR:-}
source_revision=${SOURCE_REVISION:-unresolved}
draft_model_dir=${DRAFT_MODEL_DIR:-}

die() { echo "serve.sh: $*" >&2; exit 2; }
[[ -d "$model_dir" ]] || die "model path does not exist: $model_dir (set MODEL_DIR; input is mounted read-only)"
[[ -d "$hf_cache" ]] || die "HF cache path does not exist: $hf_cache (set HF_CACHE_HUB)"
[[ "$image" =~ @sha256:[0-9a-fA-F]{64}$ ]] || die "IMAGE must be an immutable image@sha256:digest; unresolved default is not runnable"
[[ "$container_port" =~ ^[0-9]+$ && "$container_port" -ge 1 && "$container_port" -le 65535 ]] || die "invalid CONTAINER_PORT: $container_port"

case "$profile" in
  tp1-bf16-safe) default_tp=1; default_gpus=0; default_port=11436; default_name=sglang-qwen38-27b-tp1-bf16-safe ;;
  tp1-bf16-production) default_tp=1; default_gpus=0; default_port=11436; default_name=sglang-qwen38-27b-tp1-bf16-production ;;
  tp2-bf16-safe) default_tp=2; default_gpus=0,1; default_port=11436; default_name=sglang-qwen38-27b-tp2-bf16-safe ;;
  tp2-bf16-production) default_tp=2; default_gpus=0,1; default_port=11436; default_name=sglang-qwen38-27b-tp2-bf16-production ;;
  replica0) default_tp=1; default_gpus=0; default_port=11436; default_name=sglang-qwen38-27b-replica0 ;;
  replica1) default_tp=1; default_gpus=1; default_port=11437; default_name=sglang-qwen38-27b-replica1 ;;
  tp1-bf16-dspark-candidate) default_tp=1; default_gpus=0; default_port=11438; default_name=sglang-qwen38-27b-tp1-bf16-dspark-candidate ;;
  tp2-bf16-dspark-candidate) default_tp=2; default_gpus=0,1; default_port=11439; default_name=sglang-qwen38-27b-tp2-bf16-dspark-candidate ;;
  *) die "unknown PROFILE '$profile' (see profiles.json)" ;;
esac
port=${port:-$default_port}; name=${name:-$default_name}
[[ "$port" =~ ^[0-9]+$ && "$port" -ge 1 && "$port" -le 65535 ]] || die "invalid PORT: $port"
case "$profile" in
  tp1-bf16-dspark-candidate|tp2-bf16-dspark-candidate)
    [[ -n "$draft_model_dir" ]] || die "DRAFT_MODEL_DIR is required for $profile (use an existing local DSpark snapshot; no download is performed)"
    [[ -d "$draft_model_dir" ]] || die "draft model path does not exist: $draft_model_dir (set DRAFT_MODEL_DIR; input is mounted read-only)"
    ;;
esac
tp=${tp:-$default_tp}; gpus=${gpus:-$default_gpus}; context=${context:-262144}
[[ "$tp" =~ ^[1-9][0-9]*$ ]] || die "TP_SIZE must be a positive integer"
[[ "$context" =~ ^[1-9][0-9]*$ ]] || die "CONTEXT_LENGTH must be a positive integer"
IFS=',' read -r -a gpu_array <<< "$gpus"
[[ "${#gpu_array[@]}" -eq "$tp" ]] || die "visible GPU list count (${#gpu_array[@]}) must equal TP_SIZE ($tp): $gpus"
for gpu in "${gpu_array[@]}"; do [[ "$gpu" =~ ^[0-9]+$ ]] || die "GPU_LIST must be comma-separated numeric IDs: $gpus"; done
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required to verify visible GPU count (hardware runtime is unavailable)"
gpu_total=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
for gpu in "${gpu_array[@]}"; do [[ "$gpu" -lt "$gpu_total" ]] || die "GPU_LIST contains GPU $gpu but nvidia-smi reports $gpu_total GPUs"; done

if command -v ss >/dev/null 2>&1 && ss -H -ltn "sport = :$port" | grep -q .; then
  die "host port $port is already listening; stop the existing service or choose PORT explicitly (no automatic replacement)"
fi
if [[ -z "$cache_dir" ]]; then digest=${image##*@sha256:}; cache_dir="/data/models/sglang-qwen38-27b-cache-v1-${digest:0:12}-${source_revision:0:12}-${profile}"; fi
mkdir -p "$cache_dir"

extra=(--attention-backend flashinfer --chunked-prefill-size 2048 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --mamba-ssm-dtype float32)
case "$profile" in
  tp1-bf16-safe|tp1-bf16-production|tp2-bf16-safe|tp2-bf16-production|replica0|replica1) extra+=(--mem-fraction-static 0.80) ;;
  tp1-bf16-dspark-candidate|tp2-bf16-dspark-candidate)
    extra+=(--mem-fraction-static 0.85 --speculative-algorithm DSPARK
      --speculative-draft-model-path /models/Qwen3.8-27B-DSpark
      --speculative-draft-attention-backend flashinfer
      --mamba-radix-cache-strategy extra_buffer_lazy)
    ;;
esac
if [[ -n "${SGLANG_EXTRA_ARGS:-}" ]]; then read -r -a user_extra <<< "$SGLANG_EXTRA_ARGS"; extra+=("${user_extra[@]}"); fi

command -v docker >/dev/null 2>&1 || die "docker is required to launch"
mounts=(-v "$model_dir:/models/Qwen3.8-27B:ro" -v "$hf_cache:/hf-cache:ro"
  -v "$cache_dir/torch:/cache/torch" -v "$cache_dir/triton:/cache/triton" -v "$cache_dir/flashinfer:/cache/flashinfer")
case "$profile" in
  tp1-bf16-dspark-candidate|tp2-bf16-dspark-candidate) mounts+=( -v "$draft_model_dir:/models/Qwen3.8-27B-DSpark:ro" ) ;;
esac
docker run --rm --name "$name" --gpus "device=$gpus" --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
  -p "127.0.0.1:${port}:${container_port}" \
  "${mounts[@]}" \
  -e HF_HOME=/hf-cache -e TORCHINDUCTOR_CACHE_DIR=/cache/torch -e TRITON_CACHE_DIR=/cache/triton -e FLASHINFER_WORKSPACE_DIR=/cache/flashinfer \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$image" python3 -m sglang.launch_server --model-path /models/Qwen3.8-27B --trust-remote-code --kv-cache-dtype fp8_e4m3 --context-length "$context" --tp-size "$tp" --port "$container_port" "${extra[@]}"
