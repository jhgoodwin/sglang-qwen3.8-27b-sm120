#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
profile=${PROFILE:-tp1-bf16-safe}
model_dir=${MODEL_DIR:-/data/models/Qwen3.8-27B}
model_dir_env=${MODEL_DIR:-}
hf_cache=${HF_CACHE_HUB:-/data/models}
image=${IMAGE:-UNRESOLVED@sha256:REPLACE_AFTER_IMAGE_VERIFICATION}
image_env=${IMAGE:-}
gpus=${GPU_LIST:-}
tp=${TP_SIZE:-}
context=${CONTEXT_LENGTH:-}
port=${PORT:-}
container_port=${CONTAINER_PORT:-8000}
name=${CONTAINER_NAME:-}
cache_dir=${CACHE_DIR:-}
source_revision=${SOURCE_REVISION:-unresolved}
draft_model_dir=${DRAFT_MODEL_DIR:-}
draft_model_dir_env=${DRAFT_MODEL_DIR:-}
if [[ ( "$profile" == c2 || "$profile" == c3 ) && -z "$model_dir_env" ]]; then
  model_dir=/data/models/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/319f741cce68d7914884900c138a1fbb70a42f30
fi
if [[ ( "$profile" == c2 || "$profile" == c3 ) && -z "$draft_model_dir_env" ]]; then
  draft_model_dir=/data/models/models--incoai--Qwen3.8-27B-DFlash2/snapshots/dedf8df68adfb1afeaf7b7480c0a0243108177b4
fi
draft_model_mount_src=$draft_model_dir
draft_model_mount_name=Qwen3.8-27B-DSpark
case "$profile" in
  tp1-bf16-dflash-candidate|tp1-nvfp4-dflash-cookbook-default|tp1-nvfp4-dflash-cookbook-lazy|c2|c3) draft_model_mount_name=Qwen3.8-27B-DFlash2 ;;
esac
draft_model_mount_target=/models/$draft_model_mount_name
draft_model_container_path=$draft_model_mount_target

# Hugging Face snapshots contain relative symlinks into the repository-level
# blobs/ directory. Mounting only snapshots/<revision> breaks those links in
# the container, so canonical cache snapshots are mounted with their repo root.
model_mount_src=$model_dir
model_mount_target=/models/Qwen3.8-27B
model_container_path=$model_mount_target
if [[ "$(basename "$(dirname "$model_dir")")" == snapshots ]]; then
  model_revision=$(basename "$model_dir")
  model_mount_src=$(dirname "$(dirname "$model_dir")")
  model_mount_target=/models/Qwen3.8-27B-cache
  model_container_path="$model_mount_target/snapshots/$model_revision"
fi
if [[ -n "$draft_model_dir" && "$(basename "$(dirname "$draft_model_dir")")" == snapshots ]]; then
  draft_model_revision=$(basename "$draft_model_dir")
  draft_model_mount_src=$(dirname "$(dirname "$draft_model_dir")")
  draft_model_mount_target=/models/${draft_model_mount_name}-cache
  draft_model_container_path="$draft_model_mount_target/snapshots/$draft_model_revision"
fi

die() { echo "serve.sh: $*" >&2; exit 2; }
[[ -d "$model_dir" ]] || die "model path does not exist: $model_dir (set MODEL_DIR; input is mounted read-only)"
[[ -d "$hf_cache" ]] || die "HF cache path does not exist: $hf_cache (set HF_CACHE_HUB)"
[[ "$container_port" =~ ^[0-9]+$ && "$container_port" -ge 1 && "$container_port" -le 65535 ]] || die "invalid CONTAINER_PORT: $container_port"

case "$profile" in
  tp1-bf16-safe) default_tp=1; default_gpus=0; default_port=11436; default_name=sglang-qwen38-27b-tp1-bf16-safe ;;
  tp1-bf16-production) default_tp=1; default_gpus=0; default_port=11436; default_name=qwen3.8-27b-sglang ;;
  tp2-bf16-safe) default_tp=2; default_gpus=0,1; default_port=11436; default_name=sglang-qwen38-27b-tp2-bf16-safe ;;
  tp2-bf16-production) default_tp=2; default_gpus=0,1; default_port=11436; default_name=qwen3.8-27b-sglang ;;
  replica0) default_tp=1; default_gpus=0; default_port=11436; default_name=sglang-qwen38-27b-replica0 ;;
  replica1) default_tp=1; default_gpus=1; default_port=11437; default_name=sglang-qwen38-27b-replica1 ;;
  tp1-bf16-dspark-candidate) default_tp=1; default_gpus=0; default_port=11438; default_name=sglang-qwen38-27b-tp1-bf16-dspark-candidate ;;
  tp1-nvfp4-dspark-candidate) default_tp=1; default_gpus=0; default_port=11443; default_name=sglang-qwen38-27b-tp1-nvfp4-dspark-candidate ;;
  tp2-bf16-dspark-candidate) default_tp=2; default_gpus=0,1; default_port=11439; default_name=sglang-qwen38-27b-tp2-bf16-dspark-candidate ;;
  tp1-bf16-dflash-candidate) default_tp=1; default_gpus=0; default_port=11442; default_name=sglang-qwen38-27b-tp1-bf16-dflash-candidate ;;
  tp1-nvfp4-cookbook-no-spec) default_tp=1; default_gpus=0; default_port=11444; default_name=sglang-qwen38-27b-tp1-nvfp4-cookbook-no-spec ;;
  tp1-nvfp4-dflash-cookbook-default) default_tp=1; default_gpus=0; default_port=11445; default_name=sglang-qwen38-27b-tp1-nvfp4-dflash-cookbook-default ;;
  tp1-nvfp4-dflash-cookbook-lazy) default_tp=1; default_gpus=0; default_port=11446; default_name=sglang-qwen38-27b-tp1-nvfp4-dflash-cookbook-lazy ;;
  c2) default_tp=1; default_gpus=0; default_port=11447; default_name=qwen3.8-27b-sglang; profile_image=qwen38-c2c3-evidence@sha256:c06fcb906923c13579ff0a1bd01bc8c728e2fef9e6adc549fb0677a7d21dfddb ;;
  c3) default_tp=1; default_gpus=0; default_port=11448; default_name=qwen3.8-27b-sglang; profile_image=qwen38-c2c3-evidence@sha256:c06fcb906923c13579ff0a1bd01bc8c728e2fef9e6adc549fb0677a7d21dfddb ;;
  tp1-bf16-eagle-candidate) default_tp=1; default_gpus=0; default_port=11440; default_name=sglang-qwen38-27b-tp1-bf16-eagle-candidate ;;
  tp2-bf16-eagle-candidate) default_tp=2; default_gpus=0,1; default_port=11441; default_name=sglang-qwen38-27b-tp2-bf16-eagle-candidate ;;
  *) die "unknown PROFILE '$profile' (see profiles.json)" ;;
esac
if [[ -z "$image_env" && -n "${profile_image:-}" ]]; then image=$profile_image; fi
[[ "$image" =~ @sha256:[0-9a-fA-F]{64}$ ]] || die "IMAGE must be an immutable image@sha256:digest; unresolved default is not runnable"
port=${port:-$default_port}; name=${name:-$default_name}
[[ "$port" =~ ^[0-9]+$ && "$port" -ge 1 && "$port" -le 65535 ]] || die "invalid PORT: $port"
case "$profile" in
  tp1-bf16-dspark-candidate|tp2-bf16-dspark-candidate|tp1-nvfp4-dspark-candidate|tp1-bf16-dflash-candidate|tp1-nvfp4-dflash-cookbook-default|tp1-nvfp4-dflash-cookbook-lazy|c2|c3)
    [[ -n "$draft_model_dir" ]] || die "DRAFT_MODEL_DIR is required for $profile (use an existing local draft snapshot; no download is performed)"
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
if [[ "$profile" == c2 || "$profile" == c3 ]]; then
  if docker ps -a --filter "name=^${name}$" --format '{{.Names}}' 2>/dev/null | grep -Fxq "$name"; then
    die "container name $name is already in use; stop the existing service or choose CONTAINER_NAME explicitly (no automatic replacement)"
  fi
fi
if [[ -z "$cache_dir" ]]; then digest=${image##*@sha256:}; cache_dir="/data/models/sglang-qwen38-27b-cache-v1-${digest:0:12}-${source_revision:0:12}-${profile}"; fi
mkdir -p "$cache_dir"

mamba_strategy=extra_buffer_lazy
case "$profile" in
  tp1-nvfp4-cookbook-no-spec|tp1-nvfp4-dflash-cookbook-default) mamba_strategy=extra_buffer ;;
esac
extra=(--attention-backend flashinfer --chunked-prefill-size 2048 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --mamba-ssm-dtype float32 --mamba-radix-cache-strategy "$mamba_strategy")
case "$profile" in
  tp1-bf16-safe|tp1-bf16-production|tp2-bf16-safe|tp2-bf16-production|replica0|replica1) extra+=(--mem-fraction-static 0.80) ;;
  tp1-bf16-dspark-candidate|tp2-bf16-dspark-candidate)
    extra+=(--mem-fraction-static 0.85 --speculative-algorithm DSPARK
      --speculative-draft-model-path "$draft_model_container_path"
      --speculative-draft-attention-backend flashinfer
      --mamba-radix-cache-strategy extra_buffer_lazy)
    ;;
  tp1-nvfp4-dspark-candidate)
    extra+=(--mem-fraction-static 0.85 --speculative-algorithm DSPARK
      --speculative-draft-model-path "$draft_model_container_path"
      --speculative-draft-attention-backend flashinfer
      --speculative-dspark-block-size 7
      --speculative-draft-model-quantization unquant
      --mamba-radix-cache-strategy extra_buffer_lazy)
    ;;
  tp1-bf16-dflash-candidate)
    extra+=(--mem-fraction-static 0.80 --speculative-algorithm DFLASH
      --speculative-draft-model-path "$draft_model_container_path"
      --speculative-num-draft-tokens 8)
    ;;
  tp1-nvfp4-cookbook-no-spec)
    extra+=(--mem-fraction-static 0.85
      --mamba-full-memory-ratio 2.55 --max-running-requests 1)
    ;;
  tp1-nvfp4-dflash-cookbook-default)
    extra+=(--mem-fraction-static 0.85
      --mamba-full-memory-ratio 6.63 --max-running-requests 1 --speculative-algorithm DFLASH
      --speculative-draft-model-path "$draft_model_container_path"
      --speculative-num-draft-tokens 8)
    ;;
  tp1-nvfp4-dflash-cookbook-lazy)
    extra+=(--mem-fraction-static 0.85
      --mamba-full-memory-ratio 6.12 --max-running-requests 1 --speculative-algorithm DFLASH
      --speculative-draft-model-path "$draft_model_container_path"
      --speculative-num-draft-tokens 8)
    ;;
  c2)
    extra+=(--mem-fraction-static 0.85 --max-running-requests 2 --max-mamba-cache-size 8 --speculative-algorithm DFLASH
      --speculative-draft-model-path "$draft_model_container_path"
      --speculative-num-draft-tokens 8)
    ;;
  c3)
    extra+=(--mem-fraction-static 0.85 --max-running-requests 3 --max-mamba-cache-size 12 --speculative-algorithm DFLASH
      --speculative-draft-model-path "$draft_model_container_path"
      --speculative-num-draft-tokens 8)
    ;;
  tp1-bf16-eagle-candidate)
    extra+=(--mem-fraction-static 0.80 --speculative-algorithm EAGLE --speculative-num-steps 3
      --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
    ;;
  tp2-bf16-eagle-candidate)
    extra+=(--mem-fraction-static 0.80 --speculative-algorithm EAGLE --speculative-num-steps 3
      --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
      --disable-custom-all-reduce)
    ;;
esac
if [[ -n "${SGLANG_EXTRA_ARGS:-}" ]]; then read -r -a user_extra <<< "$SGLANG_EXTRA_ARGS"; extra+=("${user_extra[@]}"); fi

command -v docker >/dev/null 2>&1 || die "docker is required to launch"
mounts=(-v "$model_mount_src:$model_mount_target:ro" -v "$hf_cache:/hf-cache:ro"
  -v "$cache_dir/torch:/cache/torch" -v "$cache_dir/triton:/cache/triton" -v "$cache_dir/flashinfer:/cache/flashinfer")
case "$profile" in
  tp1-bf16-dspark-candidate|tp2-bf16-dspark-candidate|tp1-nvfp4-dspark-candidate|tp1-bf16-dflash-candidate|tp1-nvfp4-dflash-cookbook-default|tp1-nvfp4-dflash-cookbook-lazy|c2|c3) mounts+=( -v "$draft_model_mount_src:$draft_model_mount_target:ro" ) ;;
esac
evidence_mount=()
evidence_env=()
pid_namespace=()
if [[ "$profile" == c2 || "$profile" == c3 ]]; then
  evidence_dir=${EVIDENCE_DIR:-}
  [[ -n "$evidence_dir" ]] || die "EVIDENCE_DIR is required for $profile (use an existing writable absolute directory)"
  [[ "$evidence_dir" == /* ]] || die "EVIDENCE_DIR must be an absolute path"
  [[ -d "$evidence_dir" && -w "$evidence_dir" ]] || die "EVIDENCE_DIR must be an existing writable directory: $evidence_dir"
  evidence_container_dir=/c2-c3-evidence
  evidence_file=${EVIDENCE_FILE:-}
  if [[ -z "$evidence_file" ]]; then
    evidence_file="${profile}-$(date -u +%Y%m%dT%H%M%SZ)-$$.jsonl"
  elif [[ "$evidence_file" == */* ]]; then
    die "EVIDENCE_FILE must be a filename within EVIDENCE_DIR"
  fi
  evidence_host_path="$evidence_dir/$evidence_file"
  [[ ! -e "$evidence_host_path" ]] || die "evidence target already exists; refusing to truncate: $evidence_host_path"
  (umask 077; set -o noclobber; : > "$evidence_host_path") 2>/dev/null || die "could not reserve unique evidence target: $evidence_host_path"
  evidence_mount=(-v "$evidence_dir:$evidence_container_dir:rw")
  evidence_env=(-e "SGLANG_C2C3_EVIDENCE_PATH=$evidence_container_dir/$evidence_file")
  pid_namespace=(--pid=host)
fi
mounts+=("${evidence_mount[@]}")
# Docker's --gpus parser requires the device request's CSV to retain literal
# quotes; without them, a multi-GPU value is parsed as both Count and DeviceIDs.
gpu_request="\"device=$gpus\""
docker run --rm --name "$name" "${pid_namespace[@]}" --gpus "$gpu_request" --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
  -p "127.0.0.1:${port}:${container_port}" \
  "${mounts[@]}" \
  -e HF_HOME=/hf-cache -e TORCHINDUCTOR_CACHE_DIR=/cache/torch -e TRITON_CACHE_DIR=/cache/triton -e FLASHINFER_WORKSPACE_DIR=/cache/flashinfer \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${evidence_env[@]}" \
  "$image" python3 -m sglang.launch_server --model-path "$model_container_path" --trust-remote-code --kv-cache-dtype fp8_e4m3 --context-length "$context" --tp-size "$tp" --host 0.0.0.0 --port "$container_port" "${extra[@]}"
