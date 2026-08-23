# Run

Resolve the local snapshot and set `MODEL_DIR` (default
`/data/models/Qwen3.8-27B`) and `HF_CACHE_HUB` (default `/data/models`). Both
are mounted read-only. Replace the unresolved immutable `IMAGE` with a
verified `registry/name:tag@sha256:<64 hex>` and record it in
`source.lock.json`. Run `./scripts/validate-scaffold.sh`, then
`PROFILE=tp1-bf16-safe ./serve.sh`.

The default host endpoint is `127.0.0.1:11436` and maps to container port
8000. SGLang listens on `0.0.0.0` only inside the container so Docker port
publishing works; the host-side publish remains restricted to loopback.

## Capture the host and immutable inputs

Run this from the host or serving container (it is safe in an environment
without NVIDIA or Docker access):

```sh
python3 bench/environment_capture.py --output environment.json \
  --disk-path /data/models
```

The command writes `environment.json` and `source-compatibility.md`. The generated `environment.json` is intentionally local and ignored. If it is missing, copy `environment.example.json` to `environment.json` and replace its placeholders, or generate a fresh capture with the command above. Missing tools are recorded as `unavailable` and still produce a successful capture;
that status must not be interpreted as a host hardware failure. After the
BF16 download is complete, pass only its existing snapshot directory and
explicit identity—this never downloads or starts anything:

```sh
python3 bench/environment_capture.py --output environment.json \
  --hf-snapshot-path /data/models/hub/models--Qwen--Qwen3.8-27B/snapshots/REVISION \
  --hf-repo Qwen/Qwen3.8-27B --hf-revision REVISION --disk-path /data/models
```

Use `--full-hash` only when full file hashing is deliberate and affordable.
The capture records symlink blob names as ETags where available and leaves
source/image pins unresolved until independently verified.

`GPU_LIST` must contain exactly `TP_SIZE` numeric IDs. Defaults are GPU 0 for
TP1 and `0,1` for TP2. Each named profile has a distinct default container
name; `replica1` defaults to port 11437 (and `replica0` to 11436), so both
replicas can run simultaneously. A listening host port is a hard preflight
error; the launcher never stops another service.
`--shm-size=16g`, unlimited memlock, and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` are provisions whose
usefulness remains release/runtime-dependent and must be verified.

The opt-in `tp1-bf16-dspark-candidate` and
`tp2-bf16-dspark-candidate` profiles are unqualified cookbook candidates,
not production defaults. They require `DRAFT_MODEL_DIR` to point at an
already-existing local `RadixArk/Qwen3.8-27B-DSpark` snapshot; the launcher
does not resolve or download Hugging Face artifacts and mounts the draft
read-only. The draft revision is unresolved until captured in the source
lock. Do not advance either candidate without correctness, capacity, and
stability gates, including saving the server's resolved
`max_running_requests` value.

The opt-in `tp1-bf16-eagle-candidate` and `tp2-bf16-eagle-candidate` profiles
are also unqualified, in-checkpoint EAGLE candidates. They inherit the safe
BF16/FP8-KV, float32-SSM, FlashInfer, and `extra_buffer_lazy` settings and add
exactly three draft steps, top-k 1, and four draft tokens. The TP2 candidate
also disables custom all-reduce for this host. Capacity knobs remain
user-overridable through `SGLANG_EXTRA_ARGS`; these profiles do not impose an
experiment-only request limit. Require deterministic correctness and capacity
evidence before treating either candidate as a winner.

The opt-in `tp1-bf16-dflash-candidate` is an unqualified DFlash2 candidate.
It retains the safe BF16 target settings and adds exactly `DFLASH`, eight draft
tokens, and the local `incoai/Qwen3.8-27B-DFlash2` draft path. Set
`DRAFT_MODEL_DIR` to the existing canonical snapshot
`/data/models/models--incoai--Qwen3.8-27B-DFlash2/snapshots/dedf8df68adfb1afeaf7b7480c0a0243108177b4`;
the launcher mounts its repository root read-only so blob symlinks remain
valid and never downloads the draft. Require matched no-spec correctness,
capacity, and stability evidence before interpreting this candidate.

The opt-in `tp1-nvfp4-dspark-candidate` uses the pinned
`RadixArk/Qwen3.8-27B-NVFP4` snapshot from `source.lock.json` with local DSpark
draft weights. Set `MODEL_DIR` to that NVFP4 snapshot and `DRAFT_MODEL_DIR` to
the existing DSpark snapshot. It uses FlashInfer, FP32 Mamba SSM,
`extra_buffer_lazy`, DSpark block size 7, and unquantized draft loading. The
profile is explicitly unqualified and defaults to GPU 0 on host port 11443;
the launcher performs no model downloads.

The cache directory is derived from cache schema v1, image digest, source
revision, and profile unless `CACHE_DIR` is set. Torch, Triton, and FlashInfer
caches are separate and must not be shared across image/source ABI changes.

To build the deterministic Phase 3A worksheet, run
`python3 bench/capacity.py`. Its `L` values are observed input-plus-output
shapes, not advertised context limits. Supply measured fixed allocation,
resolved server pools, and free-VRAM headroom separately; the calculator does
not establish that any cell fits a 96 GB GPU.
