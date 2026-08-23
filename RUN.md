# Run

Resolve the local snapshot and set `MODEL_DIR` (default
`/data/models/Qwen3.8-27B`) and `HF_CACHE_HUB` (default `/data/models`). Both
are mounted read-only. Replace the unresolved immutable `IMAGE` with a
verified `registry/name:tag@sha256:<64 hex>` and record it in
`source.lock.json`. Run `./scripts/validate-scaffold.sh`, then
`PROFILE=tp1-bf16-safe ./serve.sh`.

The default host endpoint is `127.0.0.1:11436` and maps to container port
8000.

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

The cache directory is derived from cache schema v1, image digest, source
revision, and profile unless `CACHE_DIR` is set. Torch, Triton, and FlashInfer
caches are separate and must not be shared across image/source ABI changes.

To build the deterministic Phase 3A worksheet, run
`python3 bench/capacity.py`. Its `L` values are observed input-plus-output
shapes, not advertised context limits. Supply measured fixed allocation,
resolved server pools, and free-VRAM headroom separately; the calculator does
not establish that any cell fits a 96 GB GPU.
