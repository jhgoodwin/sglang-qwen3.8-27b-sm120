# Decisions

## Phase 1 scaffold (2026-08-23)

- Scope: provide a cold-machine launch and reconstruction contract without
  asserting GPU/runtime qualification.
- Decision: safe profiles disable speculation, use float32 SSM, FP8 KV,
  FlashInfer, 2,048-token chunked prefill, and conservative 0.80 static
  memory fraction. Production names remain aliases pending measurement.
- Reason: TODO.md explicitly makes cookbook speculation a candidate and
  requires a no-speculation correctness baseline.
- Decision: image and source identities remain explicit `UNRESOLVED` values
  until independently verified; the launcher rejects unresolved image refs.
- Reason: fabricating a digest or model revision would break reproducibility.
- Decision: model/cache inputs are read-only and Torch, Triton, and FlashInfer
  compile caches use image/profile-separated host namespaces.
- Reason: prevents accidental mutation and ABI cache contamination.

## Phase 7 benchmark contract (2026-08-23)

- Scope: define reproducible workload and result-analysis contracts without
  fabricating hardware measurements or embedding a load generator.
- Decision: use `bench/phase7-minimum.json` (schema `qwen38.phase7`, version 1)
  for the minimum controlled panel; engine cells use five fixed prompt paths,
  seed 0, streaming, forced output lengths, one warmup, and explicit cold/hot
  cache modes. Natural stopping remains a separate suite.
- Decision: reject imported runs when identities, required measurements,
  interval alignment, occupancy, or zero-error/restart/OOM/malformed/clamp
  conditions are absent or invalid. Retain rejected-run reasons alongside raw
  results.
- Decision: p95 requires at least 20 accepted samples and p99 at least 100;
  unsupported statistics and acceptance gates are unresolved, never inferred.
- Reason: TODO.md requires exact token semantics, matched process blocks, and
  evidence-backed tails; small fixed-path panels must not manufacture tail
  quantiles or a pass result.
- Correction: the starvation gate requires `max_itl_ms <= 1000`; p99 ITL is
  retained only for the separate mixed-load 2x comparison. Results require
  total and free VRAM (or a validated derived fraction), valid immutable
  digests/identities, resolved capacity, and raw-run references. The manifest
  validator enforces all panel cells, including both production-shaped C4
  workloads; the natural suite is explicitly unresolved until its corpus is
  pinned.
- Correction: manifest validation compares every cell and execution-contract
  field to the generated canonical contract, including production request
  token shapes. Imported evidence must use ordered timestamps, positive
  resolved capacity, strict flags/counters, and occupancy matching the cell;
  analysis rejects mixed cell IDs or process metadata shapes. The checked-in
  manifest is tested for exact equality with a freshly generated manifest.

## Host-compatible runtime overlay (2026-08-23)

- Scope: run the official Qwen3.8 SGLang image on this host, whose virtual CPU
  does not expose AVX, while preserving the cookbook CUDA and model stack.
- Decision: build the pinned `Containerfile` overlay on official base image
  `sha256:616a3e97f45191af975896cfa644279096cb31bd408a071c2e99ca7209c3cafe`
  and disable only the eager `nixl_ep` import. Qualifying runs use overlay image
  `sha256:d3346cea82545d982b7ec169f1f0f6f47834b0c4a70ec693e954a8d66111cb8d`.
- Reason: the unmodified image aborts during Python import when UCX initializes
  an AVX instruction. Qwen3.8-27B is dense and does not use NIXL MoE token
  dispatch; the overlay's `ServerArgs` import and live model boot both pass.
- Consequence: NIXL MoE dispatch is unavailable in this image. Rebuild and
  repin the overlay whenever its base image or patch changes.

## In-checkpoint EAGLE candidates (2026-08-23)

- Scope: expose reproducible TP1 and TP2 EAGLE launch candidates using the
  already-mounted Qwen3.8-27B checkpoint.
- Decision: `tp1-bf16-eagle-candidate` and `tp2-bf16-eagle-candidate` inherit
  the safe BF16/FP8-KV, float32-SSM, FlashInfer, and `extra_buffer_lazy`
  settings, then add exactly EAGLE/3 steps/top-k 1/4 draft tokens. TP2 also
  uses `--disable-custom-all-reduce` because custom all-reduce failed during
  startup on this host. Neither profile hardcodes an experiment-only request
  limit; capacity flags remain overridable.
- Consequence: both remain `unqualified_candidate` profiles until matched
  no-spec correctness, capacity, and stability evidence exists.

## Local DFlash2 candidate (2026-08-23)

- Scope: expose one reproducible TP1 DFlash2 candidate using the locally pinned
  `incoai/Qwen3.8-27B-DFlash2` snapshot without resolving or downloading it.
- Decision: `tp1-bf16-dflash-candidate` inherits the safe BF16 target settings,
  requires `DRAFT_MODEL_DIR`, mounts the draft repository root read-only, and
  adds exactly `--speculative-algorithm DFLASH`, the internal draft path, and
  `--speculative-num-draft-tokens 8`.
- Reason: the canonical snapshot uses relative blob symlinks; mounting its
  repository root preserves those links and keeps the launch reproducible.
- Consequence: the profile remains `unqualified_candidate` until matched
  no-spec correctness, capacity, and stability evidence exists.

## Local NVFP4 DSpark candidate (2026-08-23)

- Scope: expose one reproducible TP1 candidate using the pinned
  `RadixArk/Qwen3.8-27B-NVFP4` checkpoint and local DSpark draft.
- Decision: `tp1-nvfp4-dspark-candidate` requires `DRAFT_MODEL_DIR`, mounts the
  draft repository root read-only, and adds DSpark with block size 7,
  unquantized draft loading, FlashInfer draft attention, FP32 Mamba SSM, and
  `extra_buffer_lazy`. It uses a unique default port/name and remains
  `unqualified_candidate`.
- Reason: the NVFP4 snapshot metadata identifies modelopt `MIXED_PRECISION`
  weights with FP8 KV quantization; exact revisions and three shard sizes are
  recorded in `source.lock.json` for matched experiments.
- Consequence: no production or winner claim is implied until matched NVFP4
  no-spec correctness, capacity, and stability evidence exists.

## Coding prompt runner (2026-08-23)

- Scope: provide a reusable operational runner for the existing coding prompt
  directory; this does not define a new corpus or execute model output.
- Decision: enumerate `.txt` files in bytewise filename order and send each
  unchanged as one user message to Chat Completions. Requests default to
  `max_tokens=32768`, keep model/server thinking and sampling defaults intact,
  and omit temperature, top-p, top-k, and reasoning-effort overrides.
- Decision: prefer SSE streaming, retain the complete event stream and
  reconstructed content/reasoning, and record both end-to-end and post-first
  token completion rates when usage exposes completion-token counts. Prompt
  SHA-256, exact request bodies, server-info response, HTTP errors, and
  speculative-looking usage fields are retained in the machine-readable run.
  Streaming requests explicitly request final usage, use a 1,800-second
  default timeout for long coding traces, and use one monotonic clock for all
  elapsed-rate calculations; missing usage or first-token timing is reported
  as an explicit unavailable metric.

## Current generated cookbook panel (2026-08-23)

- Scope: add separate current-recipe NVFP4 no-spec and DFlash2 launch profiles;
  historical profiles and measurements remain immutable.
- Correction: the source-install pin `1cf2b8c54d81802abc15dcf23a29b9cc687bc01e`
  does not include DFlash. The current generated recipe at main
  `d1af3c89233c475fc1bf11939d86787e6cddd58c` does include the DFlash2 path.
- Decision: emit the generated recipe's Mamba controls directly: ratio 2.55
  for no-spec `extra_buffer`, 6.63 for default DFlash `extra_buffer`, and 6.12
  for high-throughput DFlash `extra_buffer_lazy`. The C1 cache-size values are
  recorded as metadata only to avoid redundant ratio and explicit-cache flags.
  All three profiles pin one running request, FP8 KV, FP32 Mamba state,
  FlashInfer, 2,048-token chunking, and 0.85 static memory.
- Runtime: the current panel uses the verified host-compatible overlay and
  base image already locked above; its recipe and source revisions are tracked
  under `runtime_variants.current-cookbook-qwen38-27b`.

## Current cookbook benchmark outcome (2026-08-23)

- Scope: record the measured outcome of the current generated NVFP4+DFlash2
  panel without turning exploratory speed into a production qualification.
- Decision: advance current `extra_buffer_lazy` as the top measured C1 and
  coding-prompt candidate in the tested panel: 261.063 median output tok/s
  (259.867 / 263.355 bracket) and 176.915923 aggregate coding-prompt
  completion tok/s across nine requests. This supersedes the historical
  130.3991 one-GPU decode-leader claim for the tested recipe, while retaining
  that original result as a historical comparison.
- Consequence: the candidate remains unqualified for C2/C4, 100K+ capacity,
  mixed-load starvation, vision, soak stability, and coding-quality gates.
  The operational comparison records token-count and time differences but
  makes no task-quality or quantization-causality claim.
