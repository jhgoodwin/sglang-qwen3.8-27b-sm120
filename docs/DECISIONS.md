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

## Queued C2/C3 native-context campaign (2026-08-23)

- Scope: split the next one-GPU capacity question into independently gated C2
  and C3 experiments at the native 262,144-token context and 131,072-token
  maximum output, without changing the measured winner or claiming capacity.
- Decision: use a separate queued manifest,
  `bench/c2-c3-native-context-campaign.json`, with explicit state pins: C2
  uses eight Mamba state slots and C3 uses twelve (four per active request).
  DFlash's eight verify/intermediate states remain separately runtime-allocated
  and captured. Both retain the current TP1 NVFP4+DFlash2
  `extra_buffer_lazy` runtime invariants.
- Reason: the existing sequential prompt runner cannot measure simultaneous
  arrivals, queue waves, aligned occupancy, or partial streams. A concurrent
  runner/importer is an execution prerequisite, and failures must remain
  distinguishable rather than being collapsed into incomplete success.
- Decision: omit `mamba_full_memory_ratio` from the initial C2/C3 profiles.
  The approximate native-length ratio (~0.22) is a later, mutually exclusive
  factor so state-pin effects are isolated first. Planned ports 11447/11448
  are metadata only; no launcher profiles are added by this plan.
- Consequence: stages A through E are queued and fail-fast, with three measured
  repetitions, a 1,024-token boundary-safe margin, and an optional exact
  131,072+131,072 cell gated on proof that transport preserves exact server
  prompt tokens. The manifest's multi-hour output-volume estimate is planning
  metadata, not a benchmark result.

## C2/C3 concurrent evidence boundary (2026-08-24)

- Scope: make the queued C2/C3 campaign measurable without weakening the
  queue-rejecting Phase 7 schema or the sequential operational prompt runner.
- Decision: use `qwen38.c2-c3-concurrent-run` version 1 as a distinct raw
  schema, with barrier-released streaming requests and fail-closed import.
  Admission, start, queue depth, and occupancy require request-correlated
  `qwen38.server-scheduler-event` records from the server scheduler. Client
  timing, HTTP headers, and aggregate-only gauges are not scheduler evidence.
- Decision: retain every raw SSE line/event and partial content/reasoning for
  failed requests. Token ITL is supported only by a server token ID and server
  timestamp pair for every emitted token; SSE chunk gaps remain explicitly
  unavailable. Forced-output success requires the exact final server
  completion count and `length` termination. Exact-boundary transport proof
  requires exact final server prompt usage.
- Consequence: the importer rejects missing, placeholder, malformed, or
  interval-misaligned server/process/GPU evidence, runtime identity drift,
  wrong occupancy, missing queue evidence, restarts, incomplete output, and a
  minimum observed free-VRAM fraction below 5%. Runtime integration must emit
  the scheduler JSONL and token timestamps before a GPU cell can qualify; this
  harness decision itself makes no C2/C3 performance or capacity claim.
- Decision: server evidence is semantic rather than a nonempty metadata bag.
  The importer parses the preserved argument vector and requires the exact
  C2/C3 concurrency and Mamba pins plus the locked TP1 FP8-KV/FP32-SSM,
  FlashInfer, chunking, memory, `extra_buffer_lazy`, and DFlash8 values. It
  rejects duplicate/conflicting flags, the initial ratio flag, disabled CUDA
  graphs, and pool/graph objects without measured byte counts, state shape,
  capture coverage, and a non-placeholder source.

## C2/C3 runtime evidence overlay (2026-08-24)

- Scope: connect the concurrent harness evidence contract to the pinned
  SGLang production request, scheduler, output, and OpenAI SSE paths without
  adding campaign profiles or making a GPU qualification claim.
- Decision: patch the runtime with a separate `0002` overlay controlled only
  by `SGLANG_C2C3_EVIDENCE_PATH`. Disabled mode preserves upstream request IDs,
  SSE fields, and scheduler I/O. Enabled mode requires `X-Request-ID`, uses it
  as the internal request ID, records post-transition queue/running counts in
  scheduler JSONL, and pairs scheduler output IDs with strictly increasing
  API emission timestamps on OpenAI SSE events.
- Decision: scheduler evidence uses one locked, compact `O_APPEND` write per
  transition without `fsync`. This retains every completed line with low hot
  path overhead, but does not claim durability against sudden host or kernel
  failure. The runner and primary TP scheduler must share a PID namespace so
  `pid:<pid>:start_ticks:<ticks>` is independently reproducible from `/proc`.
- Consequence: static patched-source tests prove the exact pinned call sites,
  compilation, JSONL state transitions, request correlation, and SSE token
  fields. They make no admission, memory, throughput, queue-wave, or GPU claim;
  those remain gated on a built overlay and later live campaign profiles.

## C2/C3 runtime evidence bridge (2026-08-24)

- Scope: expose directly measured pool, DFlash, CUDA-graph, and resolved-capacity evidence through the opt-in scheduler state used by `/server_info`.
- Decision: `resolved_capacity` contains observed runtime facts only. Campaign output limits and planned ports remain explicit `campaign_request_limits` and `launch_metadata` fields. Because pinned SGLang returns a flattened `ServerArgs` dataclass rather than argv, the collector records an explicit `observed_server_args` mapping and rejects any missing or drifted campaign field.
- Decision: `/server_info` cannot attest its container. Image ID/reference, source-revision label, exact launch command, host-to-container model mounts, and GPU UUID come from a separately retained Docker/NVIDIA provenance artifact. The image must match the locked `c2-c3-evidence-overlay` and runnable profile; its parent overlay is not accepted.
- Decision: `--server-pid auto` is allowed only after a valid scheduler JSONL transition identifies a live `/proc` PID/start-ticks pair; stale, malformed, or multiple identities fail closed.
- Decision: when evidence mode is enabled, SGLang's built-in `_execute_server_warmup` adds the reserved request ID `__sglang_c2c3_startup_warmup__`. This is assigned only at the internal warmup client boundary; external chat requests still require `X-Request-ID`. The append-only warmup event remains available for PID bootstrap, while each measured runner tail starts at the JSONL EOF observed before its barrier so warmup and earlier cells cannot enter that cell's correlation set.
- Consequence: no benchmark result is importable from descriptive metadata or client timing, and the live server must expose positive measured pool/graph fields before a GPU probe.

## C2/C3 launcher evidence profiles (2026-08-24)

- Scope: make the queued C2 and C3 server processes independently launchable
  with the already-built evidence overlay; this does not qualify either
  profile or claim measured memory, throughput, pool, or CUDA-graph behavior.
- Decision: `c2` and `c3` use the immutable evidence image digest, ports
  11447/11448, the canonical production container name
  `qwen3.8-27b-sglang`, and the measured TP1 NVFP4+DFlash2 recipe. They pin
  two/three running requests and eight/twelve Mamba cache states while
  omitting `--mamba-full-memory-ratio`.
- Decision: each C2/C3 launch requires an existing writable absolute
  `EVIDENCE_DIR`. The launcher reserves a unique per-profile JSONL filename,
  refuses any pre-existing target, creates the reserved target with mode
  `0600`, binds the directory read-write at
  `/c2-c3-evidence`, and passes `SGLANG_C2C3_EVIDENCE_PATH` to the server.
  Model, draft, and HF-cache inputs remain read-only.
- Decision: C2/C3 alone use `--pid=host` so the host runner can validate the
  scheduler `pid:<pid>:start_ticks:<ticks>` identity from JSONL against
  `/proc`; the runner's `--server-pid` is the primary scheduler PID from the
  evidence, not an API PID or `docker top` result. Name and port collisions
  are rejected without replacing containers.
- Consequence: the broadened PID visibility and evidence bind are limited to
  these opt-in campaign profiles. The exact campaign maximum output remains
  request metadata and is not emitted as a server argument.
- Decision: the evidence overlay build uses temporary local parent alias
  `qwen38-noavx-base:c2c3-build` only after its image ID is verified equal to
  immutable parent `sha256:d3346cea...`; BuildKit resolves that alias at the
  same digest, and the alias is removed after the build. The source lock keeps
  the upstream official base distinct from this actual parent and build path.
