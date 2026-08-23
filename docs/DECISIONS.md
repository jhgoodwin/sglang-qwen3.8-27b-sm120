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
