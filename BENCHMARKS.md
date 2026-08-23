# Benchmarks

`bench/smoke.py` checks health, `/v1/models`, deterministic text, structured
tool-call, streaming, reasoning controls, and image requests. It saves
request/response JSON and server identity/capacity when available. It does not
qualify GPU capacity or quality.

`bench/capacity.py` generates the Phase 3A predicted KV/GDN sizing worksheet:

```sh
python3 bench/capacity.py --json bench/results/capacity-worksheet.json \
  --markdown bench/results/capacity-worksheet.md
```

The worksheet covers the seven observed total-token lengths, C1/C2/C4,
float32/bfloat16 state, FP8/BF16 KV, all four state-cache strategies, and
none/MTP/DSpark/DFlash2. It uses the approximate decimal byte values in
`TODO.md`; decimal fields are labeled kB/MB/GB and binary fields KiB/MiB/GiB. MTP and DFlash2 retain
an unresolved `D` rather than inventing a memory effect. The balanced ratio is
independent of concurrency, while total state and KV memory scale with the
number of live requests. These are predictions only: fixed model/graph/runtime
memory, measured free VRAM, and headroom must be recorded from a server run.
No profile is claimed to fit 96 GB without those measurements.

The in-checkpoint EAGLE candidates (`tp1-bf16-eagle-candidate` and
`tp2-bf16-eagle-candidate`) must be benchmarked against the corresponding safe
no-spec profile with identical prompts, token limits, and cache settings.
Report acceptance only after deterministic correctness, resolved capacity,
and stability evidence is retained; their names do not imply qualification.

The `tp1-bf16-dflash-candidate` uses the local DFlash2 draft with exactly eight
draft tokens and must be benchmarked against `tp1-bf16-safe` using identical
requests, limits, and cache settings. It is explicitly unqualified; retain
the raw launch identity, draft snapshot identity, acceptance/capacity data,
and stability evidence separately from the no-spec baseline.

The `tp1-nvfp4-dspark-candidate` is a separate, unqualified quantization and
speculation candidate. Compare it with the matched no-spec NVFP4 launch using
the same request panel, and retain model/draft revisions, DSpark block size 7,
draft quantization mode `unquant`, resolved capacity, and raw acceptance and
stability evidence.

Store runs under `bench/results/<UTC>-<profile>/` with `metadata.json`,
`responses.json`, and `server-info.json` when supported. Metadata must include
exact image/source/model identities, GPU list, TP, context, and port. Use
`--allow-no-server` only for static/harness checks; runtime tests otherwise
fail clearly when the server is unavailable.

## Phase 7 contract

`bench/phase7-minimum.json` is the versioned, hardware-independent controlled
panel. Regenerate it with:

```sh
python3 bench/benchmark_contract.py manifest bench/phase7-minimum.json
python3 bench/benchmark_contract.py validate bench/phase7-minimum.json
```

It contains decode contexts 8K/32K/100K/128K at C1/C2/C4 and cache-cold
prefill contexts 8K/32K/64K/100K/128K/200K/near-262K, with every valid
1K/4K/16K output and a 32K output where the combined native budget permits.
It also includes balanced 4×(25K+4K) and asymmetric
100K+16K plus 3×(8K+4K) production-shaped cells. Manifest validation rejects
missing or duplicate cells.
Every engine cell names five fixed prompt paths, seed 0, streaming, forced
lengths, one warmup, and cold/hot cache modes. Natural stopping is a separate
suite contract and is not mixed into engine measurements.

Imported results must retain raw runs and all identity/process/cache/occupancy
metadata. `validate_result` rejects missing metrics, interval misalignment,
queueing, errors, restarts, OOMs, malformed responses, and silent clamps.
It also requires a valid immutable image digest, non-placeholder identities,
server arguments, resolved `max_running_requests`, and total/free VRAM so the
5% margin is computable. The one-second starvation gate uses measured maximum
ITL gap, not p99.
`analyze_cell` reports every accepted run, median/min/max, sample stdev/CV and
only quantiles supported by the sample count (p95 requires 20, p99 requires
100). `evaluate_gates` returns pass, fail, or unresolved; unresolved never
passes. This tooling validates real measurements and is not a load generator.

## Queued C2/C3 native-context campaign

`bench/c2-c3-native-context-campaign.json` is a queued, queue-only experiment
plan for the current TP1 NVFP4+DFlash2 `extra_buffer_lazy` winner. It splits
the multivariate question into staged C2 and C3 profiles at the native
262,144-token context limit, with a 131,072-token maximum output reserve:

- C2 pins `--max-running-requests 2` and `--max-mamba-cache-size 8`.
- C3 pins `--max-running-requests 3` and `--max-mamba-cache-size 12`.
- Both retain FP8 KV, FP32 SSM, FlashInfer, 2,048-token chunks, 0.85 static
  memory, DFlash2 with eight draft tokens, and `extra_buffer_lazy`.
- The initial profiles deliberately omit `--mamba-full-memory-ratio`; the
  approximate native-length ratio (~0.22) is a later, separate factor.
- Stages fail fast: boot/admission, near-native prefill, max-output decode,
  boundary-safe combined occupancy, then four simultaneous arrivals to expose
  C2/C3 queue waves.

The campaign is not executable yet. The existing prompt runner is sequential;
a concurrent runner/importer must preserve aligned occupancy and admission
timestamps, partial streams, exact server prompt-token counts, and timeout,
OOM, restart, HTTP-error, incomplete, and clamp outcomes before any GPU run.
The full three-repetition campaign is multi-hour because each forced-output
cell requests 131,072 tokens. No launcher profile or Phase 7 manifest is
changed by this queue plan.
