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

The campaign uses the separate `bench/c2_c3_runner.py` concurrent streaming
client and the fail-closed `bench/c2_c3_importer.py`; the sequential coding
prompt runner and Phase 7 contract remain unchanged. The version-1 raw format
is machine-described by `bench/c2-c3-run-schema.json`. A non-networked request
and barrier plan can be checked with:

```sh
python3 bench/c2_c3_runner.py dry-run \
  --spec bench/c2-c3-run-spec.example.json --output /tmp/c2-c3-dry-run.json
```

Live mode additionally requires a static server-evidence JSON object, the host
server PID, and a server-generated scheduler JSONL stream. Each scheduler line
must use schema `qwen38.server-scheduler-event`, source `server_scheduler`, and
carry a UTC timestamp, event (`queued`, `admitted`, `started`, `completed`, or
`failed`), client request ID, server process identity, and the resulting
running/queued counts. The request ID is sent as `X-Request-ID`. This is the
only accepted admission/queue source; response headers, client timing, and
aggregate-only gauges do not establish scheduler admission.

The instrumented image assigns SGLang's built-in startup warmup the reserved
ID `__sglang_c2c3_startup_warmup__`. Its scheduler event provides the PID/start
identity used by `--server-pid auto`. A measured run begins tailing at the
current JSONL end before releasing its request barrier, so startup warmup and
earlier-cell history are preserved on disk but excluded from that run.

```sh
python3 bench/server_evidence.py capture-provenance \
  --container qwen3.8-27b-sglang --profile c2 --gpu 0 \
  --output launch-provenance.json --raw-output raw-docker-inspect.json
python3 bench/server_evidence.py collect \
  --url http://127.0.0.1:11447 --profile c2 \
  --launch-provenance launch-provenance.json \
  --output server-evidence.json --raw-output raw-server-info.json
python3 bench/c2_c3_runner.py run --spec run-spec.json --output raw-run.json \
  --server-evidence server-evidence.json --scheduler-events scheduler.jsonl \
  --server-pid auto --gpu 0 --sample-interval 1
python3 bench/c2_c3_importer.py raw-run.json --output imported-run.json
```

The server-evidence object must contain immutable image/source/model/draft,
recipe, and hardware identities; flattened observed server fields; independent
launch provenance; resolved native context and profile concurrency; memory-pool
and CUDA-graph details. Pinned SGLang's `/server_info` is a flattened dataclass,
not an argv record, so it is retained as `observed_server_args`. Docker inspect
separately supplies the immutable image, source-revision label, exact container
command, model mount mapping, and GPU UUID. The initial command must omit
`--mamba-full-memory-ratio` and `--disable-cuda-graph`.
Resolved capacity records only observed runtime facts: native context, TP,
profile concurrency, token capacity, and state pin. The 131,072 output cap is
stored as `campaign_request_limits.max_output_tokens`; the planned profile port
is stored as `launch_metadata.planned_port`, separately from the observed API
endpoint. `bench/server_evidence.py` validates both sources and preserves their
raw references. The accepted image is read from the locked C2/C3 evidence
overlay, never hardcoded to its parent image.

Memory evidence is not an arbitrary metadata map. It records a non-placeholder
measurement source plus positive byte counts for the FP8 KV pool, FP32 Mamba
state pool (8 C2 or 12 C3 slots), and eight-state DFlash intermediate pool.
CUDA-graph evidence records its measurement source, enabled state, positive
memory bytes, and unique captured batch sizes covering batch one and exact
profile concurrency. If the live runtime cannot yet extract those fields, the
import remains rejected; operators must not insert descriptive placeholders.
The runner continuously samples GPU state and `/proc/PID/stat` over the request
interval and retains every SSE line/event even when the stream fails. Import
requires aligned arrivals, request-correlated scheduler admission/start,
observed exact occupancy and queueing where expected, interval-covering GPU and
process samples, at least 5% free VRAM, final server usage, and exact
forced-output count with `length` termination. Prompt, completion, reasoning,
and visible counts come only from final server usage. Optional boundary proof
comes only from exact server `prompt_tokens` usage.

SSE event gaps are never labeled token ITL. ITL/max-ITL are available only
when every content-bearing event supplies server `token_ids` paired one-to-one
with `token_timestamps_s`; otherwise the artifact explicitly records the
metric as unavailable and import rejects the qualification claim. The runtime
profile/instrumentation unit must provide the scheduler JSONL and token timing
extension before GPU campaign cells can be accepted. The full campaign remains
multi-hour, and no launcher profile or Phase 7 manifest is changed by this
harness unit.

The runtime overlay supplies that evidence path but keeps it disabled by
default. Set `SGLANG_C2C3_EVIDENCE_PATH` to an absolute JSONL filename in an
existing directory before server startup. The scheduler opens the file in
append mode with permissions `0600`; the operator must retain the file rather
than truncate it during a run. When enabled, every OpenAI Chat request must
carry a unique, nonempty `X-Request-ID` of at most 256 visible characters. The
header becomes the internal scheduler request ID, scheduler transitions are
appended to the JSONL file, and generated token IDs receive strictly increasing
API-server emission timestamps in the corresponding OpenAI SSE events. With
the environment variable absent, request IDs, SSE payloads, and scheduler I/O
retain their upstream behavior.

The runner's `--server-pid` must name the primary TP scheduler process, and the
runner must see that process through the same PID namespace. This is required
because both the JSONL event and `/proc/PID/stat` sampler identify it as
`pid:<pid>:start_ticks:<ticks>`. The JSONL path must likewise be visible at the
same location to the server and runner, for example through a shared bind mount.
The overlay does one locked `O_APPEND` write per transition and deliberately
does not `fsync` the scheduler hot path. Completed writes therefore preserve
partial evidence across client failures and normal shutdown, while a sudden
host or kernel failure may lose the final buffered filesystem writes.

The runnable `c2` and `c3` launcher profiles require `EVIDENCE_DIR` to be an
existing writable absolute host directory. Each launch reserves a new
`<profile>-<UTC>-<pid>.jsonl` target (or an unused `EVIDENCE_FILE` basename),
refuses stale targets, creates the reserved target with mode `0600`, binds the
directory read-write at `/c2-c3-evidence`, and passes the fixed container path
to the server. They
also use `--pid=host`; the host runner's `--server-pid` must be the primary
scheduler PID and matching `start_ticks` from scheduler JSONL, not an API PID
or `docker top` value. The profiles refuse a live canonical-name or port
collision and do not replace an existing container.

For the campaign overlay build, tag the already-qualified no-AVX image as the
temporary local alias `qwen38-noavx-base:c2c3-build`, verify that the alias and
immutable parent both resolve to image ID `sha256:d3346cea...`, pass the alias
as `BASE_IMAGE`, and set `BASE_HAS_NOAVX=1`. BuildKit must resolve the parent as
that alias at digest `sha256:d3346cea...`; remove the alias after the build.
The Containerfile verifies the no-AVX source guard before skipping patch
`0001`, then applies only the new evidence patch. The default build remains the
full official-base path and applies both `0001` and `0002`.
