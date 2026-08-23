# Qwen3.8-27B on dual RTX PRO 6000 Blackwell Max-Q: qualification plan

## Goal

Produce a reproducible SGLang repository and a measured recommendation for
serving Qwen3.8-27B on this workstation. The primary decision is the best
configuration for coding-agent workloads with more than 100K tokens of usable
context and 1-4 simultaneous requests. The result must say when one 96 GB SM120
GPU is sufficient, quantify what TP2 buys or costs over PCIe, and qualify image
input for screenshot review or computer-use workloads.

This document is an experiment plan, not a claim that the cookbook defaults are
already optimal for this machine. The cookbook's RTX PRO 6000 coverage used an
8,192-token input, 1,024-token output, and concurrency 1. It does not establish
100K+ capacity, long-context quality, vision correctness, or TP2 performance.

## User requirements and decision priorities

- [ ] Serve the BF16 `Qwen/Qwen3.8-27B` checkpoint currently being downloaded
  through `HF_CACHE_HUB=/data/models`; do not start or duplicate that download.
- [ ] Preserve the model's native 262,144-token context if the hardware can do
  so reliably; treat 100K usable context as the minimum target.
- [ ] Optimize in this order unless measurements justify a different tradeoff:
  1. correctness and stability;
  2. interactive C1 coding latency;
  3. C2/C4 coding-agent fan-out;
  4. long-prompt time to first token (TTFT);
  5. aggregate throughput, memory use, and power.
- [ ] Prefer a one-GPU production profile if it meets the context, concurrency,
  quality, and latency gates. Leave the other GPU genuinely free for unrelated
  work, rather than merely idle inside a TP2 process.
- [ ] Quantify TP1 versus TP2 rather than assuming tensor parallelism is faster.
  The mainboard's x16 lanes are split between the cards, and each GPU is confirmed
  at PCIe Gen5 x8. TP2 still adds collectives to every layer and may
  hurt low-batch decode even when it helps prefill or capacity. Capture actual
  upstream switch/root-complex path and P2P behavior under load.
- [ ] Compare TP2 with two independent TP1 replicas for C2/C4. For this workload,
  replication may be a better use of two GPUs than tensor parallelism.
- [ ] Support the OpenAI-compatible chat endpoint, structured Qwen tool calls,
  reasoning controls, streaming, and image inputs.
- [ ] Use `127.0.0.1:11436` as the default effective host endpoint, matching the
  current DeepSeek-V4-Flash `serve.sh`. The container may listen on port 8000
  internally, with `127.0.0.1:11436:8000` as the default publish mapping.
- [ ] Treat `unsloth/Qwen3.8-27B-GGUF` as an optional checkpoint/backend branch.
  Start with its BF16 and `UD-Q8_K_XL` variants only if GGUF is required or a
  measured llama.cpp comparison is useful; do not delay the native safetensors
  SGLang baseline for these downloads.
- [ ] Make every published result reproducible from pinned model, image/source,
  dependency, launcher, benchmark, and hardware identities.

## Facts to preserve from authoritative sources

- Qwen3.8-27B is a dense hybrid vision-language model with 48 Gated DeltaNet
  (GDN) layers and 16 full-attention layers. Its context is natively 262,144
  tokens and can be extended to 1M; 1M is out of scope until native-context
  operation is qualified.
- SGLang serves it through the Qwen vision-language path, so the vision tower is
  active even for the text checkpoint recipe.
- RTX PRO 6000 Blackwell is SM120. The cookbook recommends
  `--attention-backend flashinfer`; `trtllm_mha` is SM100-only.
- On SM120, the cookbook recommends `--chunked-prefill-size 2048`. Larger chunks
  can block decode for hundreds of milliseconds under mixed prefill/decode load.
- The cookbook emits `--reasoning-parser qwen3` and
  `--tool-call-parser qwen3_coder` for agent harnesses.
- Hybrid GDN memory has two distinct consumers after weights: paged attention KV
  and a worst-case-reserved recurrent-state pool. Ordinary transformer KV-only
  capacity calculations are insufficient.
- For this model, one GDN state slot is approximately 153.9 MB at float32 and
  78.4 MB at bfloat16. FP8 KV is approximately 32.8 KB/token and BF16 KV is
  approximately 65.5 KB/token.
- `extra_buffer_lazy`, the cookbook's high-throughput strategy, uses four state
  slots per active request (`S=4`). `extra_buffer` uses five, the no-buffer mode
  uses three, and disabling radix cache uses one.
- DSpark's default block size is seven and its verify window contributes eight
  intermediate states (`D=8`). The cookbook's balanced sizing equation is:

  ```text
  mamba_full_memory_ratio = (S + D) * state_bytes
                            / (average_total_request_tokens * kv_bytes_per_token)
  ```

- The alternative explicit state-pool pin is
  `--max-mamba-cache-size = target_concurrency * S`. `D` is excluded from this
  explicit pin because verify intermediates are allocated separately.
- The default `--mamba-full-memory-ratio 0.9` can silently cap concurrency. The
  server's resolved `max_running_requests` startup line is required for every
  capacity result; also save `/get_server_info` when that release exposes it.

## Provisional acceptance thresholds to freeze before benchmarking

Replace these only before the first measured campaign, with the change and
rationale committed. Do not move a threshold after seeing an unfavorable result.

- Minimum one-request envelope: at least 100,000 input tokens plus 16,384
  generated-token capacity, without auto-truncation, within the configured
  `--context-length`. The native-context target is a combined input-plus-output
  budget of 262,144, not 262,144 input tokens followed by extra output.
- Production concurrency: admit four live requests and sustain the explicitly
  published simultaneous-token shape. The initial minimum shapes are four
  requests of 25K input + 4K output, and an asymmetric mix of one 100K input +
  16K output request with three 8K input + 4K output requests. Queueing beyond
  qualified shapes is acceptable; silently resolving the cap below four is not.
- VRAM safety: at least 5% measured free VRAM after graph capture during the
  worst qualified prefill/decode mix, followed by a soak with zero OOMs/restarts.
- Decode starvation: while a long prefill joins an active decode, no inter-token
  gap over 1 second and mixed-load p99 ITL no more than 2x isolated p99 ITL.
- Material quality regression: any new deterministic corruption/divergence,
  tool/API failure, or more than 2 percentage points absolute loss on the frozen
  coding task pass rate. Long-context retrieval must remain 100% on the small
  deterministic position-probe corpus.
- Meaningful TP2 gain: at least 10% improvement in a priority TTFT/ITL/throughput
  metric without failing another gate, or a required context/concurrency envelope
  that TP1 cannot admit. Always publish smaller measured differences as facts.

## Phase 0: capture the actual machine and immutable inputs

- [x] Confirm PCIe lane allocation: both GPUs run at PCIe Gen5 x8 from the
  mainboard's bifurcated x16 lane allocation (user-confirmed 2026-08-23).
- [ ] When run from a shell with GPU access, capture to a machine-readable
  environment record:
  - exact GPU names, VRAM, VBIOS, clocks, persistence mode, ECC state, and
    enforced power limit;
  - `nvidia-smi topo -m`, upstream PCIe path, P2P capability, NUMA placement,
    CPU model, RAM, kernel, Docker, NVIDIA Container Toolkit, and driver;
  - idle GPU processes and display attachment;
  - ambient/steady temperatures and whether the Max-Q cards sustain the target
    300 W limit.
- [ ] Record that Codex's current sandbox cannot communicate with the NVIDIA
  driver; do not interpret that as a host hardware failure. Run the capture
  from the host or from the serving container.
- [ ] After the ongoing BF16 download completes, resolve its exact Hugging Face
  snapshot revision, local snapshot path under `/data/models`, shard count,
  total bytes, and file hashes/ETags. Never benchmark a mutable model name alone.
- [ ] Verify enough disk space for the BF16 snapshot, draft checkpoints,
  image layers, compiled kernels, raw benchmark responses, and result artifacts.
- [ ] Pin either the cookbook-tested SGLang commit
  `1cf2b8c54d81802abc15dcf23a29b9cc687bc01e` and its exact environment, or a
  newer commit that contains required correctness fixes. Record why the selected
  revision supersedes the cookbook pin.
- [ ] Pin the container by digest, not just
  `lmsysorg/sglang:dev-qwen38-27b-dflash2`, and record SGLang, FlashInfer,
  PyTorch, CUDA, sgl-kernel, and driver versions. FlashInfer used with MTP must
  have the `uniform_q_len` prefill-plan support newer than `0.6.15.post1`.
- [ ] Give every source composition a separate persistent compiled-kernel cache
  namespace. Do not mix caches across image/source revisions.

Deliverable: a locally generated, ignored `environment.json` (start from `environment.example.json`) plus a short source/compatibility note.

## Phase 1: scaffold a reproducible, safe repository

Reuse the useful shape of `~/src/sglang-deepseek-v4-flash-sm120`, but do not
copy its DeepSeek-specific kernels, patches, environment flags, context limit,
or benchmark conclusions.

- [x] Add a checked launcher with explicit, validated inputs for model path,
  cache path, image digest, GPU list, TP size, context length, port, and profile.
- [x] Default to host `127.0.0.1:11436`, publishing to the serving container's
  internal port (initially 8000). Retain an explicit `PORT` override for tests,
  but use 11436 in the production wrapper, health checks, examples, and client
  configuration. Document authentication before exposing beyond localhost.
- [x] Detect and clearly report a port collision. DeepSeek and Qwen cannot both
  bind host port 11436 simultaneously; do not silently stop or replace the other
  service.
- [x] Mount the model read-only and use persistent, image-specific Torch/Triton/
  FlashInfer compilation caches.
- [ ] Use adequate shared memory, unlimited memlock, and the expandable CUDA
  allocator, then verify each is still useful for the selected release.
- [x] Validate that the number of visible GPUs equals the requested TP size.
- [x] Add named profiles rather than one opaque command, initially:
  - `tp1-bf16-safe`: no speculation, float32 SSM, conservative capacity;
  - `tp1-bf16-production`: the winner after qualification;
  - `tp2-bf16-safe` and `tp2-bf16-production`;
  - optional `replica0`/`replica1` TP1 profiles on separate ports.
- [x] Start from this cookbook-derived BF16/DSpark candidate, but do **not** make
  speculation the safe baseline:

  ```text
  --model-path Qwen/Qwen3.8-27B
  --trust-remote-code
  --kv-cache-dtype fp8_e4m3
  --mem-fraction-static 0.85
  --attention-backend flashinfer
  --chunked-prefill-size 2048
  --reasoning-parser qwen3
  --tool-call-parser qwen3_coder
  --speculative-algorithm DSPARK
  --speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark
  --speculative-draft-attention-backend flashinfer
  --mamba-radix-cache-strategy extra_buffer_lazy
  --mamba-ssm-dtype float32
  ```

- [x] Add health, `/v1/models`, deterministic text, structured tool-call, and
  image smoke tests. Save server arguments and resolved capacity with results.
- [x] Add release/source lock files, a cache schema, `README.md`, `RUN.md`,
  `BENCHMARKS.md`, and a machine-readable `bench/results/` convention.

Deliverable: a cold-machine launch path whose first request succeeds without
manual edits and whose exact source identities can be reconstructed.

## Phase 2: establish correctness-first TP1 baselines

### 2A. Boot and memory accounting

- [x] Boot BF16 weights on GPU 0 with no speculative decoding, float32 SSM,
  FlashInfer attention, a 2,048 prefill chunk, C1 graph capture, and a deliberately
  conservative context/capacity setting.
- [ ] Save startup logs and measure weights, vision tower, CUDA graphs, non-torch
  allocations, KV pool, GDN state pool, and free safety margin separately.
- [x] Confirm the reported model id, tokenizer/chat template, native context,
  `max_req_input_len`, `max_total_num_tokens`, and resolved running-request cap.
- [ ] Set and verify `--context-length` explicitly. Confirm the server enforces
  input plus maximum output within it, rejects rather than silently truncates an
  oversized request, and exposes the intended maximum request-token budget.
- [ ] Repeat on GPU 1 to detect card-specific thermal, PCIe, or capacity variance.
- [ ] Run short deterministic prompts across independent restarts and verify
  identical token IDs before tuning performance.

### 2B. API and coding quality gate

- [x] Verify streaming and non-streaming Chat Completions.
- [ ] Verify thinking enabled/disabled, supported `reasoning_effort` values,
  `preserve_thinking`, finish reasons, stop handling, and maximum output behavior.
- [ ] Verify single and parallel tool calls are emitted as structured
  `tool_calls`, not raw `<tool_call>` text. Include malformed-argument recovery.
- [ ] Create a small, versioned coding corpus representative of actual use:
  repository navigation, bug diagnosis, patch generation, test repair, long-file
  editing, tool use, and answers requiring facts near the start/middle/end of a
  long prompt. Keep expected checks executable where possible.
- [ ] Record parse success, compile/test pass rate, task success, unwanted empty
  answers, repetition, premature EOS, and output truncation. Do not use token
  throughput as a proxy for coding quality.

Gate: no tuning candidate advances if it corrupts output, loses tool-call
structure, regresses the coding corpus materially, or produces unexplained
non-determinism under greedy decoding.

## Phase 3: find the one-GPU 100K+ capacity envelope

### 3A. Build a memory model before sweeping

- [ ] Measure fixed memory at each graph batch size and derive available dynamic
  memory at each candidate `--mem-fraction-static` value.
- [ ] Generate a sizing worksheet for all relevant combinations of:
  - average total request length `L`: 16K, 32K, 64K, 100K, 128K, 200K, 262K;
  - target concurrency: 1, 2, 4;
  - SSM dtype: float32, bfloat16;
  - KV dtype: FP8, BF16;
  - state strategy: `extra_buffer_lazy` first, then only useful alternatives;
  - speculation: none, in-checkpoint MTP, DSpark, and DFlash2 if stable.
- [ ] Calculate both the balanced ratio and explicit
  `--max-mamba-cache-size`. Label decimal/binary units and predicted headroom.
- [ ] Treat `L` as observed input plus output length, not the advertised context
  limit. Produce at least three workload profiles: ordinary coding, long-repo
  coding, and near-native-context.
- [ ] Verify predictions against resolved server pools and observed peak VRAM.
  Investigate discrepancies rather than adjusting pins blindly.

### 3B. Admission and near-limit tests

- [ ] For TP1, test exact prompt lengths 8K, 32K, 64K, 100K, 128K, 200K, and
  near 262K with output reserves of 1K, 4K, 16K, and 32K where meaningful.
- [ ] Test C1/C2/C4 combinations whose simultaneous live tokens reflect real
  use, including asymmetric loads such as one 128K request plus three short
  coding requests. A context limit alone does not prove aggregate capacity.
- [ ] At each cell, record admission success, request queueing, OOM/restart,
  TTFT, prompt throughput, inter-token latency (ITL), output throughput, peak
  VRAM, KV-cache utilization, GDN slots, and resolved concurrency cap.
- [ ] Include a near-limit request with content-dependent checks at the beginning,
  middle, and end. Allocating 262K tokens is not proof the model uses them well.
- [ ] Soak the winning C1/C2/C4 profiles with repeated cache-cold and cache-hot
  long prompts. Include cancellation, disconnect, and a new short request arriving
  during long prefill.
- [ ] Require zero OOMs, server deaths, malformed responses, and silent
  concurrency clamps. Retain at least 5% measured VRAM safety margin after
  warmup/graph capture unless a smaller margin survives an explicit soak.

Deliverable: a table of **usable** TP1 input/output/concurrency envelopes, not
just the largest `--context-length` that boots.

## Phase 4: tune the important knobs with controlled comparisons

Change one factor at a time from the safe TP1 baseline. Run a quick screen,
then repeat finalists in matched process blocks to separate small effects from
startup/thermal variance.

### 4A. KV and recurrent-state precision

- [ ] Compare FP8 versus BF16 KV at 32K, 100K, 128K, and near 262K. Measure
  capacity and speed, then run long-context retrieval and coding-quality checks.
- [ ] Compare float32 versus bfloat16 SSM state at C1/C2/C4, with and without
  the eventual speculative decoder. BF16 roughly halves state memory, but is an
  accuracy decision first.
- [ ] Add a recurrent-state drift test: generate long greedy outputs, compare
  against the float32/no-spec oracle, and rebuild selected prefixes through
  ordinary prefill to distinguish accumulated state drift from local context.
- [ ] Do not qualify BF16 SSM solely because short benchmark scores match.

### 4B. State-cache allocation and prefix reuse

- [ ] Compare `extra_buffer_lazy` (`S=4`) with `extra_buffer` (`S=5`) for shared
  coding prefixes and with no-buffer (`S=3`) if supported and correct.
- [ ] Test `--disable-radix-cache` (`S=1`) only as a capacity-oriented control;
  quantify the loss when agents repeatedly reuse a system prompt/repository
  prefix.
- [ ] Compare the calculated ratio with an explicit max-state-cache pin for
  C1/C2/C4. Confirm neither starves KV nor silently caps running requests.
- [ ] Test cache-hot multi-turn coding conversations and cache-cold unrelated
  prompts separately; report hit rate and latency rather than averaging them.

### 4C. Prefill, scheduling, and CUDA graphs

- [ ] Sweep chunked prefill 1,024 / 2,048 / 4,096 / 8,192 at C1 and under an
  asymmetric mixed load. Report decode ITL stalls as well as aggregate speed.
- [ ] Sweep graph max batch sizes 1 / 2 / 4 / 8 and measure their fixed VRAM
  cost. Do not capture batches irrelevant to the 1-4 request target without a
  demonstrated benefit.
- [ ] Test max-running-requests 1 / 2 / 4 and a small queue. Confirm a long
  chunked prefill cannot starve waiting short requests.
- [ ] Compare FlashInfer with Triton only as a diagnostic/fallback or if a
  correctness issue requires it; do not spend the main matrix on unsupported
  backends.
- [ ] Screen memory fraction conservatively (for example 0.80, 0.85, 0.88,
  0.90, 0.92) and stop before unsafe cells. Every value requires cold boot,
  graph capture, long prefill, and mixed-load validation.

### 4D. Speculative decoding

- [x] Establish no-speculation results first.
- [x] Test the in-checkpoint MTP recipe:

  ```text
  --speculative-algorithm EAGLE
  --speculative-num-steps 3
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4
  ```

- [ ] Test DSpark with pinned `RadixArk/Qwen3.8-27B-DSpark`; record its exact
  revision, inferred/explicit gamma, draft memory, acceptance length/rate, and
  verify overhead. Include `--speculative-draft-attention-backend flashinfer`
  unless the pinned release's exact recipe establishes a different backend.
- [ ] Before broad speculative sweeps, run the user's exact cookbook selection—
  BF16 weights, FP8 KV, float32 SSM, DSpark, `extra_buffer_lazy`, and FlashInfer
  target/draft attention—at C1/C2/C4 with 100K and 128K input shapes. Treat it as
  a candidate subject to the same correctness gates, not as the oracle.
- [x] Test DFlash2 only on a source/image revision where it is supported and
  stable; pin its draft revision and eight-token block size.
- [ ] For each method, report decode forward passes/s separately from useful
  output tokens/s and accepted tokens/step. A path-dependent acceptance gain is
  not an engine-kernel speedup.
- [ ] Repeat the full coding, tool-call, long-output, and recurrent-state gates.
  Specifically resolve or work around SGLang issue #35150 before recommending
  DSpark; float32 state reportedly delays but may not eliminate divergence.
- [ ] Test prefix-cache hits, cancellation, mixed prompt lengths, and independent
  restarts. Disable speculation in the production recommendation if correctness
  is not convincingly equivalent to the base decoder.

Deliverable: a small Pareto set (safe, low-latency, and max-throughput), with
all non-winning knobs removed from the final launcher.

### 4E. Deferred-by-default GGUF/backend branch

GGUF is primarily a storage and quantized-weight ecosystem, not inherently a
better checkpoint for SGLang. Current SGLang documentation lists GGUF loading,
but generic format support does not prove that Qwen3.8's hybrid GDN, Unsloth's
dynamic quant layouts, MTP, vision tower, or SM120 kernels are supported or
fast. Unsloth's model card documents llama.cpp as its direct GGUF path.

- [ ] Keep this whole branch deferred unless native BF16/official FP8/NVFP4
  cannot meet the required envelope, GGUF is specifically required by a chosen
  client/runtime, or a llama.cpp comparison becomes an explicit decision target.
- [ ] Do not download GGUF until the native BF16 baseline has a measured memory
  breakdown and the compatibility gates below show a reason to do so.
- [ ] Pin the exact files/revisions and verify published sizes before download.
  The model card currently reports approximately 54.7 GB for GGUF BF16 and
  31.5 GB for `UD-Q8_K_XL`.
- [ ] Clarify the decision being tested:
  - GGUF BF16 is a format/backend control. It should not have lower perplexity
    than the same source BF16 safetensors merely because it is GGUF.
  - `UD-Q8_K_XL` is the high-quality quantized candidate. Its roughly 23 GB
    smaller weight file may release VRAM for cache/state, but only if the chosen
    runtime keeps that saving on GPU and does not add larger workspaces or
    dequantized copies.
- [ ] First test SGLang GGUF compatibility in a disposable profile: tokenizer
  and chat template, all 64 hybrid layers, GPU offload, TP1/TP2, FlashInfer,
  GDN-state allocation, FP8/BF16 KV controls, MTP/speculation, tool parser, and
  image input. A successful load and one text response are not qualification.
- [ ] If SGLang falls back to slow/general kernels or lacks a required feature,
  move GGUF to a pinned llama.cpp comparison rather than patching it into the
  main SGLang matrix.
- [ ] For llama.cpp, pin the source/binary and CUDA build options; confirm
  Qwen3.8 hybrid-GDN support, full GPU offload, CUDA graph behavior, KV/cache
  dtypes, recurrent-state placement, tensor split across both PCIe GPUs, tool
  calling, reasoning controls, and OpenAI API compatibility.
- [ ] Vision requires a matching supported multimodal projector/runtime path.
  The Unsloth repository page does not currently expose an `mmproj` in its model
  card, and generic llama.cpp multimodal support is not evidence that this exact
  Qwen3.8 GGUF works. Keep GGUF text-only unless an exact projector and
  ground-truth vision tests pass.
- [ ] Compare native BF16 safetensors/SGLang, GGUF BF16, and GGUF Q8 on identical
  greedy logits/tokens, long-context retrieval, coding tasks, tool calls, memory,
  TTFT, ITL, output throughput, power, startup, and C1/C2/C4 capacity.
- [ ] Add perplexity or token-level cross-entropy on a fixed held-out corpus as
  a sensitive quantization diagnostic, but use end-task coding and long-context
  quality for the production decision.
- [ ] Attribute reclaimed capacity correctly: weights, static runtime memory,
  attention KV, GDN recurrent state, and graph/workspace memory must be reported
  separately. Do not describe all freed VRAM as KV cache.

Gate: adopt a GGUF profile only if it is feature-complete for its stated scope
and offers a measured quality/capacity/performance point not already covered by
native BF16, official FP8, or NVFP4 checkpoints.

## Phase 5: quantify TP2 and the best use of the second GPU

### 5A. TP1 versus TP2, matched

- [ ] Verify TP divisibility/support for this architecture and vision tower.
- [ ] Run the full TP1/TP2 comparison for BF16, the intended initial production
  checkpoint. Defer FP8/NVFP4 TP2 matrices unless one becomes a production
  finalist; do not conflate weight-precision and topology effects.
- [ ] Record actual topology and collective backend. Start with standard NCCL;
  port DeepSeek's PCIe-IPC/custom-all-reduce choices only if Qwen profiling shows
  eligible collectives and a measured benefit.
- [ ] Compare TP1 and TP2 with the same source, precisions, decoder, request
  tokens, sampling, cache state, graph batch, and safety margin at:
  - cold prefill: 8K / 32K / 64K / 100K / 128K / 200K / near 262K;
  - steady decode: representative 8K, 32K, 100K contexts at C1/C2/C4;
  - mixed load: one 100K-128K prefill arriving while 1-3 requests decode;
  - real coding corpus and long-output tasks.
- [ ] Record per-request TTFT, ITL p50/p95/p99, prompt/output tok/s, request
  latency, GPU utilization, per-GPU VRAM, PCIe throughput, collective time,
  power, temperature, and energy/request.
- [ ] Measure the TP2 capacity gained: maximum safe context/output reserve and
  simultaneous live-token envelope. Do not report only percentage speedup.
- [ ] Profile representative C1 decode and 128K prefill runs to attribute TP2
  gains/losses to compute, collectives, scheduling, or memory pressure.

### 5B. TP2 versus two TP1 replicas

- [ ] Run two identical TP1 servers, one per GPU, with isolated compiled caches
  and ports. Route C2/C4 requests round-robin first; test cache-aware routing only
  if prefix reuse makes it worthwhile.
- [ ] Compare aggregate throughput, per-user tail latency, fault isolation, cache
  duplication, and ability to admit a single 100K/128K/262K request.
- [ ] Test realistic imbalance: one long session plus several short ones. Record
  whether routing leaves capacity stranded or avoids TP collective overhead.
- [ ] Keep a decision table:

  | Question | TP1 | TP2 | 2 x TP1 replicas |
  |---|---:|---:|---:|
  | Maximum safe C1 context + output reserve | TBD | TBD | same as TP1 |
  | C1 TTFT / ITL | TBD | TBD | TBD |
  | C2/C4 per-user tail latency | TBD | TBD | TBD |
  | Aggregate output throughput | TBD | TBD | TBD |
  | Free GPU available for other work | yes | no | no |
  | Prefix-cache duplication | none | none | yes |
  | Failure isolation | N/A | coupled | independent |

Decision rule: recommend TP2 only if it enables a required envelope TP1 cannot
meet or produces a meaningful measured latency/throughput improvement after
collective overhead. Otherwise make TP1 the default. If both GPUs are dedicated
to serving and requests are independently routable, compare TP2 against replicas,
not just against one server.

## Phase 6: vision and computer-use qualification

- [ ] Before testing, check the status/fix revision of SGLang issue #35345
  (Qwen3.6/Qwen3.8 multimodal mRoPE positions reaching a 1D fused QK
  RMSNorm+RoPE kernel). Do not call vision production-ready based on a response
  that merely looks plausible.
- [ ] Verify OpenAI `image_url` inputs from data URLs and permitted local/remote
  sources, single and multiple images, supported formats, resolution limits,
  and clear failures for malformed/oversized images.
- [ ] Build a fixed screenshot corpus with deterministic ground truth:
  - OCR for terminals, editors, dialogs, menus, and tiny text;
  - UI element identification and pixel/normalized bounding boxes;
  - state comparison between before/after screenshots;
  - error-message diagnosis and next-action selection;
  - multi-monitor, scaled, dark/light, occluded, and high-resolution cases.
- [ ] Include reference checks against a known-correct model/backend and, where
  possible, executable UI targets. Score OCR, element selection, grounding,
  action validity, hallucinations, and refusal to guess when evidence is absent.
- [ ] Measure image preprocessing time, vision tokens, TTFT, peak VRAM, and text
  decode ITL on TP1 and the selected TP2 comparison. Test image requests mixed
  with long text requests at C2/C4.
- [ ] Test multi-turn computer-use conversations with repeated and changed
  screenshots; determine what the radix cache reuses and whether stale visual
  context causes errors.
- [ ] Run vision with the final precision/speculation candidate through the same
  long-output and state-drift gates. Maintain a text-safe production profile if
  vision requires a different source revision or backend.
- [ ] For every speculative vision candidate, run a matched no-speculation image
  control and compare decoded content/grounding, not just latency and acceptance.

Gate: no known multimodal position/kernel mismatch; consistent ground-truth
scores; no silent corruption; stable mixed text/image service at target load.

## Phase 7: benchmark method and reporting contract

- [ ] Reuse/adapt the DeepSeek repository's separation of engine measurements,
  production-shaped coding workloads, correctness checks, and near-context tests.
- [ ] Use exact token shapes, fixed seeds, streaming, and forced output lengths
  for engine tests. Use natural stopping and model-recommended sampling in a
  separate real-workload suite.
- [ ] Minimum controlled panel:
  - decode contexts 8K / 32K / 100K / 128K at C1/C2/C4;
  - cache-cold prefill 8K / 32K / 64K / 100K / 128K / 200K / near 262K;
  - outputs 1K / 4K / 16K, plus targeted 32K+ long coding outputs;
  - at least five fixed prompt/seed paths per published cell;
  - one warmup for every distinct shape, then measured runs in one unchanged
    process; use alternating independent process blocks for close finalists.
- [ ] Capture client metrics and server counters over the same steady-state
  interval. Reject intervals with wrong occupancy, queueing, prefill work,
  counter resets, request errors, or inadequate samples.
- [ ] Report median and every run, min/max, sample standard deviation/CV, and
  p50/p95/p99 latency where the sample supports it. Do not invent significance
  from same-process repetitions.
- [ ] Flush/bust caches for cold-prefill cells; preserve and report prefix-cache
  behavior in explicit hot-cache cells.
- [ ] Monitor GPU metrics at a fixed sampling rate. Keep compilation and model
  load time out of warm inference results, but publish cold startup separately.
- [x] Store raw responses privately when needed for grading; publish summaries
  without secrets, proprietary prompts, or sensitive source code.
- [x] Record failures and surprising results instead of deleting them.

## Phase 8: acceptance criteria and final artifacts

### Required gates

- [ ] TP1 can admit and complete at least one >100K-token coding request with a
  useful output reserve and the stated VRAM margin, or the final report clearly
  establishes why TP2 is required.
- [ ] The recommended profile survives C1/C2/C4 mixed-load and long-context soak
  tests with no OOM, restart, malformed output, silent concurrency clamp, or
  unacceptable decode starvation.
- [ ] Long-context retrieval and coding quality pass at 100K and 128K; native
  262K support is labeled qualified only after a near-limit quality test.
- [ ] Tool calls parse correctly, reasoning controls work, and outputs do not show
  drift/repetition/premature-EOS regressions against the base oracle.
- [ ] Vision is either qualified by the ground-truth suite or explicitly marked
  experimental/blocked with the relevant upstream issue and source revision.
- [ ] TP2 benefit is reported in absolute and percentage terms for latency,
  throughput, capacity, power, and VRAM; two TP1 replicas are included when both
  cards are dedicated to serving.

### Repository outputs

- [ ] Reproducible Containerfile/image composition and immutable source lock.
- [ ] Validated TP1 and TP2 launchers with minimal named profiles.
- [ ] Model/download instructions compatible with the existing `/data/models`
  Hugging Face cache and pinned snapshot revisions.
- [ ] Health/API/correctness/vision smoke tests.
- [ ] Capacity calculator for KV plus GDN-state sizing.
- [ ] Benchmark configs, runners, analyzers, raw machine-readable summaries, and
  regression tests for the analyzers.
- [ ] `README.md`, `RUN.md`, `BENCHMARKS.md`, and a concise recommendation table
  covering safe default, maximum-context, max-throughput, and vision profiles.
- [ ] A limitations section listing unqualified context lengths, concurrency,
  precisions, speculation modes, vision paths, and upstream blockers.

## Initial hypotheses to test (not conclusions)

1. BF16 weights should fit on one 96 GB card with useful 100K+ capacity, but
   graph memory, vision, the GDN state pool, KV precision, and speculative draft
   memory determine whether C2/C4 and a large output reserve also fit.
2. TP1 may win C1 decode because it avoids PCIe collectives; TP2 may improve
   long-prompt prefill and capacity. The break-even point must be measured.
3. If two GPUs are dedicated to C2/C4 independent requests, two TP1 replicas may
   beat TP2, unless long individual requests exceed TP1 capacity or prefix-cache
   duplication dominates.
4. FP8 KV and bfloat16 SSM are likely the largest memory levers. They cannot be
   recommended until long-context and recurrent-state correctness pass.
5. Speculation may improve useful output throughput, but current upstream drift
   reports make no-speculation the correctness oracle and likely safe fallback.

## Source and issue watchlist (checked 2026-08-23)

- SGLang Qwen3.8-27B cookbook:
  <https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-27B>
- Cookbook configuration source:
  <https://github.com/sgl-project/sglang/blob/1cf2b8c54d81802abc15dcf23a29b9cc687bc01e/docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx>
- BF16 model card: <https://huggingface.co/Qwen/Qwen3.8-27B>
- FP8 comparison checkpoint: <https://huggingface.co/Qwen/Qwen3.8-27B-FP8>
- NVFP4 comparison checkpoint:
  <https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4>
- Optional Unsloth GGUF variants:
  <https://huggingface.co/unsloth/Qwen3.8-27B-GGUF>
- SGLang model-loading formats:
  <https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/model_loading.mdx>
- llama.cpp multimodal requirements:
  <https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md>
- DSpark GDN-state divergence: <https://github.com/sgl-project/sglang/issues/35150>
- Multimodal mRoPE/fused-kernel report:
  <https://github.com/sgl-project/sglang/issues/35345>
- Chunked-prefill queue starvation report:
  <https://github.com/sgl-project/sglang/issues/35537>
- Qwen3.8 model issues, including long-context/premature-EOS reports:
  <https://github.com/QwenLM/Qwen3.8/issues>

Before locking a release, recheck all issue states and inspect the actual merged
fixes/tests. An issue being closed is not by itself proof that the selected image
contains the fix.
