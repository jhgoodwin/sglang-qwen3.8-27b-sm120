# Benchmark results

These are matched exploratory C1 decode runs: `random-ids`, 8,192 input
tokens, 1,024 forced output tokens, five measured requests plus one warmup,
seed 101, streaming, and `max_running_requests=1`. The server used the pinned
host-compatible SGLang overlay, FlashInfer, FP8 KV, float32 SSM, and a
131,072-token configured context. The benchmark JSON files are intentionally
kept in the ignored `bench/results/` tree; the paths below are the raw
evidence.

| configuration | GPUs | input tok/s | output tok/s | mean TTFT | mean TPOT / ITL | p99 ITL | max ITL | accept length | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TP1 base | GPU 0 | 200.2759 | 25.0345 | 1,748.69 ms | 38.2665 ms | 39.2322 ms | 75.6475 ms | — | operational no-spec oracle |
| TP2 base | GPUs 0,1 | 367.7918 | 45.9740 | 1,328.41 ms | 20.4694 ms | 21.1335 ms | 40.4393 ms | — | operational no-spec comparison |
| TP1 EAGLE/MTP | GPU 0 | 594.6573 | 74.3322 | 1,828.22 ms | 11.6781 ms | 21.0195 ms | 46.3710 ms | 3.8684 | one-GPU operational candidate |
| TP1 DSpark | GPU 0 | — | 89.1607 | 1,795.83 ms | 9.4634 ms | 43.1050 ms | 44.3600 ms | 4.49224 | one-GPU exploratory candidate |
| TP2 EAGLE/MTP | GPUs 0,1 | 984.3189 | 123.0399 | 1,360.97 ms | 6.8040 ms | 12.1222 ms | 48.2276 ms | 3.8841 | absolute speed leader; exploratory |
| TP1 BF16+DFlash2 | GPU 0 | 1,043.19 | **130.40** | 1,798.77 ms | 5.91 ms | 14.75 ms | 46.04 ms | 7.2444 | one-GPU raw leader; candidate only |
| TP1 NVFP4+DSpark | GPU 0 | — | 117.6417 | 874.65 ms | 7.6499 ms | 21.0525 ms | 36.9900 ms | 2.66458 | one-GPU candidate; exploratory |

All seven matched configurations completed five requests and exactly 5,120 output tokens with no
reported request errors. The TP1 NVFP4+DSpark run lasted 43.522 s, completed
5/5 requests, and had a separate sampled peak of 390 output tokens/s. Relative
output-throughput speedups are:

- TP2 base vs TP1 base: **1.836x** (+83.6%).
- TP1 EAGLE vs TP1 base: **2.969x** (+196.9%).
- TP1 DSpark vs TP1 base: **3.5615x** (+256.1514%).
- TP1 DSpark vs TP1 EAGLE: **1.1995x** (+19.9490%).
- TP1 DSpark vs TP2 EAGLE: **0.7246x** (-27.5351%).
- TP1 BF16+DFlash2 vs TP1 base: **5.208814x** (+420.8814%).
- TP1 BF16+DFlash2 vs TP1 EAGLE: **1.754288x** (+75.4288%).
- TP1 BF16+DFlash2 vs TP1 DSpark: **1.462528x** (+46.2528%).
- TP1 BF16+DFlash2 vs TP2 EAGLE: **1.059819x** (+5.9819%).
- TP1 NVFP4+DSpark vs TP2 EAGLE: **0.9553x** (-4.4674%).
- TP1 NVFP4+DSpark vs TP1 BF16+DFlash2: **0.9025x** (-9.7833%).
- TP2 EAGLE vs TP2 base: **2.676x** (+167.6%).
- TP2 EAGLE vs TP1 EAGLE: **1.655x** (+65.5%).
- TP2 EAGLE vs TP1 base: **4.916x** (+391.6%).

The historical best in this original panel was **130.399 sustained output
tokens/s on TP1 BF16+DFlash2** (one-GPU leader at that time). The current
cookbook rerun below supersedes that historical rank for the tested recipe;
the original rows remain immutable for comparison.

TP1 DSpark used the BF16 target checkpoint with BF16 draft revision
`85ef153be924f17ce4bf62726954eeaa4a73e854`, block size 7 and verify window 8,
FP32 SSM, and 12.25 GB free after graph capture. Its reported peak output
throughput was 182 tok/s; this is a peak sample, not sustained throughput.
The upstream state-drift warning remains applicable to this DSpark result.

TP1 BF16+DFlash2 used the BF16 target checkpoint with draft revision
`dedf8df68adfb1afeaf7b7480c0a0243108177b4`, block size 8, FP32 SSM, and
16.96 GB free after graph capture. The run lasted 39.264 s, completed 5/5
requests with zero errors and 1,024 output tokens each. Its separate peak
output throughput was 183 tok/s, not sustained throughput.
The fixed smoke semantic paths were identical. Random output matched TP1 base
and TP1 EAGLE on 4/5 requests and TP1 DSpark on 5/5 requests. This remains a
candidate measurement, not a production qualification or correctness
conclusion.

Raw artifacts:

- [TP1 base](bench/results/20260823T1342Z-tp1-bf16-safe-decode/tp1-c1-8k-1024-seed101.json)
- [TP2 base](bench/results/20260823T1352Z-tp2-bf16-safe-decode/tp2-c1-8k-1024-seed101.json)
- [TP2 EAGLE/MTP](bench/results/20260823T1357Z-tp2-mtp-decode/tp2-mtp-c1-8k-1024-seed101.json)
- [TP1 EAGLE/MTP](bench/results/20260823T1401Z-tp1-mtp-decode/tp1-mtp-c1-8k-1024-seed101.json)
- [TP1 DSpark](bench/results/20260823T1413Z-tp1-dspark-decode/tp1-dspark-c1-8k-1024-seed101.json)
- [TP1 BF16+DFlash2](bench/results/20260823T1422Z-tp1-dflash-decode/tp1-dflash-c1-8k-1024-seed101.json)
- [TP1 BF16+DFlash2 telemetry](bench/results/20260823T1422Z-tp1-dflash-decode/tp1-dflash-c1-8k-1024-seed101-telemetry.json)
- [TP1 NVFP4+DSpark](bench/results/20260823T145116Z-tp1-nvfp4-dspark-decode.json)

## Operational coding prompts

The three unchanged prompts in `bench/prompts/` were each run
three times on TP1 BF16+DFlash2, TP1 BF16+DSpark, and TP1 NVFP4+DSpark
(nine requests per configuration). Requests used `reasoning_effort=medium`, `max_tokens=32768`,
streaming with usage included, and omitted temperature, top-p, and top-k so
the server/model sampling defaults applied. All requests naturally stopped;
none was forced to a token length. Completion tokens include reasoning tokens;
visible tokens are `completion_tokens - reasoning_tokens`. E2E throughput is
completion tokens divided by wall-clock completion time, including first-token
latency.

For each cell below, values are min / median / max across the three
repetitions. Times are seconds, token counts are tokens, and throughput is
completion tokens/s.

| prompt | BF16+DFlash2 completion time | completion tokens | reasoning tokens | visible tokens | e2e throughput |
|---|---:|---:|---:|---:|---:|
| django-varbit | 99.751 / 151.821 / 152.740 | 9,488 / 14,012 / 14,965 | 7,287 / 11,925 / 12,040 | 2,087 / 2,201 / 2,925 | 92.293 / 95.117 / 97.977 |
| flappy-bird | 37.077 / 47.497 / 47.789 | 4,085 / 5,635 / 5,714 | 15 / 23 / 30 | 4,062 / 5,620 / 5,684 | 110.176 / 118.638 / 119.568 |
| slack-clone | 220.407 / 291.865 / 343.134 | 17,708 / 22,835 / 25,344 | 5,368 / 11,877 / 14,853 | 10,491 / 10,958 / 12,340 | 73.860 / 78.238 / 80.342 |

| prompt | NVFP4+DSpark completion time | completion tokens | reasoning tokens | visible tokens | e2e throughput |
|---|---:|---:|---:|---:|---:|
| django-varbit | 65.605 / 101.066 / 131.756 | 10,285 / 16,238 / 22,192 | 6,722 / 12,215 / 20,388 | 1,804 / 3,563 / 4,023 | 156.772 / 160.668 / 168.432 |
| flappy-bird | 24.882 / 28.441 / 29.110 | 4,435 / 5,002 / 5,010 | 16 / 26 / 43 | 4,419 / 4,967 / 4,976 | 172.105 / 175.875 / 178.245 |
| slack-clone | 95.429 / 166.536 / 190.485 | 12,888 / 19,586 / 22,816 | 2,585 / 10,300 / 12,352 | 9,286 / 10,303 / 10,464 | 117.608 / 119.778 / 135.054 |

| prompt | BF16+DSpark completion time | completion tokens | reasoning tokens | visible tokens | e2e throughput |
|---|---:|---:|---:|---:|---:|
| django-varbit | 132.031 / 203.842 / 221.886 | 10,368 / 15,133 / 17,214 | 6,927 / 12,270 / 15,040 | 2,174 / 2,863 / 3,441 | 74.239 / 77.580 / 78.527 |
| flappy-bird | 50.638 / 57.755 / 65.442 | 4,290 / 4,997 / 5,829 | 18 / 26 / 33 | 4,257 / 4,979 / 5,803 | 84.719 / 86.521 / 89.071 |
| slack-clone | 321.438 / 325.047 / 492.948 | 18,692 / 20,483 / 29,521 | 7,283 / 9,240 / 19,919 | 9,602 / 11,243 / 11,409 | 58.151 / 59.887 / 63.016 |

| configuration | requests | total completion tokens | total reasoning tokens | total visible tokens | total time | aggregate e2e tok/s |
|---|---:|---:|---:|---:|---:|---:|
| BF16+DFlash2 | 9 | 119,786 | 63,418 | 56,368 | 1,392.081 s | 86.048 |
| BF16+DSpark | 9 | 126,527 | 70,756 | 55,771 | 1,871.027 s | 67.624 |
| NVFP4+DSpark | 9 | 118,452 | 64,647 | 53,805 | 833.309 s | 142.147 |

Across this operational sample, NVFP4+DSpark completed 1.651941x as many
completion tokens per wall-clock second in aggregate, with 1.113653% fewer total
completion tokens and 1.9% more reasoning tokens than BF16+DFlash2. The
per-prompt distributions show that token-count variation is material: for
example, slack-clone completion ranged from 12,888 to 22,816 tokens on
NVFP4+DSpark and from 17,708 to 25,344 on BF16+DFlash2. This is a task-time
comparison, not a quantization-only causal test: the configurations differ in
both target precision (BF16 versus NVFP4 mixed precision) and speculative
draft/runtime (DFlash2 versus DSpark).

BF16+DSpark versus NVFP4+DSpark is the quantization-controlled comparison:
both use the same DSpark draft snapshot and block-7/8-token settings. Across
these three stochastic repetitions per prompt, NVFP4+DSpark used 55.462474%
less wall time (equivalently, BF16+DSpark used 124.529760% more), 6.382037%
more completion tokens, 8.633897% more reasoning
tokens, and 3.525130% more visible tokens; NVFP4+DSpark's aggregate
throughput was 2.102002x higher. Median completion-time reductions for
NVFP4+DSpark were 50.419669% (django-varbit), 50.755923% (flappy-bird), and
48.765445% (slack-clone). This isolates the target-precision difference in
the runtime setup, but remains a three-sample timing comparison and does not
establish output quality equivalence.

The operational completion-validity contract is HTTP 200, no request error,
and `finish_reason=stop`. A `length` finish is truncated/incomplete and is
excluded or explicitly flagged, never counted as success. All 27 requests in
these nine raw files satisfied the contract; the maximum observed completion
was 29,521 tokens, below the 32,768-token cap. If a future run reaches
`length`, preserve that outcome and rerun with a context-safe cap near the
128K model ceiling after subtracting prompt and chat-template tokens. This is
distinct from the matched random-ID decode contract, which intentionally
forces a fixed output length.

Raw operational artifacts: [BF16+DFlash2 rep1](bench/results/20260823T000000Z-tp1-dflash-operational-prompts.json),
[rep2](bench/results/20260823T000100Z-tp1-dflash-operational-prompts-rep2.json),
[rep3](bench/results/20260823T000200Z-tp1-dflash-operational-prompts-rep3.json),
[NVFP4+DSpark rep1](bench/results/20260823-tp1-nvfp4-dspark-operational-prompts.json),
[rep2](bench/results/20260823-tp1-nvfp4-dspark-operational-prompts-rep2.json), and
[rep3](bench/results/20260823-tp1-nvfp4-dspark-operational-prompts-rep3.json),
[BF16+DSpark rep1](bench/results/20260823-tp1-bf16-dspark-operational-prompts-rep1.json),
[rep2](bench/results/20260823-tp1-bf16-dspark-operational-prompts-rep2.json), and
[rep3](bench/results/20260823-tp1-bf16-dspark-operational-prompts-rep3.json).

## Current cookbook NVFP4+DFlash2 rerun (2026-08-23)

This is a separately versioned rerun of the generated current cookbook recipe;
historical rows above are preserved and are not overwritten. The documentation
recipe was main revision `d1af3c89233c475fc1bf11939d86787e6cddd58c` (the source
install pin `1cf2b8c54d81802abc15dcf23a29b9cc687bc01e` does not contain the
DFlash path). The official base image digest was
`sha256:616a3e97f45191af975896cfa644279096cb31bd408a071c2e99ca7209c3cafe`;
the host-compatible overlay was `sha256:d3346cea82545d982b7ec169f1f0f6f47834b0c4a70ec693e954a8d66111cb8d`,
with SGLang source `5f55db35e926d50676f75b812640ea2410b0fe0e` as recorded in
the runtime lock. The target was NVFP4 revision
`319f741cce68d7914884900c138a1fbb70a42f30`; the DFlash2 draft was revision
`dedf8df68adfb1afeaf7b7480c0a0243108177b4`. The no-spec profile used no
draft. All tests used TP1 on GPU 0, context 131,072,
explicit `--kv-cache-dtype fp8_e4m3`, and one running request. The observed
server sampling defaults were temperature 1.0, top-k 20, and top-p 0.95;
the operational suite used `reasoning_effort=medium` and `max_tokens=32768`.

The matched panel used exact 8,192 input and 1,024 output tokens, seed 101,
streaming, C1, and one warmup plus five measured requests per repetition.
Each profile below has three independent repetitions (45 measured requests
total across the three profiles), 5,120 output tokens per repetition, and
zero errors.

| profile | output tok/s min / median / max | accept length | result |
|---|---:|---:|---|
| current no-spec | 64.174 / 64.461 / 65.261 | — | completed 15/15 |
| current DFlash2 default | 252.087 / 253.074 / 255.697 | 7.1735 | completed 15/15 |
| current DFlash2 `extra_buffer_lazy` | **259.867 / 261.063 / 263.355** | 7.5265 | completed 15/15 |

The lazy median is 4.0499x current no-spec, 1.0316x current default, and
2.0020x the historical one-GPU leader (130.3991 tok/s). These are decode
throughput comparisons only; they do not establish output quality or causal
effects from quantization.

The lazy profile was then run against the three coding prompts three times
each. All nine requests returned HTTP 200 with `finish_reason=stop`; no
128K-cap rerun was needed because no request hit the 32,768-token limit.
Values are min / median / max across repetitions (times in seconds).

| prompt | completion tokens | reasoning tokens | visible tokens | wall time | end-to-end tok/s |
|---|---:|---:|---:|---:|---:|
| django-varbit | 10,959 / 17,730 / 18,694 | 9,449 / 14,819 / 17,111 | 1,510 / 1,583 / 2,911 | 57.339 / 89.338 / 93.871 | 191.126 / 198.459 / 199.145 |
| flappy-bird | 4,146 / 4,526 / 5,171 | 17 / 19 / 30 | 4,116 / 4,509 / 5,152 | 17.992 / 18.251 / 21.343 | 230.431 / 242.279 / 247.988 |
| slack-clone | 18,493 / 23,281 / 28,408 | 8,547 / 13,103 / 18,619 | 9,789 / 9,946 / 10,178 | 114.194 / 141.744 / 188.698 | 150.547 / 161.944 / 164.247 |

Across the nine lazy requests: 131,408 completion tokens (81,714 reasoning,
49,694 visible) in 742.770906 seconds, or 176.915923 aggregate end-to-end
tokens/s. Compared arithmetically with the historical NVFP4+DSpark suite
(142.147 tok/s; 118,452 completion, 64,647 reasoning, 53,805 visible,
833.309 s), lazy is 1.244602x faster and takes 0.891179x the wall time,
while generating 1.109378x as many completion tokens, 1.264003x as many
reasoning tokens, and 0.923594x as many visible tokens. Compared with the
historical BF16+DFlash2 suite (86.048 tok/s; 119,786 completion, 63,418
reasoning, 56,368 visible, 1,392.081 s), lazy is 2.056011x faster and takes
0.533569x the wall time, while generating 1.097023x as many completion
tokens, 1.288499x as many reasoning tokens, and 0.881599x as many visible
tokens. These arithmetic comparisons are task-time/throughput observations,
not task-quality or quantization conclusions.

Raw current-panel artifacts: [benchmark command](bench/results/current-cookbook-20260823/benchmark-command.txt),
[all matched-panel artifacts](bench/results/current-cookbook-20260823/), and
[all lazy operational repetitions](bench/results/current-cookbook-20260823/operational/).

## Long-context measurements

These are separate TP1 EAGLE/MTP C1 measurements at 100,000 input tokens;
they are not part of the matched 8K/1K table above.

| cache / shape | measured | duration | client output tok/s | TTFT | mean TPOT | p99 ITL | max ITL | accept length | result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cold, 100K in / 256 out | 1 request | 45.1438 s | 5.6708 | 41,681.88 ms | 13.5617 ms (~73.74 tok/s steady reciprocal) | 35.0847 ms | 50.4215 ms | 3.675 | completed; no OOM/truncation |
| hot, 100K in / 1,024 out | 1 request | — | 76.64 | 294.87 ms | 12.77 ms | 22.88 ms | 94.65 ms | 3.88 | successful; not cold-publishable |

The cold run was confirmed by the server log with `#cached-token 0`. Its
configured context was 131,072 and `max_total_num_tokens` was 550,706.
The 5.6708 tok/s client aggregate includes the long prefill; the reciprocal
of steady-state mean TPOT is approximately 73.74 tok/s.

## Qualification caveats

These numbers are not a production qualification. GPU telemetry was sampled,
but these are samples rather than full-run aggregates: TP1 MTP showed GPU0 at
299–300 W, 99% SM, approximately 80–81 C, and 77,742 MB framebuffer use, with
GPU1 at 2 MB and approximately 7 W; TP2 base showed approximately 299–300 W
each, 99% SM, and 80,814/80,094 MB; TP2 MTP showed approximately 300 W each,
97–98% SM, and 81,688/80,968 MB. The 100K+ capacity envelope beyond the
reported shape, C2/C4 behavior, soak stability, mixed prefill/decode
starvation, coding corpus, and native-context gates remain unresolved.

EAGLE/MTP acceptance length is reported by SGLang (3.8684 TP1 and 3.8841
TP2); DSpark reports 4.49224 and the no-speculation rows have no
acceptance-length metric. The DSpark run completed 5/5 requests, produced
5,120 output tokens in 57.4244 s, and reported five empty error strings. Raw
random-ID generated text is not a semantic quality test: TP1 DSpark matched
TP1 base and TP1 EAGLE on 4/5 random outputs. The fixed smoke semantic outputs
(text, streaming, structured tool call, reasoning control, and image request)
matched. This is evidence of operational consistency, not a declaration of
production correctness or model-quality equivalence.

The DSpark telemetry available for that run consists of idle samples only and
must not be interpreted as load telemetry. DFlash2 telemetry sampled GPU0 at
299–300 W, 64–69 C, 98–100% SM, 27–89% memory utilization, and clocks of
1567–1912 MHz; GPU1 was idle. These are samples, not full-run aggregates.

The first TP2 launch attempt failed during startup in the custom all-reduce
path. The successful TP2 runs required the NCCL fallback; this is part of the
runtime condition for the TP2 measurements and should remain explicit in any
reproduction.

## C2/C3 native-context campaign (2026-08-24)

This is the measured follow-up to the queued C2/C3 campaign. It retains the
TP1 NVFP4+DFlash2 `extra_buffer_lazy` recipe, FP8 KV, FP32 SSM, FlashInfer,
2,048-token chunked prefill, DFlash8, and 0.85 static-memory fraction. The
server context was 262,144 and the request output ceiling was 131,072. C2 used
`--max-running-requests 2 --max-mamba-cache-size 8`; C3 used 3 and 12. Neither
profile used `--mamba-full-memory-ratio`. The evidence image was
`qwen38-c2c3-evidence@sha256:c06fcb906923c13579ff0a1bd01bc8c728e2fef9e6adc549fb0677a7d21dfddb`
at SGLang revision `5f55db35e926d50676f75b812640ea2410b0fe0e`.

| profile | resolved token pool | FP8 KV pool | FP32 Mamba pool | DFlash intermediate | CUDA graphs |
|---|---:|---:|---:|---:|---:|
| C2 | 1,289,769 | 42,263,183,360 B | 1,385,496,576 B / 8 slots | 3,653,369,856 B / 8 states | batches 1,2; 1,814,036,480 B |
| C3 | 1,246,816 | 40,855,699,456 B | 2,001,272,832 B / 12 slots | 4,871,159,808 B / 8 states | batches 1,2,3; 1,866,465,280 B |

All three cold C2 and C3 admission probes passed. C2 A1 was accepted through
the preserved corrected reimport after the original importer rejected its
reasoning-token accounting; the raw result was not overwritten. C2 admitted
two short requests and queued the excess arrival; C3 admitted three short
requests and queued the fourth. Short admission therefore did not establish
the near-native occupancy result.

The nominal B shape used an exact server-side prompt count of 261,120 and
requested 1,024 forced output tokens. The server terminated each stream at
1,022 output tokens with `finish_reason=length`, leaving a two-token internal
boundary reserve. C2 nevertheless reached two simultaneous near-native
requests, observed 11.2037% minimum free VRAM, and produced 2,044 aggregate
completion tokens in 245.283 s. This outcome is preserved as an exact-B
failure; an explicit dependency-only waiver allowed later, boundary-safe
cells to run. It did not relabel B as success or relax the later 131,072-token
exact-output checks.

C3's B attempt admitted three exact 261,120-token prompts but observed at most
two running and two queued requests, not the required occupancy of three.
Each returned 1,022 tokens with length termination; aggregate wall time was
378.130 s, aggregate completion throughput was 8.108 tok/s, and minimum free
VRAM was 11.1343%. Because the failure included the exact-occupancy gate, the
two-token dependency waiver was not applicable. The fail-fast campaign did
not spend GPU time on C3 C/D/E cells.

C2 completed three measured repetitions of every later cell. Values below are
min / median / max aggregate completion tokens/s. Every request returned the
exact forced output count with `finish_reason=length`, and no run had an OOM,
restart, disconnect, or HTTP error.

| cell | request shape and arrivals | accepted | aggregate completion tok/s | minimum free VRAM |
|---|---|---:|---:|---:|
| C | 2 simultaneous: 1,024 + 131,072 | 3/3 | 126.246 / 134.238 / 138.693 | 11.3733% |
| D | 2 simultaneous: 130,048 + 131,072 | 3/3 | 100.480 / 114.772 / 174.512 | 11.1813% |
| E | 4 simultaneous D-shape arrivals | 3/3 | 123.674 / 128.906 / 143.445 | 11.1731% |

Each E repetition produced 524,288 completion tokens, observed maximum
occupancy 2, maximum queue depth 3, and exactly two completion-driven waves.
Aggregate wall times were 4,239.288, 4,067.225, and 3,654.972 seconds.
Queued-request TTFT depended on which speculative-decoding stream freed a
slot: observed queued TTFTs ranged from 1,082.222 to 2,252.942 seconds. This
wide tail and the D throughput range reflect strong generated-content
sensitivity in DFlash acceptance, not an admission or memory failure.

For the tested full-context service shape, C2 is the measured choice. It
sustained two boundary-safe 261,120-total-token requests and correctly drained
four simultaneous arrivals in two waves. C3 consumed more recurrent/graph
memory, exposed a smaller KV/token pool, and did not realize a third
near-native resident. These measurements qualify the stated capacity and
queue behavior only; they do not establish long-context answer quality,
vision correctness, C4 operation, mixed-prefill starvation, or soak stability.

Preserved machine-readable artifacts are under
`bench/results/c2-c3-native-20260824/measured/`, including accepted imports,
failed B outcomes, cold-boot provenance, scheduler JSONL, GPU/process samples,
and server evidence. The results directory remains excluded from ordinary Git
publication; the summary above is the tracked record.
