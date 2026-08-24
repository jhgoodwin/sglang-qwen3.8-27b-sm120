# Qwen3.8-27B SGLang SM120 service

Run `./serve.sh` for the qualified one-GPU, concurrency-2 NVFP4+DFlash2 service
on this workstation. It uses the pinned local snapshots and immutable image
recorded in `source.lock.json`, native context 262,144, and the loopback
endpoint `127.0.0.1:11436`.

This work started from the
[SGLang Qwen3.8-27B cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)
and extends it with the pinned host-compatible runtime, qualification harness,
and benchmark results in this repository.

Named safe and experimental profiles remain available through `PROFILE`.
The default binds only `127.0.0.1:11436` to container port 8000; authenticate
before exposing another interface. See `RUN.md` for overrides and benchmark
profile instructions.

## Tested hardware and configuration

The service was developed on a workstation with two NVIDIA RTX PRO 6000
Blackwell Max-Q GPUs (SM120, 96 GB each). The qualified default uses one GPU
(GPU 0, tensor parallelism 1); the second GPU is not required. The host uses a
virtual CPU without AVX, so the pinned runtime overlay disables the unused
NIXL MoE import that otherwise aborts during startup.

The default runs the NVFP4 target with the DFlash2 draft model at the pinned
revisions in `source.lock.json`. It uses FlashInfer, 2,048-token chunked
prefill, FP8 KV cache, FP32 Mamba state, `extra_buffer_lazy`, a 0.85 static
memory fraction, and eight DFlash draft tokens. At most two requests run at
once, with eight Mamba-cache slots available. The configured context length is
262,144 tokens.

The locked runtime is SGLang `5f55db35`, PyTorch `2.13.0+cu130`, CUDA 13.0,
FlashInfer 0.6.17, and SGL Kernel 0.4.6.post1.

## Benchmark summary

| workload | measured result | scope |
|---|---:|---|
| Matched 8,192-input / 1,024-output decode | **261.063 output tok/s median** (259.867–263.355) | One active request; 15/15 completed with zero errors; 4.0499x the matched no-spec median |
| Three included coding prompts, three runs each | **176.916 aggregate end-to-end tok/s** | 131,408 completion tokens in 742.77 s; all 9 requests stopped normally |
| Concurrency 2, short input + 131,072 output | **134.238 aggregate tok/s median** | Two simultaneous requests; 3/3 accepted repetitions |
| Concurrency 2, 130,048 input + 131,072 output | **114.772 aggregate tok/s median** | Two simultaneous 261,120-total-token requests; 3/3 accepted repetitions |
| Four near-native arrivals at concurrency 2 | **128.906 aggregate tok/s median** | Two active requests at a time, drained in two waves; 3/3 accepted repetitions |

These results qualify the measured capacity and queue behavior of this exact
host and pinned recipe. They do not establish long-context answer quality,
vision correctness, concurrency 4, mixed-load latency, or soak stability. An
attempt to keep three near-native requests resident did not reach occupancy
three, which is why concurrency 2 is the default. See `BENCHMARK_RESULTS.md`
for full methods, min/median/max values, historical comparisons, failures, and
artifact references.
