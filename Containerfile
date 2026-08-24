ARG BASE_IMAGE=lmsysorg/sglang@sha256:616a3e97f45191af975896cfa644279096cb31bd408a071c2e99ca7209c3cafe
FROM ${BASE_IMAGE}
ARG BASE_HAS_NOAVX=0

COPY patches/sglang/0001-noavx-disable-nixl-ep-import.patch /tmp/noavx.patch
RUN if [ "${BASE_HAS_NOAVX}" = "1" ]; then \
      grep -q '^use_nixl = False$' /sgl-workspace/sglang/python/sglang/srt/layers/moe/token_dispatcher/nixl.py; \
    else \
      patch --batch --forward -d /sgl-workspace/sglang -p1 < /tmp/noavx.patch; \
    fi \
    && rm /tmp/noavx.patch

COPY patches/sglang/0002-c2c3-server-evidence.patch /tmp/c2c3-evidence.patch
RUN patch --batch --forward -d /sgl-workspace/sglang -p1 < /tmp/c2c3-evidence.patch \
    && rm /tmp/c2c3-evidence.patch

LABEL org.opencontainers.image.title="SGLang Qwen3.8-27B SM120 no-AVX overlay" \
      org.opencontainers.image.description="Official Qwen3.8 image with the no-AVX guard and opt-in C2/C3 evidence wiring" \
      ai.sglang.base.digest="sha256:616a3e97f45191af975896cfa644279096cb31bd408a071c2e99ca7209c3cafe" \
      ai.sglang.noavx="nixl_ep import disabled; NIXL MoE dispatch unavailable" \
      ai.sglang.c2c3-evidence="opt in with SGLANG_C2C3_EVIDENCE_PATH"
