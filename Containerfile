ARG BASE_IMAGE=lmsysorg/sglang@sha256:616a3e97f45191af975896cfa644279096cb31bd408a071c2e99ca7209c3cafe
FROM ${BASE_IMAGE}

COPY patches/sglang/0001-noavx-disable-nixl-ep-import.patch /tmp/noavx.patch
RUN patch --batch --forward -d /sgl-workspace/sglang -p1 < /tmp/noavx.patch \
    && rm /tmp/noavx.patch

LABEL org.opencontainers.image.title="SGLang Qwen3.8-27B SM120 no-AVX overlay" \
      org.opencontainers.image.description="Official Qwen3.8 image with eager nixl_ep/UCX import disabled for this AVX-less host" \
      ai.sglang.base.digest="sha256:616a3e97f45191af975896cfa644279096cb31bd408a071c2e99ca7209c3cafe" \
      ai.sglang.noavx="nixl_ep import disabled; NIXL MoE dispatch unavailable"
