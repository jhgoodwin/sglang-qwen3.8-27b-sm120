# Supply BASE_IMAGE only after verifying an immutable digest.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.title="sglang Qwen3.8-27B SM120 scaffold"
WORKDIR /workspace
# Dependencies and source pins are deliberately unresolved in Phase 1.
