# Qwen3.8-27B SGLang SM120 scaffold

This is the hardware-independent Phase 1 launch scaffold. It does not claim
that the model, image, SGLang revision, or GPU runtime has been qualified.
Replace unresolved identities in `release.json` and `source.lock.json` only
after verification.

Safe profiles disable speculation, use float32 recurrent state, FP8 KV,
FlashInfer, and 2,048-token chunked prefill. Production names are aliases
pending measurement. The default binds only `127.0.0.1:11436` to container
port 8000; authenticate before exposing another interface.
