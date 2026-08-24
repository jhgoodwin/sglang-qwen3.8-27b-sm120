# Qwen3.8-27B SGLang SM120 service

Run `./serve.sh` for the qualified one-GPU C2 NVFP4+DFlash2 service on this
workstation. It uses the pinned local snapshots and immutable image recorded in
`source.lock.json`, native context 262,144, and the loopback endpoint
`127.0.0.1:11436`.

Named safe and experimental profiles remain available through `PROFILE`.
The default binds only `127.0.0.1:11436` to container port 8000; authenticate
before exposing another interface. See `RUN.md` for overrides and benchmark
profile instructions.
