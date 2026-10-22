# Kaleido CPU delivery

Requires Python 3.12 and uv 0.12.23. Validated on macOS arm64 and Linux x86_64.
No source checkout, Qwen weights or training data are needed.

    python3.12 install.py --verify-only
    python3.12 install.py --env /absolute/path/to/new-venv

The first install downloads the exact runtime artifacts in pylock.cpu.toml.
With a prepared uv wheel cache, add --offline --cache-dir /absolute/path/to/cache.
Dependency wheels are not included.
The installer refuses an existing environment and preserves failed environments for diagnosis.

    /absolute/path/to/new-venv/bin/kaleido predict --bundle /absolute/path/to/decoder-42

Input one JSON line, e.g. {"prompt":"Alice was","max_new_tokens":12}.
CPU, batch 1, greedy cropped-window decoding, PyTorch 2.14.1; no HTTP server or KV cache export.
Checksums ensure consistency with this manifest, not an independent signature.
