# Troubleshooting (Apple Silicon)

## `mlx-vlm` import fails

- Ensure Python 3.11+: `python --version`
- Reinstall inside the project venv:
  ```
  source .venv/bin/activate
  pip install --upgrade "mlx>=0.11" "mlx-vlm>=0.0.10"
  ```
- If pip fails at build time, check that Xcode command-line tools are installed:
  `xcode-select --install`

## Model download stalls

MLX-VLM pulls from Hugging Face on first use. If the download stalls:
- Set a mirror or resume manually with:
  `huggingface-cli download mlx-community/Qwen2-VL-2B-Instruct-4bit`
- Check `~/.cache/huggingface/hub/` has enough free space (2 GB+ for 2B/4-bit models).

## RAM pressure / kernel_task spikes

- Drop to a smaller/quantised model (`Qwen2-VL-2B-Instruct-4bit` vs `-7B`).
- Set `inference.workers: 1` in config.yaml.
- Set `inference.batch_size: 16`.
- Close other memory-hungry apps.

## `ollama serve` isn't detected

- `ollama serve` runs on `http://127.0.0.1:11434` by default.
  Override via `DERMLAT_OLLAMA_HOST=http://host:port`.
- Confirm the vision model is pulled: `ollama pull qwen2-vl:7b` (or your model of choice).

## SQLite is locked

- The pipeline uses WAL mode and single-writer semantics. If a stale process is holding
  the DB open, kill it:
  ```
  lsof -f -- /path/to/output/audit.sqlite
  kill -9 <pid>
  ```
- Reruns are always idempotent, so it's safe to relaunch after killing.

## "unsupported extension" for files that look like images

- Rename to a supported extension (see `constants.SUPPORTED_EXTENSIONS`) or open a PR
  adding the extension.

## The report shows lots of `NOT_VERIFIABLE`

- You haven't loaded supplied annotations (`--annotations`), so agreement-based claims
  can't be checked. Verify column names against `docs/ANNOTATION_GUIDELINES.md`.

## Everything works but is slow

- Confirm you're using MLX-VLM, not Ollama, on Apple Silicon.
- Check Activity Monitor's GPU tab; the model should be pinned to the GPU.
- Larger models are dramatically slower — start with the 2B/4-bit build.

## macOS asks for camera / microphone / files permission

- The pipeline reads image files only. If macOS is prompting for camera or microphone,
  a browser extension or another running process is the culprit — not this pipeline.
