# mlx-autoresearch

Autoresearch-style LoRA experiment scaffold for Apple Silicon.

## Quick start

```bash
python prepare.py --augment
python train.py --preset smoke_real
```

`train.py` now assumes the current Python environment already has `mlx` and `mlx_lm`, and that `config/model_config.yaml` points at a valid local model.

Environment setup details are documented in [docs/08-environment-setup.md](./docs/08-environment-setup.md).

If you want `prepare.py` to call DeepSeek in OpenAI-compatible mode, configure `base_url` and `api_key` first, either via flags or environment variables:

```bash
export DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
export DEEPSEEK_API_KEY="..."
python prepare.py --augment
```

Without `base_url` and `api_key`, `prepare.py` falls back to the built-in template generator.

Default seed questions live in [data/seed_questions.txt](./data/seed_questions.txt), so you can edit them directly before running `prepare.py`.

```bash
python train.py --preset baseline
```

The current repository implements the phase-one MVP from `docs/`:

- frozen validation/test split generation
- repeatable metadata and file hashing
- parse-friendly training metrics output
- real MLX training and evaluation flow
