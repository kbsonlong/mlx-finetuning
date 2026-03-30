from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import yaml


def check_module(name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        return True, version
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def summarize_model(config: dict) -> tuple[bool, list[str]]:
    model_path = Path(config["model"]["path"])
    messages = [f"model_path={model_path}"]
    if not model_path.exists():
        messages.append("missing model directory")
        return False, messages

    required = config["model"].get("required_files", [])
    missing = [name for name in required if not (model_path / name).exists()]
    if missing:
        messages.append("missing files=" + ",".join(missing))
        return False, messages

    has_weights = bool(list(model_path.glob("*.safetensors"))) or (model_path / "model.safetensors.index.json").exists()
    if not has_weights:
        messages.append("missing safetensors weights")
        return False, messages

    messages.append("model files look complete")
    return True, messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether the current Python environment is ready for real MLX training.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when the environment is not ready.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open("config/model_config.yaml", encoding="utf-8") as handle:
        model_config = yaml.safe_load(handle)

    mlx_ok, mlx_msg = check_module("mlx")
    mlx_lm_ok, mlx_lm_msg = check_module("mlx_lm")
    model_ok, model_msgs = summarize_model(model_config)

    ready = mlx_ok and mlx_lm_ok and model_ok
    print(f"mlx: {mlx_ok} ({mlx_msg})")
    print(f"mlx_lm: {mlx_lm_ok} ({mlx_lm_msg})")
    for msg in model_msgs:
        print(f"model: {msg}")
    print(f"ready: {str(ready).lower()}")

    if args.strict and not ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

