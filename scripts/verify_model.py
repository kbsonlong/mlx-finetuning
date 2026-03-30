from __future__ import annotations

from pathlib import Path

import yaml


def main() -> None:
    with open("config/model_config.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    model_path = Path(config["model"]["path"])
    required = config["model"].get("required_files", [])
    missing = [name for name in required if not (model_path / name).exists()]
    safetensors = list(model_path.glob("*.safetensors"))
    has_weights = bool(safetensors) or (model_path / "model.safetensors.index.json").exists()

    print(f"model_path: {model_path}")
    if not model_path.exists():
        raise SystemExit("Model path does not exist")
    if missing:
        raise SystemExit(f"Missing required files: {', '.join(missing)}")
    if not has_weights:
        raise SystemExit("No safetensors weights or weight index found")
    print("status: ok")


if __name__ == "__main__":
    main()
