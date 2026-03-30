from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from mlx_lm import load
from mlx_lm.tuner import evaluate
from mlx_lm.tuner.datasets import CacheDataset, CompletionsDataset


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_adapter_dir(adapter_path: Path) -> Path:
    if adapter_path.is_dir():
        return adapter_path
    raise ValueError("adapter-path must be an experiment adapter directory, not a single file.")


def evaluate_adapter(adapter_dir: Path, model_path: str, valid_file: Path, max_seq_length: int) -> dict:
    valid_records = load_jsonl(valid_file)
    model, tokenizer = load(model_path, adapter_path=str(adapter_dir), lazy=False)
    dataset = CacheDataset(
        CompletionsDataset(
            valid_records,
            tokenizer,
            prompt_key="prompt",
            completion_key="completion",
            mask_prompt=False,
        )
    )
    val_loss = float(
        evaluate(
            model,
            dataset,
            batch_size=1,
            num_batches=max(1, len(valid_records)),
            max_seq_length=max_seq_length,
        )
    )
    tokens = sum(min(len((item.get("text") or "").split()), max_seq_length) for item in valid_records)
    return {
        "val_loss": val_loss,
        "perplexity": math.exp(val_loss),
        "tokens_evaluated": tokens,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a real MLX LoRA adapter directory against the validation set.")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--valid-file", default="data/mlx_lm/valid.jsonl")
    parser.add_argument("--max-seq-length", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_dir = resolve_adapter_dir(Path(args.adapter_path))
    result = evaluate_adapter(adapter_dir, args.model_path, Path(args.valid_file), args.max_seq_length)
    metadata_path = adapter_dir / "metadata.json"
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        result["adapter_experiment_id"] = payload.get("experiment_id")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
