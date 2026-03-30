from __future__ import annotations

# Config block: the intended AI-editable surface.
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LEARNING_RATE = 5e-5
BATCH_SIZE = 4
ITERS = 100
MAX_SEQ_LENGTH = 512
OPTIMIZER = "adam"
GRAD_CHECKPOINT = True
LR_SCHEDULE = "constant"

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import yaml
from mlx_lm import load
from mlx_lm.tuner import TrainingArgs, evaluate, linear_to_lora_layers, train
from mlx_lm.tuner.datasets import CacheDataset, CompletionsDataset

from scripts.data_guard import verify_frozen_integrity


PRESET_OVERRIDES = {
    "baseline": {
        "LORA_RANK": 8,
        "LORA_ALPHA": 16,
        "LORA_DROPOUT": 0.05,
        "LEARNING_RATE": 5e-5,
        "BATCH_SIZE": 4,
        "ITERS": 100,
        "MAX_SEQ_LENGTH": 512,
    },
    "fast_test": {
        "LORA_RANK": 4,
        "LORA_ALPHA": 8,
        "LORA_DROPOUT": 0.05,
        "LEARNING_RATE": 5e-5,
        "BATCH_SIZE": 8,
        "ITERS": 50,
        "MAX_SEQ_LENGTH": 256,
    },
    "smoke_real": {
        "LORA_RANK": 4,
        "LORA_ALPHA": 8,
        "LORA_DROPOUT": 0.0,
        "LEARNING_RATE": 5e-5,
        "BATCH_SIZE": 1,
        "ITERS": 1,
        "MAX_SEQ_LENGTH": 128,
    },
    "large_rank": {
        "LORA_RANK": 16,
        "LORA_ALPHA": 32,
        "LORA_DROPOUT": 0.05,
        "LEARNING_RATE": 5e-5,
        "BATCH_SIZE": 4,
        "ITERS": 100,
        "MAX_SEQ_LENGTH": 512,
    },
    "long_seq": {
        "LORA_RANK": 8,
        "LORA_ALPHA": 16,
        "LORA_DROPOUT": 0.05,
        "LEARNING_RATE": 5e-5,
        "BATCH_SIZE": 4,
        "ITERS": 100,
        "MAX_SEQ_LENGTH": 1024,
    },
    "deep_train": {
        "LORA_RANK": 8,
        "LORA_ALPHA": 16,
        "LORA_DROPOUT": 0.05,
        "LEARNING_RATE": 5e-5,
        "BATCH_SIZE": 4,
        "ITERS": 200,
        "MAX_SEQ_LENGTH": 512,
    },
    "small_batch": {
        "LORA_RANK": 8,
        "LORA_ALPHA": 16,
        "LORA_DROPOUT": 0.05,
        "LEARNING_RATE": 5e-5,
        "BATCH_SIZE": 2,
        "ITERS": 100,
        "MAX_SEQ_LENGTH": 512,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real MLX LoRA training experiment.")
    parser.add_argument("--preset", choices=sorted(PRESET_OVERRIDES), help="Apply one of the built-in training presets.")
    return parser.parse_args()


def apply_preset_overrides(preset: str | None) -> str | None:
    if not preset:
        return None
    globals().update(PRESET_OVERRIDES[preset])
    return preset


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compute_experiment_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"exp_{stamp}"


def build_lora_config() -> dict:
    return {
        "rank": LORA_RANK,
        "alpha": LORA_ALPHA,
        "dropout": LORA_DROPOUT,
        "scale": 10.0,
    }


def ensure_results_header(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "experiment_id\tdataset_id\tval_loss\tval_accuracy\tperplexity\ttrain_time_seconds\tmode\tstatus\tdescription\n",
        encoding="utf-8",
    )


def append_result(path: Path, row: dict) -> None:
    ensure_results_header(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            "{experiment_id}\t{dataset_id}\t{val_loss:.6f}\t{val_accuracy:.6f}\t{perplexity:.6f}\t"
            "{train_time_seconds:.2f}\t{mode}\t{status}\t{description}\n".format(**row)
        )


def save_adapter_metadata(output_dir: Path, payload: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_dir / payload["experiment_id"]
    adapter_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = adapter_dir / "metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata_path


def ensure_adapter_config(adapter_dir: Path) -> None:
    config = {
        "fine_tune_type": "lora",
        "num_layers": 16,
        "lora_parameters": build_lora_config(),
    }
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def create_datasets(train_records: list[dict], valid_records: list[dict], tokenizer):
    train_dataset = CacheDataset(
        CompletionsDataset(
            train_records,
            tokenizer,
            prompt_key="prompt",
            completion_key="completion",
            mask_prompt=False,
        )
    )
    valid_dataset = CacheDataset(
        CompletionsDataset(
            valid_records,
            tokenizer,
            prompt_key="prompt",
            completion_key="completion",
            mask_prompt=False,
        )
    )
    return train_dataset, valid_dataset


def choose_optimizer():
    if OPTIMIZER == "adamw":
        return optim.AdamW(learning_rate=LEARNING_RATE)
    return optim.Adam(learning_rate=LEARNING_RATE)


def run_training(
    model_path: str,
    train_records: list[dict],
    valid_records: list[dict],
    adapter_file: Path,
    valid_sample_limit: int,
) -> dict:
    model, tokenizer = load(model_path, lazy=False)
    linear_to_lora_layers(model, 16, build_lora_config(), use_dora=False)
    train_dataset, valid_dataset = create_datasets(train_records, valid_records, tokenizer)

    effective_batch_size = max(1, min(BATCH_SIZE, len(train_records), len(valid_records)))
    args = TrainingArgs(
        batch_size=effective_batch_size,
        iters=ITERS,
        val_batches=max(1, min(len(valid_records), valid_sample_limit)),
        steps_per_report=max(1, min(10, ITERS)),
        steps_per_eval=max(1, ITERS),
        steps_per_save=max(1, ITERS),
        max_seq_length=MAX_SEQ_LENGTH,
        adapter_file=str(adapter_file),
        grad_checkpoint=GRAD_CHECKPOINT,
    )

    adapter_file.parent.mkdir(parents=True, exist_ok=True)
    train(model, choose_optimizer(), train_dataset, valid_dataset, args=args)

    val_loss = float(
        evaluate(
            model,
            valid_dataset,
            batch_size=1,
            num_batches=max(1, min(len(valid_records), valid_sample_limit)),
            max_seq_length=MAX_SEQ_LENGTH,
        )
    )
    tokens = sum(min(len((item.get("text") or "").split()), MAX_SEQ_LENGTH) for item in valid_records)
    return {
        "mode": "mlx",
        "val_loss": val_loss,
        "perplexity": float(mx.exp(mx.array(val_loss)).item()),
        "tokens_evaluated": tokens,
        "val_accuracy": max(0.0, min(1.0, 1.0 / (1.0 + val_loss))),
        "adapter_file": str(adapter_file),
    }


def main() -> None:
    args = parse_args()
    random.seed(42)
    os.environ.setdefault("PYTHONHASHSEED", "0")
    preset = apply_preset_overrides(args.preset)

    base_config = load_yaml("config/base_config.yaml")
    model_config = load_yaml("config/model_config.yaml")
    metadata = verify_frozen_integrity()

    model_path = Path(model_config["model"]["path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    train_path = Path(base_config["data"]["train"])
    valid_path = Path(base_config["data"]["valid"])
    train_records = load_jsonl(train_path)
    valid_records = load_jsonl(valid_path)
    if not train_records:
        raise ValueError("Training dataset is empty.")
    if not valid_records:
        raise ValueError("Validation dataset is empty.")

    experiment_id = compute_experiment_id()
    adapter_dir = Path(base_config["output"]["adapter_dir"]) / experiment_id
    adapter_file = adapter_dir / "adapters.safetensors"

    start = time.time()
    metrics = run_training(
        model_path=str(model_path),
        train_records=train_records,
        valid_records=valid_records,
        adapter_file=adapter_file,
        valid_sample_limit=base_config["evaluation"]["valid_sample_limit"],
    )
    train_time_seconds = time.time() - start

    ensure_adapter_config(adapter_dir)
    payload = {
        "experiment_id": experiment_id,
        "dataset_id": metadata["dataset_id"],
        "data_hash": metadata["data_hash"],
        "model_path": str(model_path),
        "config": {
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "iters": ITERS,
            "max_seq_length": MAX_SEQ_LENGTH,
            "optimizer": OPTIMIZER,
            "grad_checkpoint": GRAD_CHECKPOINT,
            "lr_schedule": LR_SCHEDULE,
        },
        "metrics": metrics,
        "train_time_seconds": round(train_time_seconds, 3),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "preset": preset,
    }
    metadata_path = save_adapter_metadata(Path(base_config["output"]["adapter_dir"]), payload)

    append_result(
        Path(base_config["output"]["results_tsv"]),
        {
            "experiment_id": experiment_id,
            "dataset_id": metadata["dataset_id"],
            "val_loss": metrics["val_loss"],
            "val_accuracy": metrics["val_accuracy"],
            "perplexity": metrics["perplexity"],
            "train_time_seconds": train_time_seconds,
            "mode": metrics["mode"],
            "status": "ok",
            "description": f"adapter={metadata_path.parent.name}",
        },
    )

    print("---")
    print(f"experiment_id:    {experiment_id}")
    print(f"dataset_id:       {metadata['dataset_id']}")
    print(f"mode:             {metrics['mode']}")
    print(f"val_loss:         {metrics['val_loss']:.6f}")
    print(f"val_accuracy:     {metrics['val_accuracy']:.6f}")
    print(f"perplexity:       {metrics['perplexity']:.6f}")
    print(f"tokens_evaluated: {metrics['tokens_evaluated']}")
    print(f"train_time:       {train_time_seconds:.2f}")
    print(f"lora_rank:        {LORA_RANK}")
    print(f"learning_rate:    {LEARNING_RATE}")
    print(f"batch_size:       {BATCH_SIZE}")
    if preset:
        print(f"preset:           {preset}")
    print(f"adapter_file:     {metrics['adapter_file']}")
    print(f"adapter_path:     {metadata_path.parent}")


if __name__ == "__main__":
    main()
