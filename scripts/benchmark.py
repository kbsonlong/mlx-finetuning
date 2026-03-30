from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml


BENCHMARK_PRESETS = {
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

SUMMARY_FIELDS = {
    "experiment_id",
    "dataset_id",
    "mode",
    "val_loss",
    "val_accuracy",
    "perplexity",
    "tokens_evaluated",
    "train_time",
    "lora_rank",
    "learning_rate",
    "batch_size",
    "preset",
    "adapter_file",
    "adapter_path",
}


def parse_metrics(stdout: str) -> dict:
    summary_block = stdout.rsplit("---", maxsplit=1)[-1]
    metrics = {}
    for line in summary_block.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized_key = key.strip()
        if normalized_key not in SUMMARY_FIELDS:
            continue
        metrics[normalized_key] = value.strip()
    return metrics


def run_train(preset_name: str) -> dict:
    proc = subprocess.run(
        ["python3", "train.py", "--preset", preset_name],
        capture_output=True,
        text=True,
        check=True,
    )
    metrics = parse_metrics(proc.stdout)
    metrics["preset"] = preset_name
    return metrics


def make_report(results: list[dict]) -> dict:
    baseline = next((row for row in results if row["preset"] == "baseline"), None)
    train_times = [float(row.get("train_time", 0.0)) for row in results]
    val_losses = [float(row["val_loss"]) for row in results if "val_loss" in row]
    return {
        "benchmark": {
            "baseline": {
                "val_loss": float(baseline["val_loss"]) if baseline else None,
                "train_time_minutes": (float(baseline.get("train_time", 0.0)) / 60.0) if baseline else None,
            },
            "recommendations": {
                "time_budget_minutes": round(max(train_times, default=0.0) / 60.0 + 5, 2),
                "max_memory_gb": 16.0,
                "target_val_loss": round(min(val_losses, default=0.0), 6) if val_losses else None,
            },
            "runs": results,
        }
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark presets against train.py.")
    parser.add_argument("--presets", nargs="*", default=list(BENCHMARK_PRESETS.keys()))
    parser.add_argument("--output", default="config/benchmark_results.yaml")
    parser.add_argument("--json-output", default="outputs/benchmark_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = [run_train(preset_name) for preset_name in args.presets]
    report = make_report(results)

    yaml_path = Path(args.output)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(report, handle, sort_keys=False, allow_unicode=True)

    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"yaml_report: {yaml_path}")
    print(f"json_report: {json_path}")
    print(f"runs: {len(results)}")


if __name__ == "__main__":
    main()
