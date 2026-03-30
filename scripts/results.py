from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


RESULTS_TSV = Path("outputs/results.tsv")


def load_results(path: Path = RESULTS_TSV) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    for row in rows:
        for key in ("val_loss", "val_accuracy", "perplexity", "train_time_seconds"):
            if row.get(key):
                row[key] = float(row[key])
    return rows


def get_best(results: list[dict]) -> dict | None:
    if not results:
        return None
    return min(results, key=lambda item: item["val_loss"])


def summarize(results: list[dict]) -> dict:
    if not results:
        return {"count": 0}
    best = get_best(results)
    return {
        "count": len(results),
        "best_experiment_id": best["experiment_id"],
        "best_val_loss": best["val_loss"],
        "avg_val_loss": sum(row["val_loss"] for row in results) / len(results),
        "modes": sorted({row["mode"] for row in results}),
        "datasets": sorted({row["dataset_id"] for row in results}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query experiment history.")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_results()
    if args.summary:
        payload = summarize(results)
    else:
        payload = sorted(results, key=lambda item: item["val_loss"])[: args.top]
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if args.summary:
        for key, value in payload.items():
            print(f"{key}: {value}")
        return
    for row in payload:
        print(
            f"{row['experiment_id']}  val_loss={row['val_loss']:.6f}  "
            f"mode={row['mode']}  dataset={row['dataset_id']}"
        )


if __name__ == "__main__":
    main()

