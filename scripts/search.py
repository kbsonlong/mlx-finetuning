from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from results import load_results
else:
    from scripts.results import load_results


def load_search_space() -> dict:
    with open("config/search_space.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def random_sample_config(search_space: dict, fixed_params: dict | None = None) -> dict:
    config = {}
    if fixed_params:
        config.update(fixed_params)
    for param, values in search_space.get("random", {}).items():
        config[param] = random.choice(values)
    return config


def generate_stage1_configs(n_samples: int = 20, seed: int = 42) -> list[dict]:
    random.seed(seed)
    search_space = load_search_space()
    stage1 = search_space["stage1_quick_screen"]
    configs = []
    for i in range(n_samples):
        configs.append(
            {
                "experiment_id": f"stage1_exp_{i:04d}",
                "stage": "quick_screen",
                **random_sample_config(stage1, stage1.get("fixed", {})),
            }
        )
    return configs


def generate_stage2_configs(top_n: int = 5, repeats: int = 3) -> list[dict]:
    search_space = load_search_space()
    stage2 = search_space["stage2_fine_validation"]
    results = sorted(load_results(), key=lambda item: item["val_loss"])[:top_n]
    configs = []
    for rank, row in enumerate(results, start=1):
        adapter_name = row["description"].split("adapter=", 1)[-1]
        metadata_path = Path("outputs/adapters") / adapter_name / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        base_config = metadata["config"]
        for repeat in range(repeats):
            configs.append(
                {
                    "experiment_id": f"stage2_rank{rank:02d}_rep{repeat+1}",
                    "stage": "fine_validation",
                    "seed": stage2["seeds"][repeat % len(stage2["seeds"])],
                    **base_config,
                }
            )
    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate search configs from the configured strategy.")
    parser.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", help="Optional output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "stage1":
        payload = generate_stage1_configs(n_samples=args.n_samples, seed=args.seed)
    else:
        payload = generate_stage2_configs(top_n=args.top_n, repeats=args.repeats)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote: {output_path}")
        return
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
