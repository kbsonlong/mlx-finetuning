from __future__ import annotations

import hashlib
import json
import os
import random
from datetime import datetime
from pathlib import Path

import yaml


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dump_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def compute_hash(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def maybe_link(src: Path, dest: Path) -> None:
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        os.symlink(src.resolve(), dest)
    except OSError:
        dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def normalize(records: list[dict]) -> list[dict]:
    normalized = []
    for idx, record in enumerate(records, start=1):
        text = record.get("text") or f"{record.get('prompt', '')}\n{record.get('completion', '')}".strip()
        normalized_record = dict(record)
        normalized_record["id"] = record.get("id", f"sample_{idx:05d}")
        normalized_record["prompt"] = record.get("prompt", "")
        normalized_record["completion"] = record.get("completion", "")
        normalized_record["text"] = text
        normalized.append(normalized_record)
    return normalized


def infer_dataset_metadata(records: list[dict]) -> dict:
    teacher_model = next(
        (
            record.get("teacher_model_version") or record.get("teacher_model")
            for record in records
            if record.get("teacher_model_version") or record.get("teacher_model")
        ),
        "synthetic-seed",
    )
    prompt_template_version = next(
        (record.get("prompt_template_version") for record in records if record.get("prompt_template_version")),
        "v1.0",
    )
    distillation_date = next(
        (record.get("distillation_date") for record in records if record.get("distillation_date")),
        datetime.now().date().isoformat(),
    )
    return {
        "teacher_model_version": teacher_model,
        "prompt_template_version": prompt_template_version,
        "distillation_date": distillation_date,
    }


def prepare_dataset(raw_path: Path, output_dir: Path, split_seed: int = 42) -> dict:
    records = normalize(load_jsonl(raw_path))
    if len(records) < 3:
        raise ValueError("Need at least 3 records to create train/valid/test splits.")
    inferred_metadata = infer_dataset_metadata(records)

    rng = random.Random(split_seed)
    rng.shuffle(records)

    n_total = len(records)
    n_valid = max(1, round(n_total * 0.1))
    n_test = max(1, round(n_total * 0.1))
    n_train = max(1, n_total - n_valid - n_test)

    train_records = records[:n_train]
    valid_records = records[n_train:n_train + n_valid]
    test_records = records[n_train + n_valid:]

    frozen_dir = output_dir / "frozen"
    mlx_dir = output_dir / "mlx_lm"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    mlx_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = f"ds_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    valid_name = f"valid_{dataset_id}.jsonl"
    test_name = f"test_{dataset_id}.jsonl"

    train_path = mlx_dir / "train.jsonl"
    valid_frozen_path = frozen_dir / valid_name
    test_frozen_path = frozen_dir / test_name
    valid_link = mlx_dir / "valid.jsonl"
    test_link = mlx_dir / "test.jsonl"

    dump_jsonl(train_path, train_records)
    dump_jsonl(valid_frozen_path, valid_records)
    dump_jsonl(test_frozen_path, test_records)
    maybe_link(valid_frozen_path, valid_link)
    maybe_link(test_frozen_path, test_link)

    data_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    metadata = {
        "dataset_id": dataset_id,
        "data_hash": data_hash,
        "split_seed": split_seed,
        "split_ratio": [len(train_records) / n_total, len(valid_records) / n_total, len(test_records) / n_total],
        "prompt_template_version": inferred_metadata["prompt_template_version"],
        "teacher_model_version": inferred_metadata["teacher_model_version"],
        "distillation_date": inferred_metadata["distillation_date"],
        "train_samples": len(train_records),
        "valid_samples": len(valid_records),
        "test_samples": len(test_records),
        "file_hashes": {
            valid_name: compute_hash(valid_frozen_path),
            test_name: compute_hash(test_frozen_path),
        },
    }
    with (frozen_dir / "metadata.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False, allow_unicode=True)

    os.chmod(valid_frozen_path, 0o444)
    os.chmod(test_frozen_path, 0o444)
    return metadata
