from __future__ import annotations

import hashlib
import os
from pathlib import Path

import yaml

FROZEN_DIR = Path("data/frozen")
METADATA_FILE = FROZEN_DIR / "metadata.yaml"


def load_data_metadata() -> dict:
    if not METADATA_FILE.exists():
        raise FileNotFoundError("Missing frozen dataset metadata. Run `python prepare.py` first.")
    with METADATA_FILE.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def compute_file_hash(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with filepath.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_frozen_integrity() -> dict:
    metadata = load_data_metadata()
    for dataset_name, expected_hash in metadata["file_hashes"].items():
        filepath = FROZEN_DIR / dataset_name
        if not filepath.exists():
            raise FileNotFoundError(f"Frozen dataset missing: {filepath}")
        actual_hash = compute_file_hash(filepath)
        if actual_hash != expected_hash:
            raise ValueError(f"Frozen dataset hash mismatch for {dataset_name}")
    return metadata


def protect_frozen_data() -> None:
    metadata = load_data_metadata()
    for dataset_name in metadata["file_hashes"]:
        os.chmod(FROZEN_DIR / dataset_name, 0o444)

