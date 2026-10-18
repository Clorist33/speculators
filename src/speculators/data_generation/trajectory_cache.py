"""Load a saved full-trajectory dataset and verify its preprocessing manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

from datasets import Dataset, load_from_disk

from speculators.data_generation import trajectory


def load_trajectory_cache(
    path: str | Path,
    *,
    model_path: str,
    max_length: int,
    block_size: int,
    overlength: str,
    shard_names: Sequence[str] | None = None,
    split_path: str | None = None,
    quarantine_path: str | None = None,
    tokenizer=None,
) -> tuple[Dataset, float, dict]:
    """Return the memory-mapped Arrow dataset and its authoritative split ratio."""
    path = Path(path)
    manifest_path = path / "preprocess_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Cannot read trajectory cache manifest {manifest_path}: {exc}"
        ) from exc

    def require(key: str, expected) -> None:
        if manifest.get(key) != expected:
            raise ValueError(
                f"Trajectory cache {key} mismatch: expected {expected!r}, "
                f"found {manifest.get(key)!r}"
            )

    require("data_format", "trajectory-jsonl")
    require("model_path", model_path)
    require("max_length", max_length)
    require("block_size", block_size)
    require("overlength", overlength)
    require("preprocessing_sha256", trajectory.sha(trajectory.__file__))
    if shard_names is not None:
        input_paths = manifest.get("input_paths")
        if (not isinstance(input_paths, list)
                or [Path(source).name for source in input_paths] != list(shard_names)):
            raise ValueError(
                "Trajectory cache source shards do not match training configuration"
            )
    if split_path is not None:
        require("split_path", str(Path(split_path).resolve()))
        require("split_sha256", trajectory.sha(split_path))
    if quarantine_path is not None:
        require("quarantine_path", str(Path(quarantine_path).resolve()))
        require("quarantine_sha256", trajectory.sha(quarantine_path))

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, use_fast=True
        )
    if not tokenizer.is_fast or not tokenizer.chat_template:
        raise ValueError(
            "Trajectory cache requires the original fast tokenizer and chat template"
        )
    require(
        "template_sha256",
        hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
    )

    dataset = load_from_disk(str(path))
    if not isinstance(dataset, Dataset):
        raise ValueError(f"Trajectory cache must contain one Arrow Dataset: {path}")
    columns = dataset.features
    if not {"input_ids", "loss_mask", "seq_len"} <= set(columns):
        raise ValueError("Trajectory cache is missing required Arrow columns")
    if (
        getattr(getattr(columns["input_ids"], "feature", None), "dtype", None)
        != "int64"
        or getattr(getattr(columns["loss_mask"], "feature", None), "dtype", None)
        != "bool"
        or getattr(columns["seq_len"], "dtype", None) != "int64"
    ):
        raise ValueError(
            "Trajectory cache Arrow column types do not match preprocessing"
        )
    train_rows = manifest.get("train_rows")
    validation_rows = manifest.get("validation_rows")
    ratio = manifest.get("train_ratio")
    if (not isinstance(train_rows, int) or not isinstance(validation_rows, int)
            or not isinstance(ratio, (int, float))
            or not 0 < train_rows < len(dataset)
            or train_rows + validation_rows != len(dataset)
            or int(len(dataset) * ratio) != train_rows):
        raise ValueError("Trajectory cache row counts or train ratio are inconsistent")
    counts = manifest.get("counts")
    if not isinstance(counts, dict) or sum(dataset["seq_len"]) != counts.get("input_tokens"):
        raise ValueError("Trajectory cache token count does not match its manifest")
    return dataset, float(ratio), manifest
