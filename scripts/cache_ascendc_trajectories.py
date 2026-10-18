"""Save the training trajectory preprocessing result as a reusable Arrow dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from speculators.data_generation.trajectory import prepare_in_memory_trajectories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument(
        "--overlength", choices=("error", "skip", "truncate"), required=True
    )
    parser.add_argument("--shards", nargs="+", required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            parser.error(f"Output directory must be empty: {output}")
    else:
        output.mkdir(parents=True)

    print(f"Preprocessing {len(args.shards)} shards into memory...", flush=True)
    dataset, ratio, manifest = prepare_in_memory_trajectories(
        args.input,
        args.split,
        args.model,
        args.max_length,
        args.block_size,
        args.overlength,
        shard_names=args.shards,
        quarantine_path=args.quarantine,
    )
    print(f"Preprocessed: {manifest['counts']}", flush=True)
    print(f"Saving {len(dataset)} rows to {output}...", flush=True)
    dataset.save_to_disk(str(output), max_shard_size="1GB")
    manifest["train_ratio"] = ratio
    manifest["saved_dataset_path"] = str(output)
    (output / "preprocess_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Saved {len(dataset)} rows (train={manifest['train_rows']}, "
          f"validation={manifest['validation_rows']}).", flush=True)


if __name__ == "__main__":
    main()
