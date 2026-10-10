"""Prepare one real AscendC shard for DSpark, using native prepare-data masks.

The source split is the previously audited trajectory-level split.  Each chunk
is an independent native prepare-data invocation so a long render job can be
resumed without repeating completed chunks.  No hidden states are cached here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

MODEL = Path("/home/docker/Qwen3.6-35B-A3B-agentical-ascendc-hf-4t-bf16")
RAW = Path("/home/t00972278/dataset/ascendc-kernelgen-data/dataset/agent_dataset/train-00000.jsonl")
SPLIT = Path("/home/t00972278/Dspark_sft_docs/待删除/20260929/trajectory_split.jsonl")
SEQ_LENGTH = 262112
BLOCK_SIZE = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def source_chunks(root: Path, chunk_size: int) -> dict[str, list[Path]]:
    manifest_path = root / "source_manifest.json"
    split_by_line = {}
    groups = {"train": set(), "validation": set()}
    with SPLIT.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["shard"] != RAW.name:
                continue
            split_by_line[row["line"]] = row
            groups[row["split"]].add(row["group_id"])
    assert len(split_by_line) == 1000, len(split_by_line)
    assert not groups["train"] & groups["validation"]
    expected = {
        "raw_path": str(RAW),
        "raw_sha256": sha256(RAW),
        "split_path": str(SPLIT),
        "split_sha256": sha256(SPLIT),
        "seq_length": SEQ_LENGTH,
        "chunk_size": chunk_size,
        "trajectory_counts": dict(Counter(row["split"] for row in split_by_line.values())),
    }
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == expected, "Source manifest changed"
    else:
        assert not any(root.iterdir()), "Refusing to mix with another preparation run"
        handles: dict[tuple[str, int], object] = {}
        counters = Counter()
        try:
            with RAW.open() as handle:
                for line_number, line in enumerate(handle, 1):
                    meta = split_by_line[line_number]
                    side = meta["split"]
                    assert side in ("train", "validation"), side
                    source = json.loads(line)
                    assert list(source) == ["messages"]
                    chunk_number = counters[side] // chunk_size
                    key = side, chunk_number
                    if key not in handles:
                        folder = root / "chunks" / side / f"{chunk_number:04d}"
                        folder.mkdir(parents=True, exist_ok=False)
                        handles[key] = (folder / "source.jsonl").open("w")
                    handles[key].write(json.dumps({"conversations": source["messages"]}, ensure_ascii=False) + "\n")
                    counters[side] += 1
            assert line_number == 1000
            assert counters == expected["trajectory_counts"], (counters, expected)
        finally:
            for handle in handles.values():
                handle.close()
        write_json(manifest_path, expected)
    return {
        side: sorted((root / "chunks" / side).glob("*/source.jsonl"))
        for side in ("train", "validation")
    }


def prediction_mask(native_mask, block_size=BLOCK_SIZE):
    """Convert token-target masks to DSpark prediction positions."""
    if len(native_mask) <= block_size or any(x not in (0, 1) for x in native_mask):
        raise ValueError("Invalid native mask")
    shifted = list(native_mask[1:]) + [0]
    shifted[-block_size:] = [0] * block_size
    return shifted


def align_chunk(native_path: Path, output_path: Path) -> dict[str, int]:
    from datasets import load_from_disk

    native = load_from_disk(str(native_path)).with_format(None)
    assert {"input_ids", "loss_mask", "seq_len"} <= set(native.column_names)
    stats = Counter()

    def align(row: dict) -> dict:
        ids, mask = row["input_ids"], row["loss_mask"]
        assert len(ids) == len(mask) == row["seq_len"] <= SEQ_LENGTH
        assert len(ids) > BLOCK_SIZE
        assert set(mask) <= {0, 1}
        shifted = prediction_mask(mask)
        assert sum(shifted) > 0, "No valid DSpark anchors"
        stats["rows"] += 1
        stats["tokens"] += len(ids)
        stats["native_targets"] += sum(mask)
        stats["dspark_candidates"] += sum(shifted)
        stats["at_limit"] += len(ids) == SEQ_LENGTH
        return {"loss_mask": shifted}

    aligned = native.select_columns(["input_ids", "loss_mask", "seq_len"]).map(
        align, keep_in_memory=False, desc="DSpark mask alignment"
    )
    assert len(aligned) == stats["rows"]
    temporary = output_path.with_name(output_path.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    write_json(output_path.parent / "stats.json", dict(stats))
    aligned.save_to_disk(str(temporary))
    os.replace(temporary, output_path)
    return dict(stats)


def run_chunk(source: Path, endpoint: str, workers: int) -> dict[str, int]:
    folder = source.parent
    aligned = folder / "aligned"
    native = folder / "native"
    if (aligned / "state.json").is_file() and (folder / "stats.json").is_file():
        return json.loads((folder / "stats.json").read_text())
    if aligned.exists():
        shutil.rmtree(aligned)
    if not (native / "state.json").is_file():
        if native.exists():
            shutil.rmtree(native)
        command = [
            "speculators", "prepare-data", "--model", str(MODEL),
            "--data", str(source), "--output", str(native),
            "--seq-length", str(SEQ_LENGTH), "--render-endpoint", endpoint,
            "--minimum-valid-tokens", "9", "--num-preprocessing-workers", str(workers),
            "--skip-token-freq",
        ]
        write_json(folder / "prepare_command.json", command)
        with (folder / "prepare.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    return align_chunk(native, aligned)


def assemble(root: Path, sources: dict[str, list[Path]]) -> None:
    from datasets import concatenate_datasets, load_from_disk

    output = root / "data_arrow"
    if output.exists():
        assert (output / "state.json").is_file(), f"Incomplete final output: {output}"
        print(f"Final dataset already exists: {output}", flush=True)
        return
    train_paths = [source.parent / "aligned" for source in sources["train"]]
    val_paths = [source.parent / "aligned" for source in sources["validation"]]
    paths = train_paths + val_paths
    assert paths and all((path / "state.json").is_file() for path in paths)
    datasets = [load_from_disk(str(path)).with_format(None) for path in paths]
    train_rows = sum(len(dataset) for dataset in datasets[: len(train_paths)])
    val_rows = sum(len(dataset) for dataset in datasets[len(train_paths) :])
    assert train_rows > 0 and val_rows > 0
    combined = concatenate_datasets(datasets)
    assert len(combined) == train_rows + val_rows
    temporary = root / "data_arrow.partial"
    if temporary.exists():
        shutil.rmtree(temporary)
    combined.save_to_disk(str(temporary))
    ratio = math.nextafter(train_rows / len(combined), 1.0)
    assert int(len(combined) * ratio) == train_rows
    write_json(temporary / "split.json", {
        "train_rows": train_rows,
        "validation_rows": val_rows,
        "train_ratio": ratio,
        "split_method": "audited_trajectory_groups_before_native_prepare_data",
    })
    os.replace(temporary, output)
    print(f"Final dataset: {output}; train={train_rows}; validation={val_rows}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-endpoint", required=True)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    assert args.chunk_size > 0 and args.workers > 0
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    sources = source_chunks(root, args.chunk_size)
    print({side: len(paths) for side, paths in sources.items()}, flush=True)
    for side in ("train", "validation"):
        for number, source in enumerate(sources[side], 1):
            stats = run_chunk(source, args.render_endpoint, args.workers)
            print(f"{side} chunk {number}/{len(sources[side])}: {stats}", flush=True)
    assemble(root, sources)


if __name__ == "__main__":
    main()
