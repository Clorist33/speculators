"""Select a time-bounded, trajectory-aware subset from the prepared AscendC Arrow.

This only reuses already-rendered input_ids/loss_mask rows. It never calls the
teacher or changes the original source/Arrow. Each original trajectory supplies
at most a few deliberately separated assistant turns, including its longest
available context. The audited train/validation trajectory split is preserved.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from datasets import load_from_disk

from speculators.train.distributed_batch_sampler import (
    MultipackDistributedBatchSamplerV2,
)


DEFAULT_SOURCE = Path(
    "/home/t00972278/Dspark_sft_docs/formal-data/train00000_256k"
)
DEFAULT_OUTPUT = Path(
    "/home/t00972278/Dspark_sft_docs/formal-data/train00000_256k_budget6h"
)
SEQ_LENGTH = 262112
TRAIN_REPLICAS = 4


@dataclass(frozen=True)
class Row:
    side: str
    chunk: str
    trajectory: int
    global_index: int
    seq_len: int
    position: str


def message_key(messages: list[dict]) -> tuple[tuple[str, str], ...]:
    """Ignore JSONL-only empty tool metadata when matching native messages."""
    assert len(messages) >= 2
    key = tuple((m["role"], m["content"]) for m in messages[:2])
    assert all(isinstance(content, str) for _, content in key)
    return key


def trajectory_indices(native, sources: list[list[dict]]) -> list[list[int]]:
    """Identify each rendered row's source; prepare-data may interleave rows."""
    keys = [message_key(messages) for messages in sources]
    assert len(set(keys)) == len(keys), "Source first two messages are not unique"
    key_to_id = {key: i for i, key in enumerate(keys)}
    messages = native.select_columns(["messages"]).with_format(None)
    indices: list[list[int]] = [[] for _ in sources]
    for row_index, row in enumerate(messages):
        source_index = key_to_id[message_key(row["messages"])]
        indices[source_index].append(row_index)
    assert sum(map(len, indices)) == len(native)
    return indices


def candidate_rows(source: Path) -> tuple[list[Row], list[Row], list[Row], list[Row]]:
    """Return train base/optional and validation base/optional rows."""
    all_groups: dict[str, tuple[list[Row], list[Row]]] = {}
    global_offset = 0
    for side in ("train", "validation"):
        base: list[Row] = []
        optional: list[Row] = []
        for folder in sorted((source / "chunks" / side).glob("*")):
            if not folder.is_dir():
                continue
            aligned = load_from_disk(str(folder / "aligned")).with_format(None)
            native = load_from_disk(str(folder / "native")).with_format(None)
            lengths = np.asarray(aligned["seq_len"], dtype=np.int64)
            assert len(aligned) == len(native) == len(lengths)
            assert len(lengths) == 0 or (lengths <= SEQ_LENGTH).all()
            with (folder / "source.jsonl").open() as handle:
                sources = [json.loads(line)["conversations"] for line in handle]
            groups = trajectory_indices(native, sources)
            for trajectory, group in enumerate(groups):
                if not group:
                    continue
                ordered = sorted(group, key=lambda i: (lengths[i], i))
                choices = [
                    (ordered[0], "shortest"),
                    (ordered[-1], "longest"),
                    (ordered[(len(ordered) - 1) // 2], "middle"),
                    (ordered[(3 * (len(ordered) - 1)) // 4], "upper_quartile"),
                ]
                seen: set[int] = set()
                for choice_index, (local_index, position) in enumerate(choices):
                    if local_index in seen:
                        continue
                    seen.add(local_index)
                    row = Row(
                        side, folder.name, trajectory,
                        global_offset + local_index, int(lengths[local_index]),
                        position,
                    )
                    (base if choice_index < 2 else optional).append(row)
            global_offset += len(aligned)
        all_groups[side] = base, optional
    final = load_from_disk(str(source / "data_arrow"))
    split = json.loads((source / "data_arrow" / "split.json").read_text())
    assert global_offset == len(final)
    assert split["train_rows"] == sum(
        len(load_from_disk(str(folder / "aligned")))
        for folder in (source / "chunks" / "train").glob("*") if folder.is_dir()
    )
    return (*all_groups["train"], *all_groups["validation"])


def epoch_steps(rows: list[Row]) -> list[int]:
    lengths = [row.seq_len for row in sorted(rows, key=lambda r: r.global_index)]
    sampler = MultipackDistributedBatchSamplerV2(
        batch_max_length=SEQ_LENGTH,
        lengths=lengths,
        num_replicas=TRAIN_REPLICAS,
        rank=0,
    )
    result = []
    for epoch in range(5):
        sampler.set_epoch(epoch)
        result.append(len(sampler))
    return result


def report(name: str, rows: list[Row]) -> None:
    lengths = np.asarray([row.seq_len for row in rows], dtype=np.int64)
    print(
        name,
        "rows=", len(rows),
        "trajectories=", len({(r.side, r.chunk, r.trajectory) for r in rows}),
        "tokens=", int(lengths.sum()),
        "<32k=", int((lengths < 32768).sum()),
        "32-128k=", int(((lengths >= 32768) & (lengths < 131072)).sum()),
        "128-224k=", int(((lengths >= 131072) & (lengths < 224000)).sum()),
        ">=224k=", int((lengths >= 224000).sum()),
        ">=250k=", int((lengths >= 250000).sum()),
        flush=True,
    )
    print(name, "steps_by_epoch=", epoch_steps(rows), flush=True)


def select_train_to_budget(
    base: list[Row], optional: list[Row], target_steps: int
) -> tuple[list[Row], list[Row]]:
    """Remove redundant short optional turns until all epochs fit the step budget.

    Every trajectory retains its shortest and longest turn. Longer optional
    turns are favored, preserving genuine long-context supervision.
    """
    drop_order = sorted(
        optional,
        key=lambda row: (row.seq_len, row.chunk, row.trajectory, row.global_index),
    )

    def selection(num_dropped: int) -> list[Row]:
        return base + drop_order[num_dropped:]

    dropped = 0
    steps = epoch_steps(selection(dropped))
    while max(steps) > target_steps and dropped < len(drop_order):
        dropped = min(dropped + 20, len(drop_order))
        steps = epoch_steps(selection(dropped))
    if max(steps) > target_steps:
        raise ValueError(f"Base rows exceed {target_steps} steps: {steps}")
    for candidate in range(max(0, dropped - 19), dropped + 1):
        candidate_steps = epoch_steps(selection(candidate))
        if max(candidate_steps) <= target_steps:
            dropped = candidate
            break
    return selection(dropped), drop_order[:dropped]


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def materialize(source: Path, output: Path, train: list[Row], val: list[Row],
                dropped: list[Row], target_steps: int) -> None:
    train = sorted(train, key=lambda row: row.global_index)
    val = sorted(val, key=lambda row: row.global_index)
    rows = train + val
    indices = [row.global_index for row in rows]
    assert len(indices) == len(set(indices))
    source_split = json.loads((source / "data_arrow" / "split.json").read_text())
    assert all(row.global_index < source_split["train_rows"] for row in train)
    assert all(row.global_index >= source_split["train_rows"] for row in val)
    original = load_from_disk(str(source / "data_arrow")).with_format(None)
    assert max(indices) < len(original)

    output.mkdir(parents=True, exist_ok=False)
    with (output / "selected_rows.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    train_steps = epoch_steps(train)
    val_steps = epoch_steps(val)
    write_json(output / "selection_manifest.json", {
        "source_data_arrow": str(source / "data_arrow"),
        "source_manifest": str(source / "source_manifest.json"),
        "policy": "each trajectory's shortest and longest assistant turn, plus longer middle/upper-quartile turns within the step budget",
        "target_max_train_steps_per_epoch": target_steps,
        "train_rows": len(train),
        "validation_rows": len(val),
        "train_trajectories": len({(row.chunk, row.trajectory) for row in train}),
        "validation_trajectories": len({(row.chunk, row.trajectory) for row in val}),
        "train_steps_by_epoch": train_steps,
        "validation_steps_by_epoch": val_steps,
        "train_input_tokens": sum(row.seq_len for row in train),
        "validation_input_tokens": sum(row.seq_len for row in val),
        "train_rows_ge_224k": sum(row.seq_len >= 224000 for row in train),
        "train_rows_ge_250k": sum(row.seq_len >= 250000 for row in train),
        "optional_rows_dropped": len(dropped),
        "teacher_features": "generated online; no features saved in this dataset",
    })

    temporary = output / "data_arrow.partial"
    selected = original.select(indices)
    selected.save_to_disk(str(temporary))
    ratio = math.nextafter(len(train) / len(rows), 1.0)
    assert int(len(rows) * ratio) == len(train)
    write_json(temporary / "split.json", {
        "train_rows": len(train),
        "validation_rows": len(val),
        "train_ratio": ratio,
        "split_method": "source_audited_trajectory_groups_then_turn_subselection",
    })
    os.replace(temporary, output / "data_arrow")
    reopened = load_from_disk(str(output / "data_arrow")).with_format(None)
    assert len(reopened) == len(rows)
    assert list(reopened["seq_len"]) == [row.seq_len for row in rows]
    print("Final subset:", output / "data_arrow", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--target-steps", type=int, default=290)
    args = parser.parse_args()
    assert args.target_steps > 0

    train_base, train_optional, val_base, val_optional = candidate_rows(args.source)
    report("train_base", train_base)
    report("train_all_four", train_base + train_optional)
    report("val_base", val_base)
    report("val_all_four", val_base + val_optional)
    selected_train, dropped = select_train_to_budget(
        train_base, train_optional, args.target_steps
    )
    report("selected_train", selected_train)
    print("optional_rows_dropped=", len(dropped), flush=True)
    if args.plan:
        return
    materialize(args.source, args.output, selected_train, val_base, dropped,
                args.target_steps)


if __name__ == "__main__":
    main()
