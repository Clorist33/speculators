"""Full-trajectory tokenization shared by offline preparation and in-memory training.

This module does not load teacher weights or write a tokenized dataset to disk.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

CONTROL = re.compile(r'<\|im_start\|>|<\|im_end\|>')
ASSISTANT = re.compile(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>\n', re.S)


def normalize(example):
    messages = copy.deepcopy(example['messages'])
    if not messages or messages[-1]['role'] != 'assistant':
        raise ValueError('Trajectory must end in assistant; no silent removal of trailing turns')
    for msg in messages:
        if msg['role'] not in ('system', 'user', 'assistant', 'tool'):
            raise ValueError('Unsupported message role')
        for key in ('content', 'reasoning_content'):
            value = msg.get(key)
            if value is not None and not isinstance(value, str):
                raise ValueError('Only text trajectories are supported')
        for call in msg.get('tool_calls') or []:
            fn = call.get('function', call)
            if not isinstance(fn.get('name'), str) or not fn['name']:
                raise ValueError('Missing tool function name')
            args = fn.get('arguments', {})
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(args, dict):
                raise ValueError('Tool arguments must be a JSON object')
            fn['arguments'] = args
    # Structural markers in recorded text would make boundary detection ambiguous.
    if CONTROL.search(json.dumps(messages, ensure_ascii=False)) or CONTROL.search(json.dumps(example.get('tools'), ensure_ascii=False)):
        raise ValueError('Reserved chat delimiters inside source data')
    return messages


def prediction_mask(targets, block_size=8):
    """Map token targets to DSpark next-token prediction positions."""
    if block_size < 1 or any(x not in (0, 1) for x in targets):
        raise ValueError('Invalid target mask or block size')
    if not targets:
        return []
    mask = list(targets[1:]) + [0]
    mask[max(0, len(mask) - block_size):] = [0] * min(block_size, len(mask))
    return mask


def encode_trajectory(example, tokenizer, max_length, block_size, overlength):
    messages = normalize(example)
    text = tokenizer.apply_chat_template(
        messages, tools=example.get('tools'), tokenize=False,
        add_generation_prompt=False, preserve_thinking=True,
    )
    spans = [(m.start(1), m.end()) for m in ASSISTANT.finditer(text)]
    if len(spans) != sum(m['role'] == 'assistant' for m in messages):
        raise ValueError('Unsupported chat template: assistant boundary count differs')
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded['input_ids']
    targets = []
    turn = 0
    for start, end in encoded['offset_mapping']:
        while turn < len(spans) and start >= spans[turn][1]:
            turn += 1
        overlap = turn < len(spans) and end > spans[turn][0] and start < spans[turn][1]
        if overlap and not (start >= spans[turn][0] and end <= spans[turn][1]):
            raise ValueError('Token crosses assistant boundary')
        targets.append(int(overlap and end > start))
    original_length = len(ids)
    original_targets = sum(targets)
    info = dict(original_tokens=original_length, original_target_tokens=original_targets,
                assistant_turns=len(spans), overlength=original_length > max_length)
    if original_length > max_length:
        if overlength == 'error':
            raise ValueError(f'Trajectory has {original_length} tokens, limit {max_length}')
        if overlength == 'skip':
            return None, dict(info, status='skipped_overlength'), text, targets
        ids, targets = ids[:max_length], targets[:max_length]
    mask = prediction_mask(targets, block_size)
    info.update(retained_tokens=len(ids), retained_target_tokens=sum(targets),
                removed_tokens=original_length-len(ids),
                removed_target_tokens=original_targets-sum(targets),
                prediction_candidates=sum(mask))
    if not any(mask):
        return None, dict(info, status='skipped_no_candidates'), text, targets
    return dict(input_ids=ids, loss_mask=mask, seq_len=len(ids)), dict(info, status='kept'), text, targets


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load_trajectory_splits(path, shard_names: Sequence[str]):
    """Read selected source-line splits and reject group leakage across shards."""
    splits = {name: {} for name in shard_names}
    groups = {"train": set(), "validation": set()}
    with Path(path).open() as handle:
        for line in handle:
            meta = json.loads(line)
            shard = meta["shard"]
            if shard not in splits:
                continue
            number = meta["line"]
            if number in splits[shard] or meta["split"] not in groups:
                raise ValueError("Invalid/duplicate trajectory split entry")
            splits[shard][number] = meta
            groups[meta["split"]].add(meta["group_id"])
    if any(not entries for entries in splits.values()):
        raise ValueError("No trajectory split entries for one or more selected shards")
    if groups["train"] & groups["validation"]:
        raise ValueError("Train/validation trajectory group leakage")
    return splits


def load_trajectory_split(path, shard):
    """Read one shard's explicit split, preserving the single-file API."""
    return load_trajectory_splits(path, [shard])[shard]


def inspect_trajectory_inputs(
    input_path,
    split_path,
    shard_names: Sequence[str] | None = None,
    quarantine_path=None,
    *,
    check_lines=False,
):
    """Resolve selected shards and verify that only quarantined lines lack a split."""
    source = Path(input_path)
    if shard_names is None:
        if not source.is_file():
            raise ValueError(f"Raw trajectory JSONL missing at {source}")
        sources = [source]
    else:
        if not source.is_dir() or not shard_names:
            raise ValueError("Trajectory shards require a data directory and names")
        if len(set(shard_names)) != len(shard_names):
            raise ValueError("Duplicate trajectory shard name")
        if any(Path(name).name != name or not name.endswith(".jsonl") for name in shard_names):
            raise ValueError("Trajectory shards must be JSONL filenames")
        sources = [source / name for name in shard_names]
        for item in sources:
            if not item.is_file():
                raise ValueError(f"Raw trajectory JSONL missing at {item}")

    splits = load_trajectory_splits(split_path, [item.name for item in sources])
    quarantined = set()
    if quarantine_path is not None:
        for record in json.loads(Path(quarantine_path).read_text()):
            shard_id, line_number = record["id"].split(":", 1)
            shard = f"train-{shard_id}.jsonl"
            if shard not in splits:
                continue
            key = (shard, int(line_number))
            if key in quarantined or key[1] in splits[shard]:
                raise ValueError(f"Invalid/duplicate quarantined source line: {key}")
            quarantined.add(key)

    if check_lines:
        for item in sources:
            seen = set()
            with item.open() as handle:
                for number, _ in enumerate(handle, 1):
                    seen.add(number)
                    if number not in splits[item.name] and (item.name, number) not in quarantined:
                        raise ValueError(f"Source {item.name} line {number} has no trajectory split")
            expected = set(splits[item.name]) | {
                number for shard, number in quarantined if shard == item.name
            }
            if seen != expected:
                raise ValueError(f"Trajectory split/quarantine has lines absent from {item.name}")
    return sources, splits, quarantined


def prepare_in_memory_trajectories(
    input_path,
    split_path,
    model_path,
    max_length,
    block_size,
    overlength="error",
    *,
    tokenizer=None,
    shard_names: Sequence[str] | None = None,
    quarantine_path=None,
):
    """Tokenize once at startup into RAM; return table, split ratio and audit.

    Packing needs token lengths before iteration. An in-memory Arrow table avoids
    a large Python-object cache and is never saved with save_to_disk/from_json.
    Each distributed process owns its own copy. Source messages are not retained.
    """
    import pyarrow as pa
    from datasets import Dataset
    from transformers import AutoTokenizer

    if max_length <= block_size or block_size < 1:
        raise ValueError("max_length must exceed positive block_size")
    if overlength not in ("error", "skip", "truncate"):
        raise ValueError("Unknown overlength policy")
    sources, splits, quarantined = inspect_trajectory_inputs(
        input_path, split_path, shard_names, quarantine_path
    )
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, use_fast=True
        )
    if not tokenizer.is_fast:
        raise ValueError("Fast tokenizer required for exact character boundaries")
    schema = pa.schema([
        ("input_ids", pa.list_(pa.int64())),
        ("loss_mask", pa.list_(pa.bool_())),
        ("seq_len", pa.int64()),
    ])
    batches = {"train": [], "validation": []}
    counts = Counter()
    audit = []
    for source in sources:
        source_lines = set()
        with source.open() as handle:
            for number, line in enumerate(handle, 1):
                source_lines.add(number)
                if (source.name, number) in quarantined:
                    counts["quarantined"] += 1
                    continue
                if number not in splits[source.name]:
                    raise ValueError(f"Source {source.name} line {number} has no trajectory split")
                meta = splits[source.name][number]
                try:
                    row, info, _, _ = encode_trajectory(
                        json.loads(line), tokenizer, max_length, block_size, overlength
                    )
                except Exception as exc:
                    raise ValueError(f"Source {source.name} line {number}: {exc}") from exc
                side = meta["split"]
                audit.append(dict(info, source_shard=source.name, source_line=number,
                                  split=side, group_id=meta["group_id"]))
                counts["source_trajectories"] += 1
                counts[info["status"]] += 1
                counts["overlength"] += int(info["overlength"])
                if row is None:
                    continue
                batches[side].append(pa.RecordBatch.from_arrays([
                    pa.array([row["input_ids"]], type=schema.field("input_ids").type),
                    pa.array([[bool(x) for x in row["loss_mask"]]], type=schema.field("loss_mask").type),
                    pa.array([row["seq_len"]], type=pa.int64()),
                ], schema=schema))
                counts[side] += 1
                counts["input_tokens"] += row["seq_len"]
                counts["prediction_candidates"] += sum(row["loss_mask"])
        expected = set(splits[source.name]) | {
            number for shard, number in quarantined if shard == source.name
        }
        if source_lines != expected:
            raise ValueError(f"Trajectory split/quarantine has lines absent from {source.name}")
    if not counts["train"] or not counts["validation"]:
        raise ValueError("Both train and validation need retained trajectories")
    table = pa.Table.from_batches(batches["train"] + batches["validation"], schema=schema)
    dataset = Dataset(table)
    ratio = math.nextafter(counts["train"] / len(dataset), 1.0)
    assert int(len(dataset) * ratio) == counts["train"]
    input_hashes = {source.name: sha(source) for source in sources}
    manifest = dict(
        data_format="trajectory-jsonl", storage="memory",
        input_paths=[str(source.resolve()) for source in sources],
        input_sha256=input_hashes if len(sources) > 1 else next(iter(input_hashes.values())),
        split_path=str(Path(split_path).resolve()),
        split_sha256=sha(split_path), model_path=str(model_path),
        quarantine_path=str(Path(quarantine_path).resolve()) if quarantine_path else None,
        quarantine_sha256=sha(quarantine_path) if quarantine_path else None,
        template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        preprocessing_sha256=sha(__file__), max_length=max_length,
        block_size=block_size, overlength=overlength, preserve_thinking=True,
        counts=dict(counts), train_rows=counts["train"],
        validation_rows=counts["validation"], train_ratio=ratio, trajectories=audit,
    )
    if len(sources) == 1:
        manifest["input_path"] = str(sources[0].resolve())
    assert not dataset.cache_files
    return dataset, ratio, manifest
