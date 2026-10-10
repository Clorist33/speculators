"""Build auditable DSpark prediction-position masks and a JSONL/Arrow bridge.

No model weights are loaded. JSONL is the source of truth; Arrow is a run cache.
"""

import argparse
import bisect
import copy
import hashlib
import itertools
import json
import math
import re
import sys
from pathlib import Path

DEFAULT_MODEL = "/home/docker/Qwen3.6-35B-A3B-agentical-ascendc-hf-4t-bf16"
DEFAULT_DIR = Path(
    "/home/t00972278/dataset/ascendc-kernelgen-data/dataset/agent_dataset"
)
BLOCK_SIZE = 8
SOURCE_RECORDS = 16
VOCAB_SIZE = 248320
POLICY = "reuse_first_training_window_for_smoke_validation_only"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def normalized(messages):
    result = copy.deepcopy(messages)
    for m in result:
        require(
            m["role"] in ("system", "user", "assistant", "tool"),
            "Invalid data: m['role'] in ('system', 'user', 'assistant', 'tool')",
        )
        require(
            m.get("content") is None or isinstance(m["content"], str),
            "Invalid data: m.get('content') is None or isinstance(m['content'], str)",
        )
        for call in m.get("tool_calls") or []:
            function = call.get("function", call)
            args = function.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            require(isinstance(args, dict), "Tool arguments must decode to an object")
            function["arguments"] = args
    return result


def encode_with_mask(tokenizer, messages):
    """Instrument only template control flow; verify byte-identical clean output.

    Markers are removed before tokenization. They do not modify the saved model
    template, messages or token IDs. Offset mappings locate assistant spans.
    """
    template = tokenizer.chat_template
    marker = "__DSPARK_SPAN_30a102d45b3c4d55__"
    require(
        marker not in json.dumps(messages, ensure_ascii=False),
        "Invalid data: marker not in json.dumps(messages, ensure_ascii=False)",
    )
    start = '{%- elif message.role == "assistant" %}'
    stop = '{%- elif message.role == "tool" %}'
    require(
        template.count(start) == template.count(stop) == 1,
        "Invalid data: template.count(start) == template.count(stop) == 1",
    )
    annotated = template.replace(
        start, start + "{{- '" + marker + "START_' ~ loop.index0 ~ '__' }}"
    )
    annotated = annotated.replace(
        stop, "{{- '" + marker + "END_' ~ loop.index0 ~ '__' }}" + stop
    )
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": False,
        "enable_thinking": True,
    }
    clean = tokenizer.apply_chat_template(messages, **kwargs)
    marked = tokenizer.apply_chat_template(messages, chat_template=annotated, **kwargs)
    pattern = re.compile(re.escape(marker) + r"(START|END)_(\d+)__")
    pieces, spans = [], []
    cursor = length = 0
    active = None
    for match in pattern.finditer(marked):
        piece = marked[cursor : match.start()]
        pieces.append(piece)
        length += len(piece)
        kind, index = match.group(1), int(match.group(2))
        if kind == "START":
            require(
                active is None and messages[index]["role"] == "assistant",
                "Invalid data: active is None and messages[index]['role'] == 'assistant'",  # noqa: E501
            )
            active = (index, length)
        else:
            require(
                active is not None and active[0] == index,
                "Invalid data: active is not None and active[0] == index",
            )
            spans.append((index, active[1], length))
            active = None
        cursor = match.end()
    pieces.append(marked[cursor:])
    require(
        active is None and "".join(pieces) == clean,
        "Instrumentation changed rendered text",
    )
    require(
        len(spans) == sum(m["role"] == "assistant" for m in messages),
        "Invalid data: len(spans) == sum((m['role'] == 'assistant' for m in messages))",
    )
    # The generation prompt pre-fills the assistant header and (when present)
    # <think>\n. Their tokens remain context, as do inter-message newlines.
    ranges = []
    for index, span_begin, span_end in spans:
        begin, end = span_begin, span_end
        header = "<|im_start|>assistant\n"
        require(
            clean[begin:end].startswith(header),
            "Invalid data: clean[begin:end].startswith(header)",
        )
        begin += len(header)
        if clean[begin:end].startswith("<think>\n"):
            begin += len("<think>\n")
        require(
            clean[begin:end].endswith("<|im_end|>\n"),
            "Invalid data: clean[begin:end].endswith('<|im_end|>\\n')",
        )
        end -= 1
        ranges.append(
            {
                "message_index": index,
                "start_char": begin,
                "end_char": end,
                "has_tool_calls": bool(messages[index].get("tool_calls")),
            }
        )
    encoded = tokenizer(
        clean,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        verbose=False,
    )
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    canonical = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, enable_thinking=True
    )
    if not isinstance(canonical, list):
        canonical = canonical["input_ids"]
    require(ids == canonical, "Tokenization differs from original teacher template")
    mask, owners = [], []
    boundaries = [span["start_char"] for span in ranges]
    for a, b in offsets:
        j = bisect.bisect_right(boundaries, b - 1) - 1
        inside = j >= 0 and a < b and a < ranges[j]["end_char"]
        if inside:
            span = ranges[j]
            # BPE may merge scaffold whitespace with the first generated
            # whitespace. Supervise that indivisible token, never header text.
            if a < span["start_char"]:
                require(
                    clean[a : span["start_char"]].isspace(),
                    ("Token crosses a non-whitespace prompt boundary", (a, b), span),
                )
            require(b <= span["end_char"], ("Token crosses response end", (a, b), span))
        mask.append(int(inside))
        owners.append(ranges[j]["message_index"] if inside else -1)
    require(
        len(ids) == len(mask) and sum(mask) > 0,
        "Invalid data: len(ids) == len(mask) and sum(mask) > 0",
    )
    return ids, mask, owners, ranges


def windows(ids, target_mask, length, overlap, block):
    """Sliding token windows; overlap is context, each eligible target once."""
    prediction_mask = target_mask[1:] + [0]
    previous_stop = 0
    stride = length - overlap
    for start in range(0, len(ids), stride):
        stop = min(start + length, len(ids))
        supervision_start = max(start, previous_stop)
        supervision_stop = max(start, stop - block)
        mask = [
            prediction_mask[i] if supervision_start <= i < supervision_stop else 0
            for i in range(start, stop)
        ]
        previous_stop = max(previous_stop, supervision_stop)
        if sum(mask):
            yield start, stop, mask
        if stop == len(ids):
            break


def build(args):  # noqa: C901 -- sequential build and provenance audit
    from transformers import AutoTokenizer  # noqa: PLC0415 -- check stays stdlib-only

    source, out = Path(args.source), Path(args.output_dir)
    require(
        args.length > args.overlap > BLOCK_SIZE,
        "Invalid data: args.length > args.overlap > BLOCK_SIZE",
    )
    out.mkdir(parents=True, exist_ok=True)
    full_path = out / "train-00000.first16.masked.jsonl"
    window_path = out / f"train-00000.first16.windows{args.length}.masked.jsonl"
    manifest_path = Path(str(window_path) + ".manifest.json")
    preview_path = out / "train-00000.first16.mask_preview.jsonl"
    require(
        not any(
            p.exists() for p in (full_path, window_path, manifest_path, preview_path)
        ),
        "Outputs already exist",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    require(tokenizer.is_fast, "Reliable offsets require a fast tokenizer")
    records, raw_hash = [], hashlib.sha256()
    with source.open("rb") as f:
        for raw in itertools.islice(f, 16):
            raw_hash.update(raw)
            records.append(json.loads(raw))
    require(
        len(records) == SOURCE_RECORDS, "Invalid data: len(records) == SOURCE_RECORDS"
    )
    stats, previews = [], []
    total_windows = 0
    # Write temporary files; a failed boundary check never publishes partial data.
    full_tmp, window_tmp = (
        Path(str(full_path) + ".tmp"),
        Path(str(window_path) + ".tmp"),
    )
    with full_tmp.open("x") as full, window_tmp.open("x") as output:
        for source_index, record in enumerate(records):
            messages = normalized(record["messages"])
            ids, target, owners, spans = encode_with_mask(tokenizer, messages)
            prediction = target[1:] + [0]
            full_mask = prediction.copy()
            full_mask[-8:] = [0] * min(8, len(full_mask))
            full_row = dict(
                record,
                source_record_index=source_index,
                input_ids=ids,
                assistant_token_mask=target,
                loss_mask=full_mask,
                seq_len=len(ids),
                mask_alignment="next_token_prediction_position",
                assistant_spans=spans,
            )
            full.write(json.dumps(full_row, ensure_ascii=False) + "\n")
            covered = bytearray(len(ids))
            count = supervised = 0
            for start, stop, mask in windows(ids, target, args.length, args.overlap, 8):
                row = {
                    "input_ids": ids[start:stop],
                    "loss_mask": mask,
                    "seq_len": stop - start,
                    "source_record_index": source_index,
                    "source_token_start": start,
                    "source_token_end": stop,
                    "mask_alignment": "next_token_prediction_position",
                }
                for i, value in enumerate(mask, start):
                    if value:
                        require(
                            not covered[i]
                            and target[i + 1] == 1
                            and (owners[i + 1] >= 0),
                            "Invalid data: not covered[i] and target[i + 1] == 1 and (owners[i + 1] >= 0)",  # noqa: E501
                        )
                        covered[i] = 1
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
                supervised += sum(mask)
            require(
                list(covered) == full_mask,
                "Window masks lost or duplicated eligible supervision",
            )
            require(
                count > 0, f"Source record {source_index} contributes no trainable rows"
            )
            # Human-readable boundary probes for plain answers and tool calls.
            selected = spans[:2]
            non_tool = next(
                (span for span in spans if not span["has_tool_calls"]), None
            )
            if non_tool is not None and non_tool not in selected:
                selected.append(non_tool)
            for span in selected:
                positions = [
                    i
                    for i, owner in enumerate(owners)
                    if owner == span["message_index"]
                ]
                require(positions, "Assistant message has no supervised tokens")
                for center in (positions[0], positions[-1]):
                    for i in range(max(0, center - 3), min(len(ids) - 1, center + 4)):
                        previews.append(
                            {
                                "source_record_index": source_index,
                                "token_position": i,
                                "token_id": ids[i],
                                "token": tokenizer.decode([ids[i]]),
                                "next_token": tokenizer.decode([ids[i + 1]]),
                                "assistant_token_mask": target[i],
                                "loss_mask": full_mask[i],
                                "next_token_assistant_message": owners[i + 1],
                            }
                        )
            stats.append(
                {
                    "source_record_index": source_index,
                    "messages": len(messages),
                    "tokens": len(ids),
                    "assistant_target_tokens": sum(target),
                    "supervised_prediction_positions": supervised,
                    "training_windows": count,
                }
            )
            total_windows += count
            print(json.dumps(stats[-1]), flush=True)
    full_tmp.rename(full_path)
    window_tmp.rename(window_path)
    with preview_path.open("x") as f:
        for row in previews:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "format_version": 1,
        "purpose": "functional_smoke_test_only",
        "source": str(source),
        "source_first16_sha256": raw_hash.hexdigest(),
        "source_records": 16,
        "teacher": args.model,
        "chat_template_sha256": hashlib.sha256(
            tokenizer.chat_template.encode()
        ).hexdigest(),
        "template_kwargs": {"add_generation_prompt": False, "enable_thinking": True},
        "mask_alignment": "next_token_prediction_position",
        "block_size": 8,
        "window_length": args.length,
        "overlap_tokens": args.overlap,
        "history_policy": (
            "sliding_token_window; older context is omitted, only for smoke testing"
        ),
        "train_rows": total_windows,
        "validation_policy": POLICY,
        "validation_note": "All 16 source records train. Arrow cache appends a copy "
        "of the first "
        "training window for required validation plumbing, not held-out evaluation.",
        "full_records_jsonl": str(full_path),
        "full_records_sha256": sha256(full_path),
        "training_jsonl": str(window_path),
        "training_jsonl_sha256": sha256(window_path),
        "mask_preview": str(preview_path),
        "records": stats,
    }
    dump(manifest_path, manifest)
    print("Training JSONL:", window_path, "rows:", total_windows)


def read_training_jsonl(path, limit, train_rows=None):
    path = Path(path)
    with path.open() as f:
        rows = [json.loads(line) for line in f]
    require(rows, "Empty training JSONL")
    for i, row in enumerate(rows):
        ids, mask = row["input_ids"], row["loss_mask"]
        require(
            BLOCK_SIZE < len(ids) <= limit,
            f"Row {i}: invalid length; refusing truncation",
        )
        require(len(ids) == len(mask) == row["seq_len"], f"Row {i}: length mismatch")
        require(
            all(type(t) is int and 0 <= t < VOCAB_SIZE for t in ids),
            f"Row {i}: token ID",
        )
        require(
            all(type(m) in (int, bool) and m in (0, 1) for m in mask),
            f"Row {i}: mask",
        )
        require(
            sum(mask) and (not any(mask[-8:])),
            f"Row {i}: empty mask or unsafe boundary",
        )
        require(
            row.get("mask_alignment") == "next_token_prediction_position",
            f"Row {i}: alignment",
        )
    manifest_path = Path(str(path) + ".manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest:
        require(
            manifest["training_jsonl_sha256"] == sha256(path),
            "JSONL checksum differs from manifest",
        )
        require(
            manifest["mask_alignment"] == "next_token_prediction_position",
            "Invalid data: manifest['mask_alignment'] == 'next_token_prediction_position'",  # noqa: E501
        )
    n_train = int(train_rows) if train_rows else int(manifest["train_rows"])
    if n_train == len(rows):
        require(
            manifest.get("validation_policy") == POLICY,
            "A nonempty validation split is required",
        )
        print(
            "Smoke validation reuses the first training window; "
            "not held-out evaluation.",
            file=sys.stderr,
        )
        rows.append(copy.deepcopy(rows[0]))
    require(0 < n_train < len(rows), "Invalid train/validation split")
    ratio = n_train / len(rows)
    if int(len(rows) * ratio) != n_train:
        ratio = math.nextafter(ratio, 1.0)
    require(
        int(len(rows) * ratio) == n_train,
        "Invalid data: int(len(rows) * ratio) == n_train",
    )
    return rows, n_train, ratio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    b = subs.add_parser("build")
    b.add_argument("--source", default=str(DEFAULT_DIR / "train-00000.jsonl"))
    b.add_argument("--output-dir", default=str(DEFAULT_DIR))
    b.add_argument("--model", default=DEFAULT_MODEL)
    b.add_argument("--length", type=int, default=4096)
    b.add_argument("--overlap", type=int, default=256)
    for name in ("check", "materialize"):
        p = subs.add_parser(name)
        p.add_argument("--path", required=True)
        p.add_argument("--seq-length", required=True, type=int)
        p.add_argument("--train-rows")
        if name == "materialize":
            p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args)
        return
    rows, n_train, ratio = read_training_jsonl(
        args.path, args.seq_length, args.train_rows
    )
    if args.command == "materialize":
        from datasets import Dataset, Features, Sequence, Value  # noqa: PLC0415

        require(not Path(args.output).exists(), "Cache directory already exists")
        features = Features(
            {
                "input_ids": Sequence(Value("int64")),
                "loss_mask": Sequence(Value("int64")),
                "seq_len": Value("int64"),
            }
        )
        data = Dataset.from_list(
            [{k: row[k] for k in features} for row in rows], features=features
        )
        data.save_to_disk(args.output)
        dump(
            Path(args.output) / "split.json",
            {
                "train_rows": n_train,
                "total_rows": len(rows),
                "source_jsonl": str(Path(args.path).resolve()),
                "source_sha256": sha256(args.path),
                "train_data_ratio": ratio,
            },
        )
    print(repr(ratio))


if __name__ == "__main__":
    main()
