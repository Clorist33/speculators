#!/usr/bin/env python3
"""Create cumulative chat-history samples from the first N AscendC trajectories."""

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

DEFAULT_SOURCE = Path(
    "/home/t00972278/dataset/ascendc-kernelgen-data/dataset/agent_dataset/train-00006.jsonl"
)
DEFAULT_OUTPUT = Path(
    "/home/t00972278/dataset/ascendc-kernelgen-data/dataset/agent_dataset/"
    "rollout_data/train00006_first10_trajectories_histories.json"
)


def clean_message(message):
    # Match the existing first-trajectory history file: keep nonempty optional fields.
    return {
        key: value
        for key, value in message.items()
        if key in ("role", "content", "tool_calls", "tool_call_id")
        and (key in ("role", "content") or value)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trajectories", type=int, default=10)
    parser.add_argument(
        "--source-lines", type=int, nargs="+",
        help="1-based source line numbers; overrides --trajectories",
    )
    args = parser.parse_args()
    if args.trajectories < 1:
        parser.error("--trajectories must be positive")
    selected_lines = (
        args.source_lines if args.source_lines is not None
        else list(range(1, args.trajectories + 1))
    )
    if selected_lines != sorted(set(selected_lines)) or selected_lines[0] < 1:
        parser.error("--source-lines must be positive, unique, and ascending")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    samples = 0
    roles = Counter()
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=args.output.parent,
            prefix=args.output.name + ".", suffix=".tmp", delete=False,
        ) as target:
            temp_path = Path(target.name)
            target.write("[\n")
            with args.source.open(encoding="utf-8") as source:
                current_line = 0
                for position, source_line in enumerate(selected_lines, 1):
                    while current_line < source_line:
                        line = source.readline()
                        if not line:
                            raise ValueError(f"source has fewer than {source_line} trajectories")
                        current_line += 1
                    messages = json.loads(line)["messages"]
                    if not isinstance(messages, list) or len(messages) < 2:
                        raise ValueError(f"invalid messages on source line {source_line}")
                    if messages[0].get("role") != "system":
                        raise ValueError(f"expected system first on source line {source_line}")
                    history = []
                    for end_index, message in enumerate(messages):
                        history.append(clean_message(message))
                        if end_index == 0:
                            continue
                        sample = {
                            "sample_index": samples,
                            "end_message_index": end_index,
                            "end_role": message["role"],
                            "history_messages": history,
                            "source_file": str(args.source),
                            "source_line": source_line,
                        }
                        if samples:
                            target.write(",\n")
                        json.dump(sample, target, ensure_ascii=False)
                        samples += 1
                        roles[message["role"]] += 1
                    print(
                        f"trajectory {position}/{len(selected_lines)} "
                        f"(source line {source_line}): {len(messages) - 1} samples",
                        flush=True,
                    )
            target.write("\n]\n")
        os.replace(temp_path, args.output)
    except BaseException:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    print(f"output={args.output} samples={samples} end_roles={dict(roles)}")


if __name__ == "__main__":
    main()
