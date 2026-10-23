#!/usr/bin/env python3
"""Replay cumulative history samples against a teacher or DSpark vLLM server."""

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_INPUT = Path(
    "/home/t00972278/dataset/ascendc-kernelgen-data/dataset/agent_dataset/"
    "rollout_data/train00006_first_trajectory_histories.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/t00972278/Dspark_sft_docs/train00006第一条轨迹推理日志"
)
METRIC = re.compile(
    r'^vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total(?:\{[^}]*\})?\s+([0-9.eE+-]+)$'
)


def get_json(url, timeout=15):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def spec_metrics(base):
    with urllib.request.urlopen(base + "/metrics", timeout=15) as response:
        lines = response.read().decode().splitlines()
    totals = defaultdict(float)
    for line in lines:
        match = METRIC.match(line)
        if match:
            totals[match.group(1)] += float(match.group(2))
    required = {"drafts", "draft_tokens", "accepted_tokens"}
    if not required.issubset(totals):
        raise RuntimeError(f"DSpark counters missing from /metrics: {dict(totals)}")
    return {key: int(totals[key]) for key in required}


def aggregate(rows, mode):
    ok = [row for row in rows if row["status"] == "ok"]
    generated = sum(row["completion_tokens"] for row in ok)
    elapsed = sum(row["elapsed_seconds"] for row in ok)
    result = {
        "samples": len(rows),
        "successful": len(ok),
        "statuses": dict(Counter(row["status"] for row in rows)),
        "completion_tokens": generated,
        "total_request_seconds": elapsed,
        "mean_request_seconds": elapsed / len(ok) if ok else None,
        "completion_tokens_per_request_second_including_prefill": (
            generated / elapsed if elapsed else None
        ),
    }
    if mode == "dspark":
        accepted = sum(row["accepted_tokens"] for row in ok)
        drafted = sum(row["draft_tokens"] for row in ok)
        drafts = sum(row["drafts"] for row in ok)
        result.update({
            "accepted_tokens": accepted,
            "draft_tokens": drafted,
            "drafts": drafts,
            "acceptance_rate": accepted / drafted if drafted else None,
            "mean_acceptance_length_including_bonus": (
                1 + accepted / drafts if drafts else None
            ),
        })
    return result


def summary(rows, config, mode):
    by_role = defaultdict(list)
    for row in rows:
        by_role[row["end_role"]].append(row)
    return {
        "config": config,
        "overall": aggregate(rows, mode),
        "by_end_role": {role: aggregate(items, mode) for role, items in by_role.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-url", default="http://127.0.0.1:9093")
    parser.add_argument("--model", default="qwen36")
    parser.add_argument("--mode", choices=("dspark", "teacher"), default="dspark")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as source:
        samples = json.load(source)
    if not isinstance(samples, list) or len(samples) != 153:
        raise ValueError("Expected 153 cumulative history samples in a JSON array")
    if any(sample["sample_index"] != i for i, sample in enumerate(samples)):
        raise ValueError("Sample indices are not contiguous")

    base = args.base_url.rstrip("/")
    models = get_json(base + "/v1/models")
    if args.model not in [model["id"] for model in models["data"]]:
        raise RuntimeError(f"Model {args.model!r} is not served at {base}")
    if args.mode == "dspark":
        spec_metrics(base)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    summary_path = args.output_dir / "summary.json"
    if results_path.exists() or summary_path.exists():
        raise FileExistsError("Existing results would be overwritten")
    config = {
        "input": str(args.input),
        "base_url": base,
        "model": args.model,
        "mode": args.mode,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "request_order": "sample_index ascending, one request at a time",
    }

    rows = []
    with results_path.open("x", encoding="utf-8") as results:
        for sample in samples:
            messages = sample["history_messages"]
            request = {
                "model": args.model,
                "messages": messages,
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "stream": False,
            }
            before = spec_metrics(base) if args.mode == "dspark" else None
            row = {
                "sample_index": sample["sample_index"],
                "end_message_index": sample["end_message_index"],
                "end_role": sample["end_role"],
                "history_message_count": len(messages),
            }
            start = time.monotonic()
            try:
                body = json.dumps(request, ensure_ascii=False).encode()
                http_request = urllib.request.Request(
                    base + "/v1/chat/completions",
                    data=body,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(http_request, timeout=args.timeout) as response:
                    answer = json.load(response)
                row["status"] = "ok"
                row["usage"] = answer.get("usage", {})
                row["completion_tokens"] = row["usage"].get("completion_tokens", 0)
                row["finish_reason"] = answer["choices"][0].get("finish_reason")
                row["generated_message"] = answer["choices"][0]["message"]
            except urllib.error.HTTPError as error:
                row["status"] = "http_error"
                row["http_status"] = error.code
                row["error"] = error.read(1000).decode(errors="replace")
            except Exception as error:
                row["status"] = "error"
                row["error"] = str(error)
            row["elapsed_seconds"] = round(time.monotonic() - start, 3)
            if args.mode == "dspark":
                try:
                    after = spec_metrics(base)
                    for key in ("drafts", "draft_tokens", "accepted_tokens"):
                        row[key] = after[key] - before[key]
                    row["acceptance_rate"] = (
                        row["accepted_tokens"] / row["draft_tokens"]
                        if row["draft_tokens"] else None
                    )
                except Exception as error:
                    row["metric_error"] = str(error)
                    row["drafts"] = row["draft_tokens"] = row["accepted_tokens"] = 0
            rows.append(row)
            results.write(json.dumps(row, ensure_ascii=False) + "\n")
            results.flush()
            summary_path.write_text(
                json.dumps(summary(rows, config, args.mode), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            detail = (
                f" accepted={row['accepted_tokens']}/{row['draft_tokens']}"
                if args.mode == "dspark" else ""
            )
            print(
                f"{len(rows)}/{len(samples)} end_role={row['end_role']} "
                f"status={row['status']}{detail} elapsed={row['elapsed_seconds']}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
