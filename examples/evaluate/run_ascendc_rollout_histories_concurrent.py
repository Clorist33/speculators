#!/usr/bin/env python3
"""Send cumulative AscendC histories concurrently to teacher or DSpark."""

import argparse
import json
import math
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

COUNTER = re.compile(
    r'^vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total'
    r'(?:\{[^}]*\})?\s+([0-9.eE+-]+)$'
)
POSITION = re.compile(
    r'^vllm:spec_decode_num_accepted_tokens_per_pos_total'
    r'\{([^}]*)\}\s+([0-9.eE+-]+)$'
)


def get_json(url, timeout=15):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def fetch_metrics(base):
    with urllib.request.urlopen(base + "/metrics", timeout=15) as response:
        return response.read().decode()


def parse_metrics(payload):
    counters = defaultdict(int)
    positions = defaultdict(int)
    for line in payload.splitlines():
        if match := COUNTER.match(line):
            counters[match.group(1)] += int(float(match.group(2)))
        elif match := POSITION.match(line):
            labels, value = match.groups()
            if index := re.search(r'position="(\d+)"', labels):
                positions[int(index.group(1))] += int(float(value))
    if not all(key in counters for key in ("drafts", "draft_tokens", "accepted_tokens")):
        raise RuntimeError(f"DSpark counters missing from /metrics: {dict(counters)}")
    return counters, positions


def percentile(values, percentage):
    ordered = sorted(values)
    return ordered[math.ceil(percentage * len(ordered)) - 1] if ordered else None


def send_one(sample, base, model, max_tokens, timeout):
    row = {
        "sample_index": sample["sample_index"],
        "source_line": sample["source_line"],
        "end_message_index": sample["end_message_index"],
        "end_role": sample["end_role"],
        "history_message_count": len(sample["history_messages"]),
    }
    request = {
        "model": model,
        "messages": sample["history_messages"],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    start = time.monotonic()
    try:
        body = json.dumps(request, ensure_ascii=False).encode()
        http_request = urllib.request.Request(
            base + "/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(http_request, timeout=timeout) as response:
            answer = json.load(response)
        row.update({
            "status": "ok",
            "usage": answer.get("usage", {}),
            "finish_reason": answer["choices"][0].get("finish_reason"),
            "generated_message": answer["choices"][0]["message"],
        })
        row["completion_tokens"] = row["usage"].get("completion_tokens", 0)
    except urllib.error.HTTPError as error:
        row.update({
            "status": "http_error",
            "http_status": error.code,
            "error": error.read(1000).decode(errors="replace"),
        })
    except Exception as error:
        row.update({"status": "error", "error": str(error)})
    row["elapsed_seconds"] = round(time.monotonic() - start, 3)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:9092")
    parser.add_argument("--model", default="qwen36")
    parser.add_argument("--mode", choices=("dspark", "teacher"), default="dspark")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.concurrency < 1 or args.max_tokens < 1:
        parser.error("--concurrency and --max-tokens must be positive")
    base = args.base_url.rstrip("/")
    samples = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(samples, list) or not samples:
        raise ValueError("Expected a nonempty JSON array of history samples")
    if [sample["sample_index"] for sample in samples] != list(range(len(samples))):
        raise ValueError("Sample indices are not contiguous")
    models = get_json(base + "/v1/models")
    if args.model not in [model["id"] for model in models["data"]]:
        raise RuntimeError(f"Model {args.model!r} is not served at {base}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: args.output_dir / name for name in (
        "results.jsonl", "summary.json", "metrics_before.prom", "metrics_after.prom"
    )}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Existing results or metrics would be overwritten")
    if args.mode == "dspark":
        before_text = fetch_metrics(base)
        before, before_positions = parse_metrics(before_text)
        paths["metrics_before.prom"].write_text(before_text, encoding="utf-8")

    rows = []
    batch_start = time.monotonic()
    with paths["results.jsonl"].open("x", encoding="utf-8") as result_file:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [
                executor.submit(send_one, sample, base, args.model, args.max_tokens, args.timeout)
                for sample in samples
            ]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                result_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                result_file.flush()
                if len(rows) % 10 == 0 or row["status"] != "ok" or len(rows) == len(samples):
                    print(
                        f"{len(rows)}/{len(samples)} sample={row['sample_index']} "
                        f"status={row['status']} elapsed={row['elapsed_seconds']}s",
                        flush=True,
                    )
    wall_seconds = round(time.monotonic() - batch_start, 3)
    if args.mode == "dspark":
        after_text = fetch_metrics(base)
        paths["metrics_after.prom"].write_text(after_text, encoding="utf-8")
        after, after_positions = parse_metrics(after_text)
        counters = {key: after[key] - before[key] for key in (
            "drafts", "draft_tokens", "accepted_tokens"
        )}
        positions = [after_positions[i] - before_positions.get(i, 0) for i in range(8)]
        if sum(positions) != counters["accepted_tokens"]:
            raise RuntimeError("Per-position accepted counts disagree with total accepted count")
    rows.sort(key=lambda row: row["sample_index"])
    ok = [row for row in rows if row["status"] == "ok"]
    generated = sum(row["completion_tokens"] for row in ok)
    times = [row["elapsed_seconds"] for row in ok]
    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source_line"]].append(row)
    summary = {
        "config": {
            "input": str(args.input), "base_url": base, "model": args.model,
            "mode": args.mode, "max_tokens": args.max_tokens, "temperature": 0,
            "concurrency": args.concurrency,
        },
        "overall": {
            "samples": len(rows), "successful": len(ok),
            "statuses": dict(Counter(row["status"] for row in rows)),
            "completion_tokens": generated,
            "length_finishes": sum(row.get("finish_reason") == "length" for row in ok),
            "wall_seconds": wall_seconds,
            "total_request_seconds": round(sum(times), 3),
            "mean_request_seconds": round(sum(times) / len(times), 3) if times else None,
            "p95_request_seconds": percentile(times, 0.95),
            "completion_tokens_per_wall_second": generated / wall_seconds,
        },
        "by_source_line": {
            str(line): {
                "samples": len(items),
                "successful": sum(row["status"] == "ok" for row in items),
                "completion_tokens": sum(row.get("completion_tokens", 0) for row in items),
            }
            for line, items in sorted(by_source.items())
        },
    }
    if args.mode == "dspark":
        summary["overall"].update({
            "drafts": counters["drafts"],
            "draft_tokens": counters["draft_tokens"],
            "accepted_tokens": counters["accepted_tokens"],
            "acceptance_rate": (
                counters["accepted_tokens"] / counters["draft_tokens"]
                if counters["draft_tokens"] else None
            ),
            "mean_acceptance_length_including_bonus": (
                1 + counters["accepted_tokens"] / counters["drafts"]
                if counters["drafts"] else None
            ),
            "accepted_tokens_per_position": positions,
        })
    paths["summary.json"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if len(ok) != len(rows):
        raise SystemExit(f"Only {len(ok)}/{len(rows)} requests succeeded; inspect results.jsonl")
    print(json.dumps(summary["overall"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
