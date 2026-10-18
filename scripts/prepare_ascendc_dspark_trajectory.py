"""One OpenAI trajectory per DSpark Arrow row, using the teacher's own template."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


from speculators.data_generation.trajectory import (
    encode_trajectory,
    normalize,
    prediction_mask,
    sha,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--split', type=Path, required=True, help='Existing trajectory split JSONL, with shard/line/split/group_id')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-length', type=int, default=262112)
    p.add_argument('--block-size', type=int, default=8)
    p.add_argument('--overlength', choices=['error', 'skip', 'truncate'], default='error')
    p.add_argument('--limit', type=int, help='Diagnostic subset: first N source lines')
    a = p.parse_args()
    if a.block_size < 1 or a.max_length <= a.block_size or (a.limit is not None and a.limit < 1):
        p.error('Invalid length/block-size/limit')
    from transformers import AutoTokenizer
    from datasets import Dataset, Features, Sequence, Value, concatenate_datasets
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError('Fast tokenizer required for exact character boundaries')
    splits = {}
    groups = {'train': set(), 'validation': set()}
    for line in a.split.read_text().splitlines():
        meta = json.loads(line)
        if meta['shard'] == a.input.name:
            if meta['line'] in splits or meta['split'] not in groups:
                raise ValueError('Invalid/duplicate split entry')
            splits[meta['line']] = meta
            groups[meta['split']].add(meta['group_id'])
    if groups['train'] & groups['validation']:
        raise ValueError('Train/validation group leakage')
    a.output.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    paths = {s: a.output / (s + '.jsonl') for s in groups}
    handles = {s: path.open('w') for s, path in paths.items()}
    try:
        with a.input.open() as source, (a.output/'audit.jsonl').open('w') as audit:
            for n, line in enumerate(source, 1):
                if a.limit is not None and n > a.limit:
                    break
                meta = splits[n]
                try:
                    row, info, rendered, targets = encode_trajectory(json.loads(line), tokenizer, a.max_length, a.block_size, a.overlength)
                except Exception as e:
                    raise ValueError(f'Source line {n}: {e}') from e
                info.update(source_line=n, split=meta['split'], group_id=meta['group_id'])
                audit.write(json.dumps(info, ensure_ascii=False)+'\n')
                counts['source_trajectories'] += 1
                counts[info['status']] += 1
                counts['overlength'] += int(info['overlength'])
                if row is not None:
                    handles[meta['split']].write(json.dumps(row)+'\n')
                    counts[meta['split']] += 1
                    counts['input_tokens'] += row['seq_len']
                    counts['prediction_candidates'] += sum(row['loss_mask'])
                if n == 1:
                    (a.output/'first_trajectory.txt').write_text(rendered)
                    (a.output/'first_trajectory_target_mask.json').write_text(json.dumps(targets))
    finally:
        for h in handles.values(): h.close()
    features = Features({'input_ids': Sequence(Value('int64')), 'loss_mask': Sequence(Value('int64')), 'seq_len': Value('int64')})
    parts = [Dataset.from_json(str(paths[s]), features=features) for s in ('train','validation') if counts[s]]
    if not parts:
        raise ValueError('No usable trajectories')
    combined = concatenate_datasets(parts)
    combined.save_to_disk(str(a.output/'data_arrow'))
    ratio = counts['train']/len(combined)
    if 0 < ratio < 1: ratio = math.nextafter(ratio, 1.0)
    assert int(len(combined)*ratio) == counts['train']
    split_info = dict(train_rows=counts['train'], validation_rows=counts['validation'], train_ratio=ratio)
    (a.output/'data_arrow'/'split.json').write_text(json.dumps(split_info, indent=2))
    manifest = dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                    counts=dict(counts), input_sha256=sha(a.input), split_sha256=sha(a.split),
                    script_sha256=sha(__file__), template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                    preserve_thinking=True, assistant_header_supervised=False,
                    assistant_end_and_newline_supervised=True, **split_info)
    (a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
