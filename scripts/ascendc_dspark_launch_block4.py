"""AscendC DSpark block-size=4 实验专用启动辅助。"""
import contextlib
import os
import sys
from pathlib import Path

SOURCE_BLOCK_SIZE = 8
TRAIN_BLOCK_SIZE = 4

def check_vllm():
    import os

    def positive(name):
        raw = os.environ[name]
        if not raw.isdigit() or int(raw) <= 0:
            raise SystemExit(f'{name} must be a positive integer')
        return int(raw)

    limit = positive('VLLM_MAX_MODEL_LEN')
    batch = positive('VLLM_MAX_NUM_BATCHED_TOKENS')
    seqs = positive('VLLM_MAX_NUM_SEQS')
    for name in ('VLLM_ENABLE_CHUNKED_PREFILL', 'VLLM_ENFORCE_EAGER'):
        if os.environ[name] not in ('0', '1'):
            raise SystemExit(f'{name} must be 0 or 1')
    try:
        ratio = float(os.environ['VLLM_GPU_MEMORY_UTILIZATION'])
    except ValueError:
        raise SystemExit('VLLM_GPU_MEMORY_UTILIZATION must be in (0,1]')
    if not 0 < ratio <= 1:
        raise SystemExit('VLLM_GPU_MEMORY_UTILIZATION must be in (0,1]')
    if os.environ['VLLM_KV_CACHE_MEMORY_BYTES']:
        positive('VLLM_KV_CACHE_MEMORY_BYTES')
    if limit < positive('SEQ_LENGTH') + 1 or limit % 32:
        raise SystemExit('VLLM_MAX_MODEL_LEN must cover SEQ_LENGTH+1 and be a multiple of 32')
    if os.environ['VLLM_ENABLE_CHUNKED_PREFILL'] == '0' and batch < limit:
        raise SystemExit('Without chunked prefill, VLLM_MAX_NUM_BATCHED_TOKENS must be >= VLLM_MAX_MODEL_LEN')
    if batch < seqs:
        raise SystemExit('VLLM_MAX_NUM_BATCHED_TOKENS must be >= VLLM_MAX_NUM_SEQS')


def check_config():
    import ast
    import json
    import os
    from pathlib import Path

    errors = []
    def read_config(directory):
        path = Path(directory) / 'config.json'
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            errors.append(f'{path}: {exc}')
            return {}

    teacher = read_config(os.environ['MODEL'])
    draft = read_config(os.environ['DRAFT_MODEL'])
    text = teacher.get('text_config', teacher)
    expected = {'speculators_model_type': 'dspark', 'draft_vocab_size': 248320,
                'aux_hidden_state_layer_ids': [2, 10, 20, 30, 37],
                'block_size': SOURCE_BLOCK_SIZE,
                'markov_rank': 256, 'markov_head_type': 'vanilla',
                'enable_confidence_head': True, 'confidence_head_with_markov': True,
                'sample_from_anchor': True, 'mask_token_id': 248077}
    for key, value in expected.items():
        if draft.get(key) != value:
            errors.append(f'draft {key}: expected {value!r}, got {draft.get(key)!r}')
    if os.environ.get('BLOCK_SIZE') != str(TRAIN_BLOCK_SIZE):
        errors.append(f'BLOCK_SIZE must be {TRAIN_BLOCK_SIZE} for this launcher')
    layer = draft.get('transformer_layer_config', {})
    for key, value in {'num_hidden_layers': 5, 'hidden_size': 2048}.items():
        if layer.get(key) != value:
            errors.append(f'draft transformer {key} must remain {value}')
    for key, value in {'num_hidden_layers': 40, 'hidden_size': 2048, 'vocab_size': 248320}.items():
        if text.get(key) != value:
            errors.append(f'teacher text config {key}: expected {value}, got {text.get(key)}')
    if not (Path(os.environ['DRAFT_MODEL']) / 'model.safetensors').is_file():
        errors.append('Missing pretrained model.safetensors; config-only initialization is forbidden')
    if not (Path(os.environ['MODEL']) / 'model.safetensors.index.json').is_file():
        errors.append('Missing teacher sharded weight index')
    raw = Path(os.environ['RAW_DATA_DIR'])
    print(f'Raw operator data: {raw} ({len(list(raw.glob("train-[0-9][0-9][0-9][0-9][0-9].jsonl")))} JSONL shards)')
    if (os.environ.get('DATA_FORMAT') == 'trajectory-jsonl' and
            not list(raw.glob('train-[0-9][0-9][0-9][0-9][0-9].jsonl'))):
        errors.append('No raw operator JSONL shards found')
    data = Path(os.environ['DATA_PATH'])
    if os.environ.get('DATA_FORMAT', 'arrow') == 'trajectory-jsonl':
        if os.environ.get('TRAJECTORY_SHARDS_CSV'):
            if not data.is_dir():
                errors.append(f'Raw trajectory directory missing at {data}')
        elif not data.is_file():
            errors.append(f'Raw trajectory JSONL missing at {data}')
        split = os.environ.get('TRAJECTORY_SPLIT_PATH', '')
        if not split or not Path(split).is_file():
            errors.append('TRAJECTORY_SPLIT_PATH must point to the source-line group split')
        quarantine = os.environ.get('TRAJECTORY_QUARANTINE_PATH', '')
        if quarantine and not Path(quarantine).is_file():
            errors.append(f'Trajectory quarantine file missing at {quarantine}')
        if os.environ.get('TRAJECTORY_OVERLENGTH', 'error') not in ('error', 'skip', 'truncate'):
            errors.append('Invalid TRAJECTORY_OVERLENGTH')
    elif os.environ.get('DATA_FORMAT') == 'trajectory-arrow':
        for name in ('state.json', 'dataset_info.json', 'preprocess_manifest.json'):
            if not (data / name).is_file():
                errors.append(f'Prepared trajectory cache missing {name} at {data}')
    else:
        if not (data / 'state.json').is_file() or not (data / 'dataset_info.json').is_file():
            errors.append(f'Prepared trajectory Arrow Dataset missing at {data}')
        if not os.environ['TRAIN_ROWS'] and not (data / 'split.json').is_file():
            errors.append('Set TRAIN_ROWS from the group split or supply DATA_PATH/split.json')
    if Path(os.environ['OUTPUT_DIR']).exists():
        errors.append('OUTPUT_DIR already exists; use a new run directory (old optimizer states are not resumed)')
    for key in ('SEQ_LENGTH', 'MAX_ANCHORS'):
        try:
            assert int(os.environ[key]) > 8
        except (ValueError, AssertionError):
            errors.append(f'{key} must be an integer greater than 8')
    def devices(key):
        ids = os.environ[key].split(',')
        if not ids or any(not s.isdigit() for s in ids) or len(set(ids)) != len(ids):
            errors.append(f'{key} must contain unique, comma-separated numeric NPU IDs')
        return set(ids)
    if os.environ['LAUNCH_TEACHER'] == '1':
        limit = int(os.environ['VLLM_MAX_MODEL_LEN'])
        supported = text.get('max_position_embeddings')
        if supported is not None and limit > int(supported):
            errors.append(f'Teacher limit {limit} exceeds configured max_position_embeddings={supported}')
        cache = os.environ['VLLM_KV_CACHE_MEMORY_BYTES']
        print('Teacher memory:', f'explicit cache {cache} bytes/NPU (utilization does not size cache)' if cache
              else f'automatic cache, gpu-memory-utilization={os.environ["VLLM_GPU_MEMORY_UTILIZATION"]}')
        print('Teacher limits:', limit, 'batched tokens:', os.environ['VLLM_MAX_NUM_BATCHED_TOKENS'],
              'max sequences:', os.environ['VLLM_MAX_NUM_SEQS'],
              'chunked prefill:', os.environ['VLLM_ENABLE_CHUNKED_PREFILL'])
    train_ids = devices('TRAIN_NPUS')
    if os.environ['LAUNCH_TEACHER'] == '1' and train_ids & devices('VLLM_NPUS'):
        errors.append('Teacher and trainer NPU assignments overlap')

    # Fail closed on the known final-norm bug without importing the model stack.
    # This check must be revised if the registry implementation is refactored.
    source = Path(os.environ['REPO_ROOT']) / 'src/speculators/models/utils.py'
    registry = None
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                t.id == 'GEMMA_STYLE_FINAL_NORM_MODEL_TYPES' for t in node.targets):
            if isinstance(node.value, ast.Call) and node.value.args:
                registry = set(ast.literal_eval(node.value.args[0]))
    if text.get('model_type') == 'qwen3_5_moe_text' and (
            registry is None or 'qwen3_5_moe_text' not in registry):
        errors.append('qwen3_5_moe_text is absent from the 1+w final-norm registry. '
                      'Fix/verify teacher-logit reconstruction before training; this launcher does not patch it')
    print('Teacher:', os.environ['MODEL'])
    print('Original draft teacher:', draft.get('speculators_config', {}).get('verifier', {}).get('name_or_path'))
    print('Draft: 5 layers, full vocabulary, aux=[2,10,20,30,37]; '
          'run-local config will use block size 4 and bind the new teacher')
    if errors:
        for error in errors:
            print('BLOCKED:', error)
        raise SystemExit(1)
    print('Static configuration checks passed; numerical/NPU validation is still required.')


def check_arrow():
    import json
    import math
    import os
    from pathlib import Path
    from datasets import load_from_disk

    path = Path(os.environ['DATA_PATH'])
    data = load_from_disk(str(path)).with_format(None)
    assert {'input_ids', 'loss_mask', 'seq_len'} <= set(data.column_names), 'Missing Arrow columns'
    n = len(data)
    train_rows = int(os.environ['TRAIN_ROWS'] or json.loads((path / 'split.json').read_text())['train_rows'])
    assert 0 < train_rows < n, 'Both train and validation must be nonempty'
    ratio = train_rows / n
    if int(n * ratio) != train_rows:
        ratio = math.nextafter(ratio, 1.0)
    assert int(n * ratio) == train_rows, 'Cannot represent requested split'
    print(repr(ratio))


def check_trajectory_cache():
    import os
    from speculators.data_generation.trajectory_cache import load_trajectory_cache

    dataset, ratio, manifest = load_trajectory_cache(
        os.environ['DATA_PATH'],
        model_path=os.environ['MODEL'],
        max_length=int(os.environ['SEQ_LENGTH']),
        block_size=TRAIN_BLOCK_SIZE,
        overlength=os.environ['TRAJECTORY_OVERLENGTH'],
        shard_names=os.environ['TRAJECTORY_SHARDS_CSV'].split(','),
        split_path=os.environ['TRAJECTORY_SPLIT_PATH'],
        quarantine_path=os.environ['TRAJECTORY_QUARANTINE_PATH'],
    )
    print(f"Trajectory cache: {len(dataset)} rows, "
          f"train={manifest['train_rows']}, validation={manifest['validation_rows']}",
          file=sys.stderr)
    print(repr(ratio))


def check_data():
    if os.environ.get('DATA_FORMAT', 'arrow') == 'trajectory-jsonl':
        # The definitive ratio is recomputed after overlength filtering in train.
        # This check never tokenizes or creates a processed dataset on disk.
        from speculators.data_generation.trajectory import inspect_trajectory_inputs
        shard_csv = os.environ.get('TRAJECTORY_SHARDS_CSV', '')
        sources, splits, quarantined = inspect_trajectory_inputs(
            os.environ['DATA_PATH'],
            os.environ['TRAJECTORY_SPLIT_PATH'],
            shard_names=shard_csv.split(',') if shard_csv else None,
            quarantine_path=os.environ.get('TRAJECTORY_QUARANTINE_PATH') or None,
            check_lines=True,
        )
        assignments = [m for source in sources for m in splits[source.name].values()]
        ntrain = sum(m['split'] == 'train' for m in assignments)
        if not 0 < ntrain < len(assignments):
            raise SystemExit('Both trajectory splits must be nonempty')
        print(f'Trajectory shards: {len(sources)}, assigned rows: {len(assignments)}, '
              f'quarantined rows: {len(quarantined)}', file=sys.stderr)
        print(repr(ntrain / len(assignments)))
    elif os.environ.get('DATA_FORMAT') == 'trajectory-arrow':
        check_trajectory_cache()
    else:
        check_arrow()


def stage_draft():
    import json
    import os
    from pathlib import Path
    src = Path(os.environ['DRAFT_MODEL']).resolve()
    dst = Path(os.environ['INIT_DIR'])
    config = json.loads((src / 'config.json').read_text())
    if os.environ.get('BLOCK_SIZE') != str(TRAIN_BLOCK_SIZE):
        raise ValueError(f'BLOCK_SIZE must be {TRAIN_BLOCK_SIZE} for this launcher')
    if config.get('block_size') != SOURCE_BLOCK_SIZE:
        raise ValueError(f'Original draft block_size must be {SOURCE_BLOCK_SIZE}')
    methods = config['speculators_config']['proposal_methods']
    if not methods or any(method.get('speculative_tokens') != SOURCE_BLOCK_SIZE for method in methods):
        raise ValueError(f'Original draft proposal length must be {SOURCE_BLOCK_SIZE}')
    config['block_size'] = TRAIN_BLOCK_SIZE
    for method in methods:
        method['speculative_tokens'] = TRAIN_BLOCK_SIZE
    config['speculators_config']['verifier']['name_or_path'] = str(Path(os.environ['MODEL']).resolve())
    # Only the read-only weight file is shared; never import old optimizer/run state.
    (dst / 'model.safetensors').symlink_to(src / 'model.safetensors')
    (dst / 'config.json').write_text(json.dumps(config, indent=2) + '\n')



if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('check', 'prepare'):
        raise SystemExit('用法: python scripts/ascendc_dspark_launch.py check|prepare（参数由主脚本环境传入）')
    if sys.argv[1] == 'check':
        with contextlib.redirect_stdout(sys.stderr):
            if os.environ['LAUNCH_TEACHER'] == '1':
                check_vllm()
            check_config()
        check_data()
    else:
        check_data()
        stage_draft()
