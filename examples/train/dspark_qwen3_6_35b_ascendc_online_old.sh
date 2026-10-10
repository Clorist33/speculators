#!/usr/bin/env bash
# Continue an existing DSpark on Ascend against the operator-tuned Qwen3.6.
# Raw trajectory preprocessing is separate. Pretokenized JSONL is validated
# and materialized as a run-local Arrow cache without changing masks.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash examples/train/dspark_qwen3_6_35b_ascendc_online.sh [check|smoke|train|print]

  check  Read-only configuration/data preflight (default); no NPU workloads.
  print  Print smoke launch commands without validation, writes or execution.
  smoke  Run preflight, launch teacher, then train for at most 100 steps.
  train  Run preflight, launch teacher, then train one epoch (default).

Defaults:
  MODEL=/home/docker/Qwen3.6-35B-A3B-agentical-ascendc-hf-4t-bf16
  DRAFT_MODEL=/home/weight/Qwen3.6-35B-A3B-Dspark
  RAW_DATA_DIR=/home/t00972278/dataset/ascendc-kernelgen-data/dataset/agent_dataset
  DATA_PATH=<RAW_DATA_DIR>/train-00000.first16.windows4096.masked.jsonl

DATA_PATH accepts pretokenized JSONL or a datasets.save_to_disk Dataset.
Required columns: input_ids, loss_mask, seq_len. No truncation or mask shift.
For JSONL, check is read-only; smoke/train create OUTPUT_DIR/data_arrow and
pass that cache to the trainer (which only supports Arrow).
The default first16 JSONL keeps ALL 16 source records in training. Its manifest
explicitly requests a copy of the first training window for validation plumbing:
validation numbers are NOT held-out evaluation. Use a proper group split for
large-scale training. Arrow overrides still require TRAIN_ROWS or split.json.

Environment overrides:
  DATA_PATH (JSONL or Arrow), TRAIN_ROWS, OUTPUT_DIR (must not already exist), PYTHON
  VLLM_NPUS=0,1 TRAIN_NPUS=2 VLLM_PORT=8890
  VLLM_ENDPOINT=http://host:port/v1  Reuse a configured extractor; do not launch
                                  or stop it. Verify its teacher/layers yourself.
  SEQ_LENGTH=4096 (smoke/check/print), 16384 (train)
  MAX_ANCHORS=32 (smoke/check/print), 64 (train)
  MAX_STEPS=100 (smoke), unset (train); EPOCHS=1 LR=1e-5
  FSDP_SHARD=0; VLLM_STARTUP_TIMEOUT=1800; NUM_WORKERS=2

Teacher vLLM overrides (only used when VLLM_ENDPOINT is unset):
  VLLM_MAX_MODEL_LEN=round_up(SEQ_LENGTH+1,32)
  VLLM_GPU_MEMORY_UTILIZATION=0.90  Total memory budget in automatic cache mode.
  VLLM_KV_CACHE_MEMORY_BYTES=      Empty: automatic; positive bytes: explicit
                                  cache budget PER NPU, overrides utilization.
  VLLM_ENABLE_CHUNKED_PREFILL=0    1: experimental for this extractor; validate
                                  complete hidden-state/token alignment first.
  VLLM_MAX_NUM_BATCHED_TOKENS=     Full prefill: defaults to model limit;
                                  chunked prefill: defaults to 16384.
  VLLM_MAX_NUM_SEQS=1; VLLM_ENFORCE_EAGER=1
  Prefix caching and async scheduling remain explicitly disabled.
  TP is the number of IDs in VLLM_NPUS; DP=1; API/renderer workers=1.
  safetensors-load-strategy=prefetch.

Reproduce the successful 64k capacity settings (supply prepared DATA_PATH):
  VLLM_NPUS=0,1,2,3 TRAIN_NPUS=4,5,6,7 SEQ_LENGTH=65536 \
    VLLM_KV_CACHE_MEMORY_BYTES=25769803776 MAX_ANCHORS=16 NUM_WORKERS=1 \
    MAX_STEPS=1 DATA_PATH=/path/to/prepared_arrow bash "$0" smoke
  Do not assume these settings cover 128k/256k or long-running stability.

Example, after preprocessing and correctness fixes:
  TRAIN_ROWS=120 DATA_PATH=/path/to/validated_smoke_arrow \
    VLLM_NPUS=0,1 TRAIN_NPUS=2 bash examples/train/dspark_qwen3_6_35b_ascendc_online.sh smoke

A successful static preflight does not certify teacher-logit equivalence or
NPU kernel/serving correctness. Continuous-request Ascend failures found during
online validation are still unresolved; data checks do not certify that fix.
HELP
}
MODE="${1:-check}"
case "$MODE" in
    -h|--help) usage; exit 0 ;;
    check|smoke|train|print) ;;
    *) usage >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo 'Only one mode argument is supported.' >&2; exit 2; }
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
MODEL="${MODEL:-/home/docker/Qwen3.6-35B-A3B-agentical-ascendc-hf-4t-bf16}"
DRAFT_MODEL="${DRAFT_MODEL:-/home/weight/Qwen3.6-35B-A3B-Dspark}"
RAW_DATA_DIR="${RAW_DATA_DIR:-/home/t00972278/dataset/ascendc-kernelgen-data/dataset/agent_dataset}"
if [[ "$MODE" == train ]]; then
    SEQ_LENGTH="${SEQ_LENGTH:-16384}"
    MAX_ANCHORS="${MAX_ANCHORS:-64}"
else
    SEQ_LENGTH="${SEQ_LENGTH:-4096}"
    MAX_ANCHORS="${MAX_ANCHORS:-32}"
fi
MAX_STEPS="${MAX_STEPS:-}"
[[ "$MODE" != smoke ]] || MAX_STEPS="${MAX_STEPS:-100}"
[[ "$MODE" != print ]] || MAX_STEPS="${MAX_STEPS:-100}"
DATA_PATH="${DATA_PATH:-$RAW_DATA_DIR/train-00000.first16.windows4096.masked.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/output/dspark_qwen3_6_35b_ascendc/${MODE}_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
INIT_DIR="$OUTPUT_DIR/initial_draft"
JSONL_HELPER="$REPO_ROOT/scripts/prepare_ascendc_dspark_smoke.py"
TRAIN_DATA_PATH="$DATA_PATH"
[[ "$DATA_PATH" != *.jsonl ]] || TRAIN_DATA_PATH="$OUTPUT_DIR/data_arrow"
VLLM_NPUS="${VLLM_NPUS:-0,1}"
TRAIN_NPUS="${TRAIN_NPUS:-2}"
VLLM_PORT="${VLLM_PORT:-8890}"
VLLM_ENDPOINT="${VLLM_ENDPOINT:-}"
LAUNCH_TEACHER=0
if [[ -z "$VLLM_ENDPOINT" ]]; then
    LAUNCH_TEACHER=1
    VLLM_ENDPOINT="http://127.0.0.1:$VLLM_PORT/v1"
fi
VLLM_ENDPOINT="${VLLM_ENDPOINT%/}"
TRAIN_ROWS="${TRAIN_ROWS:-}"
LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
FSDP_SHARD="${FSDP_SHARD:-0}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-1800}"
# Explicit teacher configuration. Memory ratio follows the reference base.sh;
# extractor-safe scheduling defaults follow the successful capacity probes.
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$(( ((SEQ_LENGTH + 1 + 31) / 32) * 32 ))}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_KV_CACHE_MEMORY_BYTES="${VLLM_KV_CACHE_MEMORY_BYTES:-}"
VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-0}"
if [[ "$VLLM_ENABLE_CHUNKED_PREFILL" == 1 ]]; then
    VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
else
    VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-$VLLM_MAX_MODEL_LEN}"
fi
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
export VLLM_MAX_MODEL_LEN VLLM_GPU_MEMORY_UTILIZATION VLLM_KV_CACHE_MEMORY_BYTES
export VLLM_ENABLE_CHUNKED_PREFILL VLLM_MAX_NUM_BATCHED_TOKENS VLLM_MAX_NUM_SEQS VLLM_ENFORCE_EAGER

# Validate teacher knobs even in print mode, without importing the NPU stack.
if [[ "$LAUNCH_TEACHER" == 1 ]]; then
    SEQ_LENGTH="$SEQ_LENGTH" "$PYTHON" - <<'PY_VLLM'
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
PY_VLLM
fi
export REPO_ROOT MODEL DRAFT_MODEL RAW_DATA_DIR DATA_PATH OUTPUT_DIR INIT_DIR
export SEQ_LENGTH MAX_ANCHORS TRAIN_ROWS VLLM_NPUS TRAIN_NPUS LAUNCH_TEACHER
export PYTHONPATH="$REPO_ROOT/hs_connectors/src:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Pure-stdlib preflight deliberately does not import torch or initialize NPUs.
# Every error is actionable; it is never bypassed for an actual training run.
static_check() {
    "$PYTHON" - <<'PY_CHECK'
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
            'aux_hidden_state_layer_ids': [2, 10, 20, 30, 37], 'block_size': 8,
            'markov_rank': 256, 'markov_head_type': 'vanilla',
            'enable_confidence_head': True, 'confidence_head_with_markov': True,
            'sample_from_anchor': True, 'mask_token_id': 248077}
for key, value in expected.items():
    if draft.get(key) != value:
        errors.append(f'draft {key}: expected {value!r}, got {draft.get(key)!r}')
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
if not list(raw.glob('train-[0-9][0-9][0-9][0-9][0-9].jsonl')):
    errors.append('No raw operator JSONL shards found')
data = Path(os.environ['DATA_PATH'])
if data.suffix == '.jsonl':
    if not data.is_file():
        errors.append(f'Prepared JSONL missing at {data}')
    if not os.environ['TRAIN_ROWS'] and not Path(str(data) + '.manifest.json').is_file():
        errors.append('JSONL requires its preparation manifest or explicit TRAIN_ROWS')
else:
    if not (data / 'state.json').is_file() or not (data / 'dataset_info.json').is_file():
        errors.append(f'Prepared Arrow Dataset missing at {data}')
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
print('Draft: 5 layers, full vocabulary, aux=[2,10,20,30,37]; a run-local config will bind the new teacher')
if errors:
    for error in errors:
        print('BLOCKED:', error)
    raise SystemExit(1)
print('Static configuration checks passed; numerical/NPU validation is still required.')
PY_CHECK
}

# Validate every prepared row; never silently right-truncate raw/long examples.
# This launcher consumes prediction-position masks for the current DSpark path.
data_check() {
    if [[ "$DATA_PATH" == *.jsonl ]]; then
        local args=(check --path "$DATA_PATH" --seq-length "$SEQ_LENGTH")
        [[ -z "$TRAIN_ROWS" ]] || args+=(--train-rows "$TRAIN_ROWS")
        "$PYTHON" "$JSONL_HELPER" "${args[@]}"
        return
    fi
    "$PYTHON" - <<'PY_DATA'
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
limit = int(os.environ['SEQ_LENGTH'])
for index, row in enumerate(data):
    ids, mask = row['input_ids'], row['loss_mask']
    assert 8 < len(ids) <= limit, f'Row {index}: invalid length (no implicit truncation)'
    assert len(ids) == len(mask) == row['seq_len'], f'Row {index}: length mismatch'
    assert all(isinstance(t, int) and 0 <= t < 248320 for t in ids), f'Row {index}: invalid token'
    assert all(m in (0, 1) for m in mask) and sum(mask) > 0, f'Row {index}: invalid/empty mask'
    assert not any(mask[-8:]), f'Row {index}: tail must exclude cross-document anchor blocks'
print(repr(ratio))
PY_DATA
}

read -r -a TRAIN_IDS <<< "${TRAIN_NPUS//,/ }"
read -r -a TEACHER_IDS <<< "${VLLM_NPUS//,/ }"
TRAIN_DATA_RATIO='<computed from TRAIN_ROWS>'
if [[ "$MODE" != print ]]; then
    static_check
    # datasets logging is normally on stderr; only the scalar is read on stdout.
    TRAIN_DATA_RATIO="$(data_check)"
    if [[ "$MODE" == check ]]; then
        printf 'Prepared dataset checked; train-data-ratio=%s\n' "$TRAIN_DATA_RATIO"
        echo 'No files written and no teacher/training processes started.'
        exit 0
    fi
fi

# Prefix caching stays off for the hybrid teacher. Chunked prefill is opt-in:
# reference base.sh uses it for ordinary generation, not hidden-state extraction.
# Local vLLM rejects async scheduling with method=extract_hidden_states.
teacher_cmd=(env "ASCEND_RT_VISIBLE_DEVICES=$VLLM_NPUS" "$PYTHON"
    "$REPO_ROOT/scripts/launch_vllm.py" train "$MODEL"
    --target-layer-ids 2 10 20 30 37 --include-last-layer
    --hidden-states-backend file --hidden-states-path "$OUTPUT_DIR/hidden_states"
    --provenance-dir "$OUTPUT_DIR/provenance" --
    --host 127.0.0.1 --port "$VLLM_PORT" --tensor-parallel-size "${#TEACHER_IDS[@]}"
    --data-parallel-size 1 --max-model-len "$VLLM_MAX_MODEL_LEN"
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
    --max-num-seqs "$VLLM_MAX_NUM_SEQS"
    --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS"
    --api-server-count 1 --renderer-num-workers 1
    --safetensors-load-strategy prefetch
    --no-async-scheduling --no-enable-prefix-caching)
if [[ "$VLLM_ENABLE_CHUNKED_PREFILL" == 1 ]]; then
    teacher_cmd+=(--enable-chunked-prefill)
else
    teacher_cmd+=(--no-enable-chunked-prefill)
fi
[[ "$VLLM_ENFORCE_EAGER" != 1 ]] || teacher_cmd+=(--enforce-eager)
[[ -z "$VLLM_KV_CACHE_MEMORY_BYTES" ]] || teacher_cmd+=(--kv-cache-memory-bytes "$VLLM_KV_CACHE_MEMORY_BYTES")
train_cmd=(env "ASCEND_RT_VISIBLE_DEVICES=$TRAIN_NPUS" "$PYTHON" -m torch.distributed.run
    --standalone --nproc_per_node "${#TRAIN_IDS[@]}" -m speculators.train
    --verifier-name-or-path "$MODEL" --from-pretrained "$INIT_DIR"
    --no-resume-from-checkpoint --data-path "$TRAIN_DATA_PATH"
    --train-data-ratio "$TRAIN_DATA_RATIO" --save-path "$OUTPUT_DIR/checkpoints"
    --vllm-endpoint "$VLLM_ENDPOINT" --hidden-states-backend file
    --epochs "$EPOCHS" --lr "$LR" --optimizer adamw --weight-decay 0.01
    --scheduler-type cosine --scheduler-warmup-ratio 0.03
    --total-seq-len "$SEQ_LENGTH" --max-anchors "$MAX_ANCHORS"
    --speculator-type dspark --target-layer-ids 2 10 20 30 37
    --block-size 8 --mask-token-id 248077
    --markov-rank 256 --markov-head-type vanilla
    --enable-confidence-head --confidence-head-with-markov
    --loss-fn '{"ce":0.1,"tv":0.9}' --confidence-head-alpha 1.0
    --per-position-loss-weight fixed-exp-decay --dflash-decay-gamma 4
    --draft-attn-impl sdpa --hidden-states-dtype bfloat16
    --request-timeout 600 --num-workers "$NUM_WORKERS" --prefetch-factor 2
    --on-missing generate --on-generate delete --checkpoint-freq 0.1
    --log-dir "$OUTPUT_DIR/logs" --run-name "dspark_qwen36_ascendc_$MODE")
# Do not pass --draft-vocab-size: preserve the saved full vocabulary and avoid
# frequency-based remapping from the new dataset.
[[ -z "$MAX_STEPS" ]] || train_cmd+=(--max-steps "$MAX_STEPS")
[[ "$FSDP_SHARD" != 1 ]] || train_cmd+=(--fsdp-shard)
print_command() { printf '%q ' "$@"; printf '\n'; }
if [[ "$MODE" == print ]]; then
    echo 'Preview only: preflight/data preparation have NOT passed; nothing will run.'
    if [[ "$DATA_PATH" == *.jsonl ]]; then
        cache_cmd=("$PYTHON" "$JSONL_HELPER" materialize --path "$DATA_PATH"
            --seq-length "$SEQ_LENGTH" --output "$TRAIN_DATA_PATH")
        [[ -z "$TRAIN_ROWS" ]] || cache_cmd+=(--train-rows "$TRAIN_ROWS")
        print_command "${cache_cmd[@]}"
    fi
    [[ "$LAUNCH_TEACHER" != 1 ]] || print_command "${teacher_cmd[@]}"
    print_command "${train_cmd[@]}"
    exit 0
fi

command -v setsid >/dev/null
command -v curl >/dev/null
cd "$REPO_ROOT"
# mkdir without -p rejects accidental reuse of this run's output directory.
mkdir -p -- "$(dirname -- "$OUTPUT_DIR")"
mkdir -- "$OUTPUT_DIR"
mkdir -- "$INIT_DIR" "$OUTPUT_DIR/logs" "$OUTPUT_DIR/provenance" "$OUTPUT_DIR/hidden_states"
if [[ "$DATA_PATH" == *.jsonl ]]; then
    cache_cmd=("$PYTHON" "$JSONL_HELPER" materialize --path "$DATA_PATH"
        --seq-length "$SEQ_LENGTH" --output "$TRAIN_DATA_PATH")
    [[ -z "$TRAIN_ROWS" ]] || cache_cmd+=(--train-rows "$TRAIN_ROWS")
    cache_ratio="$("${cache_cmd[@]}")"
    [[ "$cache_ratio" == "$TRAIN_DATA_RATIO" ]] || {
        echo 'JSONL split changed after preflight; refusing to train.' >&2; exit 1;
    }
fi
"$PYTHON" - <<'PY_STAGE'
import json
import os
from pathlib import Path
src = Path(os.environ['DRAFT_MODEL']).resolve()
dst = Path(os.environ['INIT_DIR'])
config = json.loads((src / 'config.json').read_text())
config['speculators_config']['verifier']['name_or_path'] = str(Path(os.environ['MODEL']).resolve())
# Only the read-only weight file is shared; never import old optimizer/run state.
(dst / 'model.safetensors').symlink_to(src / 'model.safetensors')
(dst / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
PY_STAGE
print_command "${train_cmd[@]}" > "$OUTPUT_DIR/launch_command.sh.txt"
printf 'RAW_DATA_DIR=%s\nDATA_PATH=%s\nTRAIN_DATA_PATH=%s\nTRAIN_DATA_RATIO=%s\n' \
    "$RAW_DATA_DIR" "$DATA_PATH" "$TRAIN_DATA_PATH" "$TRAIN_DATA_RATIO" > "$OUTPUT_DIR/data_source.txt"

# Own process groups, not shared/external services; terminate child workers too.
TEACHER_PID=''
TRAIN_PID=''
cleanup() {
    local pid
    for pid in "$TRAIN_PID" "$TEACHER_PID"; do
        [[ -z "$pid" ]] || kill -TERM -- "-$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ "$LAUNCH_TEACHER" == 1 ]]; then
    if curl --max-time 2 -sf "http://127.0.0.1:$VLLM_PORT/health" >/dev/null; then
        echo 'Port already serves a model; set VLLM_ENDPOINT explicitly or choose another port.' >&2
        exit 1
    fi
    print_command "${teacher_cmd[@]}" > "$OUTPUT_DIR/teacher_launch_command.sh.txt"
    setsid "${teacher_cmd[@]}" > "$OUTPUT_DIR/logs/teacher.log" 2>&1 &
    TEACHER_PID=$!
    deadline=$((SECONDS + VLLM_STARTUP_TIMEOUT))
    until curl --max-time 2 -sf "http://127.0.0.1:$VLLM_PORT/health" >/dev/null; do
        if ! kill -0 "$TEACHER_PID" 2>/dev/null || (( SECONDS >= deadline )); then
            echo "Teacher failed/timed out. See $OUTPUT_DIR/logs/teacher.log" >&2
            exit 1
        fi
        sleep 2
    done
else
    echo 'Reusing external extractor; its model, layers and shared file storage must match.'
    curl --max-time 10 -sf "${VLLM_ENDPOINT%/v1}/health" >/dev/null
fi
printf 'Training log: %s/logs/train.log\n' "$OUTPUT_DIR"
setsid "${train_cmd[@]}" > "$OUTPUT_DIR/logs/train.log" 2>&1 &
TRAIN_PID=$!
wait "$TRAIN_PID"
TRAIN_PID=''
printf 'Finished. Checkpoints: %s/checkpoints\n' "$OUTPUT_DIR"
