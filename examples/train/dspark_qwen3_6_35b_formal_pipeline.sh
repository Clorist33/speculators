#!/usr/bin/env bash
# Prepare every trajectory in train-00000.jsonl, then run five DSpark epochs.
set -euo pipefail

cd /home/t00972278/speculators
REPO_ROOT="$PWD"
MODEL="/home/docker/Qwen3.6-35B-A3B-agentical-ascendc-hf-4t-bf16"
DATA_ROOT="/home/t00972278/Dspark_sft_docs/formal-data/train00000_256k"
LOG_ROOT="/home/t00972278/Dspark_sft_docs/train-log"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
PREP_LOG_DIR="$LOG_ROOT/prepare_train00000_$RUN_ID"
mkdir -p -- "$PREP_LOG_DIR/hidden_states" "$PREP_LOG_DIR/provenance"
exec > >(tee "$PREP_LOG_DIR/pipeline.log") 2>&1
export PYTHONPATH="$REPO_ROOT/hs_connectors/src:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

TEACHER_PID=''
cleanup() {
    if [[ -n "$TEACHER_PID" ]]; then
        kill -TERM -- "-$TEACHER_PID" 2>/dev/null || true
        wait "$TEACHER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ ! -f "$DATA_ROOT/data_arrow/state.json" ]]; then
    if curl --max-time 2 -sf http://127.0.0.1:8890/health >/dev/null; then
        echo 'Port 8890 is already serving; cannot start the preparation teacher.' >&2
        exit 1
    fi
    teacher_cmd=(env ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python3
        "$REPO_ROOT/scripts/launch_vllm.py" train "$MODEL"
        --target-layer-ids 2 10 20 30 37
        --include-last-layer
        --hidden-states-backend file
        --hidden-states-path "$PREP_LOG_DIR/hidden_states"
        --provenance-dir "$PREP_LOG_DIR/provenance"
        --
        --host 127.0.0.1
        --port 8890
        --tensor-parallel-size 4
        --data-parallel-size 1
        --max-model-len 262144
        --gpu-memory-utilization 0.90
        --max-num-seqs 1
        --max-num-batched-tokens 16384
        --api-server-count 1
        --renderer-num-workers 1
        --safetensors-load-strategy prefetch
        --no-async-scheduling
        --no-enable-prefix-caching
        --enable-chunked-prefill
        --enforce-eager)
    printf '%q ' "${teacher_cmd[@]}" > "$PREP_LOG_DIR/teacher_command.txt"
    setsid "${teacher_cmd[@]}" > "$PREP_LOG_DIR/teacher.log" 2>&1 &
    TEACHER_PID=$!
    deadline=$((SECONDS + 1800))
    until curl --max-time 2 -sf http://127.0.0.1:8890/health >/dev/null; do
        if ! kill -0 "$TEACHER_PID" 2>/dev/null || (( SECONDS >= deadline )); then
            echo "Preparation teacher failed; inspect $PREP_LOG_DIR/teacher.log" >&2
            exit 1
        fi
        sleep 2
    done
    echo "Preparation teacher ready; logs: $PREP_LOG_DIR"
    python3 scripts/prepare_ascendc_dspark_formal.py \
        --output "$DATA_ROOT" \
        --render-endpoint http://127.0.0.1:8890 \
        --chunk-size 10 \
        --workers 2 > "$PREP_LOG_DIR/prepare.log" 2>&1
    echo "Formal Arrow ready: $DATA_ROOT/data_arrow"
    cleanup
    TEACHER_PID=''
fi

echo 'Starting five-epoch training from the original draft.'
bash examples/train/dspark_qwen3_6_35b_ascendc_online.sh
