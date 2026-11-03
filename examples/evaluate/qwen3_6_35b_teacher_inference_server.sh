#!/usr/bin/env bash
# Teacher-only counterpart of dspark_qwen3_6_35b_inference_server.sh.

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -euo pipefail

MODEL="${MODEL:-/home/docker/Qwen3.6-35B-A3B-agentical-ascendc-hf-4t-bf16}"
LOG_ROOT="${LOG_ROOT:-/home/t00972278/Dspark_sft_docs/rllout性能/时间对比}"
PORT="${PORT:-19096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"

[[ -f "$MODEL/config.json" ]] || { echo "Missing model config: $MODEL/config.json" >&2; exit 1; }
if [[ -n "${LOG:-}" ]]; then
    mkdir -p -- "$(dirname -- "$LOG")"
else
    mkdir -p -- "$LOG_ROOT"
    RUN_DIR="$(mktemp -d -- "$LOG_ROOT/teacher_$(date -u +%Y%m%d_%H%M%S)_XXXXXX")"
    LOG="$RUN_DIR/server.log"
fi
printf 'vLLM server log: %s\n' "$LOG"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-12,13}"
export ENABLE_MEGAKERNEL=0
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ASCEND_ENABLE_FUSED_MC2=0
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=512
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ENGINE_READY_TIMEOUT_S=10000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_IF_BASE_PORT=62000
export HCCL_HOST_SOCKET_PORT_RANGE=62000-62050
export HCCL_NPU_SOCKET_PORT_RANGE=62100-62150

python -c 'import vllm, vllm_ascend, vllm_ascend.vllm_ascend_C; print(vllm.__file__); print(vllm_ascend.__file__)'

vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --served-model-name qwen36 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --logprobs-mode processed_logprobs \
    --tensor-parallel-size 2 \
    --data-parallel-size 1 \
    --max-model-len 262144 \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.90 \
    --safetensors-load-strategy prefetch \
    --enable-chunked-prefill \
    --async-scheduling \
    --trust-remote-code \
    --api-server-count 1 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --generation-config vllm \
    --additional-config '{"enable_cpu_binding":true,"multistream_overlap_shared_expert":false}' \
    2>&1 | tee "$LOG"
