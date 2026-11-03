#!/usr/bin/env bash
# ============================================================
# 路径与变量
# ============================================================
ROOT=/workspace/vllm-ascend
# VLLM_ROOT="${VLLM_ROOT:-${ROOT}/dspark_test/worktrees/vllm-ba07}"
# ASCEND_ROOT="${ROOT}/dspark_test/worktrees/megakernel-rebase-sim"
# BINARY_ROOT="${ROOT}/vllm-ascend"

MODEL=/home/docker/Qwen3.6-35B-A3B-agentical-ascendc-hf-4t-bf16
DRAFT="${DRAFT:-/home/weight/Qwen3.6-35B-A3B-Dspark}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-8}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
CUDAGRAPH_REQUIRED_SIZE=$(( MAX_NUM_SEQS * (NUM_SPECULATIVE_TOKENS + 1) ))
CUDAGRAPH_CAPTURE_SIZE=$(( (CUDAGRAPH_REQUIRED_SIZE + 7) / 8 * 8 ))
CUDAGRAPH_CAPTURE_SIZES='[1,2,4'
for ((capture_size = 8; capture_size <= CUDAGRAPH_CAPTURE_SIZE; capture_size += 8)); do
    CUDAGRAPH_CAPTURE_SIZES+=",$capture_size"
done
if (( CUDAGRAPH_REQUIRED_SIZE % 8 != 0 )); then
    CUDAGRAPH_CAPTURE_SIZES+=",$CUDAGRAPH_REQUIRED_SIZE"
fi
CUDAGRAPH_CAPTURE_SIZES+=']'


LOG_ROOT="${LOG_ROOT:-/home/t00972278/Dspark_sft_docs/rllout性能/时间对比}"
PORT="${PORT:-9092}"
# EXPECTED_VLLM=ba07e4a48fc951300d97eb506217dd530583dea3

# ============================================================
# 前置校验
# ============================================================
# for path in "$VLLM_ROOT" "$ASCEND_ROOT/vllm_ascend" "$MODEL" "$DRAFT"; do
#     [[ -e "$path" ]] || { echo "Missing: $path" >&2; exit 1; }
# done

# [[ "$(git -C "$VLLM_ROOT" rev-parse HEAD)" == "$EXPECTED_VLLM" ]] || {
#     echo "vLLM must be $EXPECTED_VLLM" >&2
#     exit 1
# }

# ------------------------------------------------------------
# 预编译二进制软链接（按需启用）
# ------------------------------------------------------------
# ln -sfn "$BINARY_ROOT/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so" \
#     "$ASCEND_ROOT/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so"
# ln -sfn "$BINARY_ROOT/vllm_ascend/lib" \
#     "$ASCEND_ROOT/vllm_ascend/lib"
# ln -sfn "$BINARY_ROOT/vllm_ascend/_cann_ops_custom/vendors" \
#     "$ASCEND_ROOT/vllm_ascend/_cann_ops_custom/vendors"

# # ============================================================
# # 环境加载
# # ============================================================
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
# source "$ROOT/.venv/bin/activate"


set -eo pipefail
set -u

if [[ -n "${LOG:-}" ]]; then
    mkdir -p -- "$(dirname -- "$LOG")"
else
    mkdir -p -- "$LOG_ROOT"
    RUN_DIR="$(mktemp -d -- "$LOG_ROOT/dspark_$(date -u +%Y%m%d_%H%M%S)_XXXXXX")"
    LOG="$RUN_DIR/server.log"
fi
printf 'vLLM server log: %s\n' "$LOG"


# ============================================================
# 环境变量
# ============================================================
# export PYTHONPATH="$VLLM_ROOT:$ASCEND_ROOT${PYTHONPATH:+:$PYTHONPATH}"
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

# export VLLM_LOG_STATS_INTERVAL=5

#export HCCL_INTRA_ROCE_ENABLE=${HCCL_INTRA_ROCE_ENABLE:-1}
#export HCCL_INTRA_PCIE_ENABLE=${HCCL_INTRA_PCIE_ENABLE:-0}


# ============================================================
# 导入自检
# ============================================================
python -c 'import vllm, vllm_ascend, vllm_ascend.vllm_ascend_C; print(vllm.__file__); print(vllm_ascend.__file__)'

# ============================================================
# 启动 vLLM 服务
# ============================================================
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
    --compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"max_cudagraph_capture_size\":$CUDAGRAPH_CAPTURE_SIZE,\"cudagraph_capture_sizes\":$CUDAGRAPH_CAPTURE_SIZES}" \
    --generation-config vllm \
    --additional-config '{
        "enable_cpu_binding": true,
        "multistream_overlap_shared_expert": false,
        "draft_window_size": 1024
    }' \
    --speculative-config "{
        \"method\": \"dspark\",
        \"model\": \"$DRAFT\",
        \"num_speculative_tokens\": $NUM_SPECULATIVE_TOKENS,
        \"enforce_eager\": false
    }" \
    2>&1 | tee "$LOG"

#     --no-enable-prefix-caching \
