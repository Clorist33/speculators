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


#LOG="${LOG:-/home/t00972278/speculators/examples/evaluate/dspark_inference_no_prefix_cache.log}"
LOG="${LOG:-/home/t00972278/111111.log}"
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

# mkdir -p "$(dirname "$LOG")"

# # ============================================================
# # 环境加载
# # ============================================================
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
# source "$ROOT/.venv/bin/activate"


set -eo pipefail
set -u


# ============================================================
# 环境变量
# ============================================================
# export PYTHONPATH="$VLLM_ROOT:$ASCEND_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-8,9}"

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
    --max-num-seqs 4 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.90 \
    --safetensors-load-strategy prefetch \
    --enable-chunked-prefill \
    --async-scheduling \
    --trust-remote-code \
    --api-server-count 1 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --generation-config vllm \
    --additional-config '{
        "enable_cpu_binding": true,
        "multistream_overlap_shared_expert": false,
        "draft_window_size": 1024
    }' \
    --speculative-config "{
        \"method\": \"dspark\",
        \"model\": \"$DRAFT\",
        \"num_speculative_tokens\": 8,
        \"enforce_eager\": false
    }" \
    2>&1 | tee "$LOG"

#     --no-enable-prefix-caching \
