#!/usr/bin/env bash
# AscendC DSpark 在线训练：改下面的参数，然后直接 bash 本脚本。
# 流程：检查轨迹缓存 → 准备草稿配置 → 启动教师 → 读取缓存并训练。
# 不再使用 check/smoke/train/print 参数。这里只启动训练，不重新划分原始轨迹。
set -euo pipefail
[[ $# == 0 ]] || { echo '请修改顶部参数；本脚本不再接受模式参数。' >&2; exit 2; }
LAUNCH_SCRIPT="$(realpath -- "${BASH_SOURCE[0]}")"
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
REPO_ROOT="$PWD"

# ============ 1. 模型、数据、输出 ============
PYTHON="python3"
MODEL="/home/docker/Qwen3.6-35B-A3B-agentical-ascendc-hf-4t-bf16"
DRAFT_MODEL="/home/weight/Qwen3.6-35B-A3B-Dspark"
RAW_DATA_DIR="/home/dataset/ascendc-kernelgen-data/dataset/agent_dataset"
# 使用已保存的完整轨迹 Arrow 缓存；原始 JSONL 只用于构建该缓存。
DATA_FORMAT="trajectory-arrow"
DATA_PATH="$RAW_DATA_DIR/preprocess_data"
TRAJECTORY_SHARDS=(train-{00000..00005}.jsonl train-{00007..00015}.jsonl) # 使用 15 个分片，留出 train-00006 做推理评估。
TRAJECTORY_SPLIT_PATH="/home/t00972278/Dspark_sft_docs/data_split/trajectory_split.jsonl"
TRAJECTORY_QUARANTINE_PATH="/home/t00972278/Dspark_sft_docs/data_split/quarantine.json"
TRAJECTORY_OVERLENGTH="truncate"  # error / skip / truncate；超长轨迹截断到 SEQ_LENGTH。
TRAIN_ROWS=""  # 仅旧 arrow 模式使用；轨迹缓存从清单读取划分。

# ============ 2. 草稿训练参数 ============
SEQ_LENGTH="262112" # 256k 档：为额外生成 token 和 32 对齐预留空间。
EPOCHS="2"
MAX_STEPS="0"       # 不设步数上限，完整训练 EPOCHS 个 epoch。
LR="1e-5"
MAX_ANCHORS="1024"  # 每卡每个 token 打包批次最多采样 1024 个 anchor；多条轨迹共享此上限。
NUM_WORKERS="0"     # 避免 spawn worker 额外复制整份数据。
FSDP_SHARD="0"
GRADIENT_CHECKPOINTING="1" # 反向传播时重算草稿层，降低 anchor 1024 的激活显存。
VLLM_NPUS="0,1,2,3" # 教师 TP=4。
TRAIN_NPUS="4,5,6,7,8,9,10,11" # 草稿 DP=8；与教师卡不重叠。
HCCL_INTRA_ROCE_ENABLE="1" # 训练卡跨 NPU 模块时启用跨模块通信。
# 原始草稿已有 5 层、完整词表和 Markov/confidence head，训练时继续加载。
# 不设置 num-layers/draft-vocab-size，避免重新初始化或缩减词表。

RUN_TIME="$(date +%Y%m%d_%H%M%S)_${#TRAJECTORY_SHARDS[@]}shards_anchor${MAX_ANCHORS}_$$"
LOG_ROOT="/Dspark_sft_docs/train-log"
OUTPUT_DIR="$LOG_ROOT/run_$RUN_TIME"
# 控制台日志在预检前开始保存；运行目录由后续步骤创建。
mkdir -p -- "$LOG_ROOT"
exec > >(tee "$LOG_ROOT/launch_$RUN_TIME.log") 2>&1
echo "本次输出目录：$OUTPUT_DIR"

# ============ 3. vLLM 教师参数（参考同事 base.sh） ============
VLLM_PORT="8890"
VLLM_ENDPOINT="" # 留空则启动；填写则复用已有 hidden-state 提取服务。
VLLM_STARTUP_TIMEOUT="1800"
VLLM_MAX_MODEL_LEN="$(( ((SEQ_LENGTH + 1 + 31) / 32) * 32 ))" # 默认 262144。
VLLM_GPU_MEMORY_UTILIZATION="0.90" # 与 base.sh 一致。
VLLM_KV_CACHE_MEMORY_BYTES=""      # 默认空：不传缓存字节预算，由 vLLM 自动计算。
# 显式指定缓存字节数后，0.90 不再参与缓存大小推算。缓存预算是每卡，不是所有卡之和。
VLLM_ENABLE_CHUNKED_PREFILL="1"
# 按同事配置开启分块，每批最多 16384 token；完整 hidden states 仍需实测校验。
if [[ "$VLLM_ENABLE_CHUNKED_PREFILL" == 1 ]]; then
    VLLM_MAX_NUM_BATCHED_TOKENS="16384"
else
    VLLM_MAX_NUM_BATCHED_TOKENS="$VLLM_MAX_MODEL_LEN"
fi
VLLM_MAX_NUM_SEQS="1"  # 同事为 4；提取服务先用单并发。
VLLM_ENFORCE_EAGER="1" # 暂不照搬 FULL_DECODE_ONLY 图模式。
# 命令中还明确保留：DP=1、API/renderer=1、prefetch 权重加载、关闭 prefix cache。
# async scheduling 必须关闭：本地 vLLM 不支持与 extract_hidden_states 组合。
# 上轮没有迁入 base.sh 的 HCCL 端口、allocator、megakernel 等环境设置；这里也不新增。
# 它们继续由当前容器环境提供，不能把同事的端口/设备配置直接用于我们的服务。

# ============ 4. 检查数据并准备本次运行 ============
INIT_DIR="$OUTPUT_DIR/initial_draft"
TRAIN_DATA_PATH="$DATA_PATH"
LAUNCH_TEACHER=0
if [[ -z "$VLLM_ENDPOINT" ]]; then
    LAUNCH_TEACHER=1
    VLLM_ENDPOINT="http://127.0.0.1:$VLLM_PORT/v1"
fi
VLLM_ENDPOINT="${VLLM_ENDPOINT%/}"
read -r -a TEACHER_IDS <<< "${VLLM_NPUS//,/ }"
read -r -a TRAIN_IDS <<< "${TRAIN_NPUS//,/ }"
export REPO_ROOT MODEL DRAFT_MODEL RAW_DATA_DIR DATA_PATH TRAIN_DATA_PATH OUTPUT_DIR INIT_DIR
export DATA_FORMAT TRAJECTORY_SPLIT_PATH TRAJECTORY_OVERLENGTH
export TRAJECTORY_QUARANTINE_PATH
TRAJECTORY_SHARDS_CSV="$(IFS=,; echo "${TRAJECTORY_SHARDS[*]}")"
export TRAJECTORY_SHARDS_CSV
export SEQ_LENGTH MAX_ANCHORS TRAIN_ROWS VLLM_NPUS TRAIN_NPUS LAUNCH_TEACHER
export VLLM_MAX_MODEL_LEN VLLM_GPU_MEMORY_UTILIZATION VLLM_KV_CACHE_MEMORY_BYTES
export VLLM_ENABLE_CHUNKED_PREFILL VLLM_MAX_NUM_BATCHED_TOKENS VLLM_MAX_NUM_SEQS VLLM_ENFORCE_EAGER
export PYTHONPATH="$REPO_ROOT/hs_connectors/src:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# 此处检查缓存与模型、分片、划分和预处理参数是否一致。
TRAIN_DATA_RATIO="$("$PYTHON" scripts/ascendc_dspark_launch.py check)"
command -v setsid >/dev/null
command -v curl >/dev/null
mkdir -p -- "$(dirname -- "$OUTPUT_DIR")"
mkdir -- "$OUTPUT_DIR"  # 不覆盖已有运行。
mkdir -- "$INIT_DIR" "$OUTPUT_DIR/logs" "$OUTPUT_DIR/provenance" "$OUTPUT_DIR/hidden_states"
PREPARED_RATIO="$("$PYTHON" scripts/ascendc_dspark_launch.py prepare)"
[[ "$PREPARED_RATIO" == "$TRAIN_DATA_RATIO" ]] || { echo '准备前后训练划分不一致' >&2; exit 1; }
cp -- "$LAUNCH_SCRIPT" "$OUTPUT_DIR/launch_script.sh" # 保存参数和流程，便于复现。

# 退出时只清理本脚本启动的进程组；外部教师服务不受影响。
TEACHER_PID=''
TRAIN_PID=''
cleanup() {
    for pid in "$TRAIN_PID" "$TEACHER_PID"; do
        [[ -z "$pid" ]] || kill -TERM -- "-$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ============ 5. 启动教师并等待就绪 ============
teacher_cmd=(env "ASCEND_RT_VISIBLE_DEVICES=$VLLM_NPUS" "$PYTHON"
    "$REPO_ROOT/scripts/launch_vllm.py" train "$MODEL"
    --target-layer-ids 2 10 20 30 37
    --include-last-layer
    --hidden-states-backend file
    --hidden-states-path "$OUTPUT_DIR/hidden_states"
    --provenance-dir "$OUTPUT_DIR/provenance"
    --
    --host 127.0.0.1
    --port "$VLLM_PORT"
    --tensor-parallel-size "${#TEACHER_IDS[@]}"
    --data-parallel-size 1
    --max-model-len "$VLLM_MAX_MODEL_LEN"
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
    --max-num-seqs "$VLLM_MAX_NUM_SEQS"
    --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS"
    --api-server-count 1
    --renderer-num-workers 1
    --safetensors-load-strategy prefetch
    --no-async-scheduling
    --no-enable-prefix-caching)
if [[ "$VLLM_ENABLE_CHUNKED_PREFILL" == 1 ]]; then
    teacher_cmd+=(--enable-chunked-prefill)
else
    teacher_cmd+=(--no-enable-chunked-prefill)
fi
[[ "$VLLM_ENFORCE_EAGER" != 1 ]] || teacher_cmd+=(--enforce-eager)
[[ -z "$VLLM_KV_CACHE_MEMORY_BYTES" ]] || teacher_cmd+=(--kv-cache-memory-bytes "$VLLM_KV_CACHE_MEMORY_BYTES")
if [[ "$LAUNCH_TEACHER" == 1 ]]; then
    if curl --max-time 2 -sf "http://127.0.0.1:$VLLM_PORT/health" >/dev/null; then
        echo '教师端口已占用；请更换 VLLM_PORT，或用 VLLM_ENDPOINT 显式复用。' >&2
        exit 1
    fi
    printf '%q ' "${teacher_cmd[@]}" > "$OUTPUT_DIR/teacher_launch_command.sh.txt"
    setsid "${teacher_cmd[@]}" > "$OUTPUT_DIR/logs/teacher.log" 2>&1 &
    TEACHER_PID=$!
    deadline=$((SECONDS + VLLM_STARTUP_TIMEOUT))
    until curl --max-time 2 -sf "http://127.0.0.1:$VLLM_PORT/health" >/dev/null; do
        if ! kill -0 "$TEACHER_PID" 2>/dev/null || (( SECONDS >= deadline )); then
            echo "教师启动失败或超时：$OUTPUT_DIR/logs/teacher.log" >&2
            exit 1
        fi
        sleep 2
    done
else
    echo '复用教师：请确保模型、提取层和文件存储路径匹配。'
    curl --max-time 10 -sf "${VLLM_ENDPOINT%/v1}/health" >/dev/null
fi

# ============ 6. 训练草稿 ============
train_cmd=(env "ASCEND_RT_VISIBLE_DEVICES=$TRAIN_NPUS" "HCCL_INTRA_ROCE_ENABLE=$HCCL_INTRA_ROCE_ENABLE" "$PYTHON" -m torch.distributed.run
    --standalone
    --nproc_per_node "${#TRAIN_IDS[@]}" -m speculators.train
    --verifier-name-or-path "$MODEL"
    --from-pretrained "$INIT_DIR"
    --no-resume-from-checkpoint
    --data-path "$TRAIN_DATA_PATH"
    --data-format "$DATA_FORMAT"
    --train-data-ratio "$TRAIN_DATA_RATIO"
    --save-path "$OUTPUT_DIR/checkpoints"
    --vllm-endpoint "$VLLM_ENDPOINT"
    --hidden-states-backend file
    --hidden-states-path "$OUTPUT_DIR/hidden_states"
    --epochs "$EPOCHS"
    --lr "$LR"
    --optimizer adamw
    --weight-decay 0.01
    --scheduler-type cosine
    --scheduler-warmup-ratio 0.03
    --total-seq-len "$SEQ_LENGTH"
    --max-anchors "$MAX_ANCHORS"
    --speculator-type dspark
    --target-layer-ids 2 10 20 30 37
    --block-size 8
    --mask-token-id 248077
    --markov-rank 256
    --markov-head-type vanilla
    --enable-confidence-head
    --confidence-head-with-markov
    --loss-fn '{"ce":0.1,"tv":0.9}'
    --confidence-head-alpha 1.0
    --per-position-loss-weight fixed-exp-decay
    --dflash-decay-gamma 4
    --draft-attn-impl sdpa
    --hidden-states-dtype bfloat16
    --request-timeout 600
    --num-workers "$NUM_WORKERS"
    --prefetch-factor 2
    --on-missing generate
    --on-generate delete
    --checkpoint-eval-steps 100
    --log-dir "$OUTPUT_DIR/logs"
    --run-name "dspark_qwen36_ascendc")
[[ "$GRADIENT_CHECKPOINTING" != 1 ]] || train_cmd+=(--gradient-checkpointing)
if [[ "$DATA_FORMAT" == trajectory-jsonl || "$DATA_FORMAT" == trajectory-arrow ]]; then
    train_cmd+=(--trajectory-split-path "$TRAJECTORY_SPLIT_PATH"
        --trajectory-overlength "$TRAJECTORY_OVERLENGTH"
        --trajectory-quarantine-path "$TRAJECTORY_QUARANTINE_PATH"
        --trajectory-shards "${TRAJECTORY_SHARDS[@]}")
fi
[[ "$MAX_STEPS" == 0 ]] || train_cmd+=(--max-steps "$MAX_STEPS")
[[ "$FSDP_SHARD" != 1 ]] || train_cmd+=(--fsdp-shard)
printf '%q ' "${train_cmd[@]}" > "$OUTPUT_DIR/launch_command.sh.txt"
printf 'DATA_PATH=%s\nTRAIN_DATA_PATH=%s\nTRAIN_DATA_RATIO=%s\nTRAJECTORY_SHARDS=%s\nTRAJECTORY_QUARANTINE_PATH=%s\n' \
    "$DATA_PATH" "$TRAIN_DATA_PATH" "$TRAIN_DATA_RATIO" "$TRAJECTORY_SHARDS_CSV" "$TRAJECTORY_QUARANTINE_PATH" > "$OUTPUT_DIR/data_source.txt"
echo "训练日志：$OUTPUT_DIR/logs/train.log"
setsid "${train_cmd[@]}" > "$OUTPUT_DIR/logs/train.log" 2>&1 &
TRAIN_PID=$!
wait "$TRAIN_PID"
TRAIN_PID=''
echo "训练完成：$OUTPUT_DIR/checkpoints"
