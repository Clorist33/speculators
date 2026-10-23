#!/usr/bin/env bash
# AscendC DSpark 续训：从旧运行的 step 500 checkpoint 恢复，4 卡教师、8 卡草稿训练。
# 数据并行从 4 改为 8 后，首个 epoch 的采样位置会改变；按约定接受漏训/重训轨迹。
# 创建独立运行目录，复制恢复 checkpoint 与历史最佳 checkpoint，不修改旧运行。
set -euo pipefail
[[ $# == 0 ]] || { echo '请修改顶部参数；本脚本不再接受模式参数。' >&2; exit 2; }
LAUNCH_SCRIPT="$(realpath -- "${BASH_SOURCE[0]}")"
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
REPO_ROOT="$PWD"

# ============ 1. 模型、数据、输出 ============
PYTHON="python3"
MODEL="/home/docker/Qwen3.6-35B-A3B-agentical-ascendc-hf-4t-bf16"
DRAFT_MODEL="/home/weight/Qwen3.6-35B-A3B-Dspark"
RAW_DATA_DIR="/home/t00972278/dataset/ascendc-kernelgen-data/dataset/agent_dataset"
# 使用已保存的完整轨迹 Arrow 缓存；原始 JSONL 只用于构建该缓存。
DATA_FORMAT="trajectory-arrow"
DATA_PATH="$RAW_DATA_DIR/preprocess_data"
TRAJECTORY_SHARDS=(train-{00000..00005}.jsonl train-{00007..00015}.jsonl) # 使用 15 个分片，留出 train-00006 做推理评估。
TRAJECTORY_SPLIT_PATH="/home/t00972278/Dspark_sft_docs/data_split/trajectory_split.jsonl"
TRAJECTORY_QUARANTINE_PATH="/home/t00972278/Dspark_sft_docs/data_split/quarantine.json"
TRAJECTORY_OVERLENGTH="truncate"  # error / skip / truncate；超长轨迹截断到 SEQ_LENGTH。
TRAIN_ROWS=""  # 仅旧 arrow 模式使用；轨迹缓存从清单读取划分。
SOURCE_RUN_DIR="/home/t00972278/Dspark_sft_docs/train-log/run_20261018_114507_15shards_anchor512_2800854"
RESUME_CHECKPOINT="step_00000500"

# ============ 2. 草稿训练参数 ============
SEQ_LENGTH="262112" # 256k 档：为额外生成 token 和 32 对齐预留空间。
EPOCHS="5"
MAX_STEPS="0"       # 不设步数上限，完整训练 5 个 epoch。
LR="1e-5"
MAX_ANCHORS="512"  # 每卡每个 token 打包批次最多采样 512 个 anchor；多条轨迹共享此上限。
NUM_WORKERS="0"     # 避免 spawn worker 额外复制整份数据。
FSDP_SHARD="0"
GRADIENT_CHECKPOINTING="1" # 反向传播时重算草稿层，降低 anchor 512 的激活显存。
VLLM_NPUS="0,1,2,3" # 教师 TP=4；启动前确认这些 NPU 空闲。
TRAIN_NPUS="4,5,6,7,8,9,10,11" # 草稿 DP=8；与教师卡不重叠。
# 训练卡跨 NPU module（4-7 与 8-11），HCCL 需要通过 RoCE 建立卡间通信。
HCCL_INTRA_ROCE_ENABLE="1"
# 原始草稿已有 5 层、完整词表和 Markov/confidence head，训练时继续加载。
# 不设置 num-layers/draft-vocab-size，避免重新初始化或缩减词表。

RUN_TIME="$(date +%Y%m%d_%H%M%S)_${#TRAJECTORY_SHARDS[@]}shards_anchor${MAX_ANCHORS}_resume500_4infer8train_$$"
LOG_ROOT="/home/t00972278/Dspark_sft_docs/train-log"
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
[[ ${#TEACHER_IDS[@]} == 4 && ${#TRAIN_IDS[@]} == 8 ]] || {
    echo '续训要求 4 张教师卡和 8 张训练卡。' >&2
    exit 1
}
SOURCE_CHECKPOINTS="$SOURCE_RUN_DIR/checkpoints"
SOURCE_RESUME_DIR="$SOURCE_CHECKPOINTS/$RESUME_CHECKPOINT"
SOURCE_DATA_MANIFEST="$SOURCE_CHECKPOINTS/trajectory_data_manifest.json"
[[ -s "$SOURCE_DATA_MANIFEST" && -s "$DATA_PATH/preprocess_manifest.json" ]] || {
    echo '缺少原运行或当前缓存的数据清单。' >&2
    exit 1
}
cmp -s -- "$SOURCE_DATA_MANIFEST" "$DATA_PATH/preprocess_manifest.json" || {
    echo '当前轨迹缓存与 step 500 使用的数据不一致，拒绝按保存的 batch 位置续训。' >&2
    exit 1
}
for name in config.json model.safetensors optimizer_state_dict.pt scheduler_state_dict.pt training_state.json; do
    [[ -s "$SOURCE_RESUME_DIR/$name" ]] || {
        echo "缺少完整续训 checkpoint 文件：$SOURCE_RESUME_DIR/$name" >&2
        exit 1
    }
done
"$PYTHON" - "$SOURCE_RESUME_DIR/training_state.json" <<'PY'
import json
import sys

state = json.load(open(sys.argv[1], encoding="utf-8"))
if state != {"epoch": 0, "local_step": 500, "global_step": 500}:
    raise SystemExit(f"续训位置不符：{state!r}")
PY
[[ -L "$SOURCE_CHECKPOINTS/checkpoint_best" ]] || {
    echo "缺少历史最佳 checkpoint 链接：$SOURCE_CHECKPOINTS/checkpoint_best" >&2
    exit 1
}
SOURCE_BEST_NAME="$(readlink -- "$SOURCE_CHECKPOINTS/checkpoint_best")"
[[ "$SOURCE_BEST_NAME" =~ ^step_[0-9]{8}$ && -s "$SOURCE_CHECKPOINTS/$SOURCE_BEST_NAME/val_metrics.json" ]] || {
    echo "历史最佳 checkpoint 不完整：$SOURCE_BEST_NAME" >&2
    exit 1
}
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
# 训练器只从 save-path 中自动寻找 checkpoint；复制到新目录以保留旧运行。
mkdir -- "$OUTPUT_DIR/checkpoints"
cp -a -- "$SOURCE_RESUME_DIR" "$OUTPUT_DIR/checkpoints/"
cp -- "$SOURCE_DATA_MANIFEST" "$OUTPUT_DIR/checkpoints/trajectory_data_manifest.json"
if [[ "$SOURCE_BEST_NAME" != "$RESUME_CHECKPOINT" ]]; then
    cp -a -- "$SOURCE_CHECKPOINTS/$SOURCE_BEST_NAME" "$OUTPUT_DIR/checkpoints/"
fi
ln -s -- "$SOURCE_BEST_NAME" "$OUTPUT_DIR/checkpoints/checkpoint_best"
printf 'SOURCE_RUN_DIR=%s\nRESUME_CHECKPOINT=%s\nSOURCE_BEST=%s\n' \
    "$SOURCE_RUN_DIR" "$RESUME_CHECKPOINT" "$SOURCE_BEST_NAME" > "$OUTPUT_DIR/resume_source.txt"
echo "续训来源：$SOURCE_RESUME_DIR；历史最佳：$SOURCE_BEST_NAME"
echo 'step 500 的验证曾中断；本次从 step 501 训练，下一次定期验证在 step 600。'
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
    --run-name "dspark_qwen36_ascendc_resume500_4infer8train")
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
