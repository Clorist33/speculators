> 当前配置更新：教师已开启 chunked prefill，max-num-batched-tokens=16384，max-model-len=262144，自动缓存预算（utilization=0.90）。此组合尚未完成 NPU 启动及特征完整性验证。

> 2026-09-30 更新：现在直接 bash 主脚本即可。顶部配置使用明确赋值，默认读取已对齐的 64k 容量探针数据（重复样本），教师上限256k、自动缓存、4卡教师+4卡训练、单步验证。日志自动写入 `/home/t00972278/Dspark_sft_docs/train-log/launch_时间.log` 和 `run_时间/logs/`。实际样本不是256k。以下环境变量运行示例为此前接口，现请直接修改脚本顶部对应变量。

# AscendC DSpark 简化训练脚本

主脚本：`dspark_qwen3_6_35b_ascendc_online.sh`。按参考 ShareGPT 脚本的线性顺序整理，直接修改顶部变量后执行；不再接收 `check/smoke/train/print`。检查和数据格式转换位于 `../../scripts/ascendc_dspark_launch.py`，不会修改原始权重或数据。

## 数据入口

必须设置 `DATA_PATH`，指向已完成原生 mask → DSpark 预测位置对齐的 Arrow，或含 `input_ids/loss_mask/seq_len` 的预分词 JSONL。不能传原始轨迹 JSONL，也不能直接传未经 mask 对齐的原生 prepare-data 输出。训练/验证分界由 `TRAIN_ROWS` 或数据清单提供，保留既有轨迹分组。

全量 train/validation.conversations.jsonl 目前只是原始轨迹划分，尚不能直接作为本脚本训练输入。此次是启动脚本重写，没有改变 mask 生成方式。

## 运行方式

单步验证（先替换真实数据路径）：

```bash
DATA_PATH=/path/to/aligned_arrow \
bash examples/train/dspark_qwen3_6_35b_ascendc_online.sh
```

默认教师 0–3 卡、草稿 4–7 卡，SEQ_LENGTH=262112（教师上限262144）、MAX_ANCHORS=16、NUM_WORKERS=1、MAX_STEPS=1。默认不指定缓存字节预算，使用 gpu-memory-utilization=0.90 自动计算。运行前应清除环境中旧的 VLLM_KV_CACHE_MEMORY_BYTES 覆盖值。此 256k 配置尚未通过 NPU 容量验证。

正式训练需另行验证数据、容量与连续稳定性，然后设置 `MAX_STEPS=0 EPOCHS=5`；0 是脚本约定，实际不传 `--max-steps`。不再隐式选择旧 first16 数据。

`VLLM_ENDPOINT` 非空时只复用服务，不启动/停止教师，也不会把本脚本教师参数应用到外部服务。必须确保模型、提取层及隐藏状态共享路径匹配。

## vLLM/vLLM-Ascend 参数来源

参考 `/home/c00937190/spec/base.sh`，保留上一版已迁入配置：

| 参数 | 本脚本 | 同事脚本/解释 |
|---|---|---|
| gpu-memory-utilization | 0.90 | 相同；自动缓存模式的显存预算 |
| kv-cache-memory-bytes | 可选，默认自动 | 我们新增的显式缓存预算；指定后不按 utilization 推算缓存 |
| tensor-parallel-size | VLLM_NPUS 的数量，默认 4 | 同事为 2；可见 8 卡不代表 TP=8 |
| data-parallel-size | 1 | 相同 |
| max-model-len | SEQ_LENGTH+1 向上对齐32 | 同事固定262144；为提取额外token留余量 |
| chunked prefill | 默认关闭，可设 VLLM_ENABLE_CHUNKED_PREFILL=1 | 同事开启；提取路径完整性仍需验证 |
| max-num-batched-tokens | 分块开启16384；关闭时等于教师长度 | 分块开启时与同事相同 |
| max-num-seqs | 默认1，可覆盖 | 同事4；提取服务先限制并发 |
| prefix caching | 关闭 | 相同 |
| async scheduling | 关闭 | 同事开启；本地 vLLM 不支持 extract_hidden_states 与 async 组合 |
| api-server-count | 1 | 相同 |
| renderer-num-workers | 1 | 提取服务显式限制 |
| safetensors-load-strategy | prefetch | 相同 |
| enforce-eager | 默认开启，可设 VLLM_ENFORCE_EAGER=0 | 同事 FULL_DECODE_ONLY；本脚本不设置该 compilation-config |

上一版未迁入同事的 `ENABLE_MEGAKERNEL`、`VLLM_USE_V2_MODEL_RUNNER`、`VLLM_ASCEND_ENABLE_FUSED_MC2`、HCCL 环境变量/端口、`PYTORCH_NPU_ALLOC_CONF`、CPU binding 和 shared-expert additional-config。本次结构重写也不添加它们；环境变量仍继承容器当前值。不要把普通生成服务的完整设置等同于训练特征提取配置。

本脚本继续使用 vLLM-Ascend，通过 `ASCEND_RT_VISIBLE_DEVICES` 指定教师和训练设备；不使用参考脚本的 CUDA_VISIBLE_DEVICES。不修改同事 base.sh 或原始 ShareGPT 参考脚本。

## 日志与复现

每次运行使用新 OUTPUT_DIR；保存脚本副本、实际教师/训练命令、数据来源、provenance 及日志。教师日志在 `logs/teacher.log`，训练日志在 `logs/train.log`。退出时清理本脚本启动的进程组，保留外部教师服务。
