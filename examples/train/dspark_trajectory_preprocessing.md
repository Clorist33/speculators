# DSpark 完整轨迹预处理

工作分支：单条轨迹作为一条训练样本。工作目录：`/home/t00972278/speculators-single-trajectory`。

脚本：`scripts/prepare_ascendc_dspark_trajectory.py`。每条有效 OpenAI messages 轨迹输出一个 Arrow 样本，列为 input_ids、loss_mask、seq_len。沿用原始 trajectory_split.jsonl 的分组划分，训练行在前、验证行在后，比例写入 data_arrow/split.json。

采用教师 tokenizer 的 chat_template，并显式设置 preserve_thinking=True。学习所有 assistant 的正文（包括思考、工具调用、结束标记和随后的换行），不学习 assistant 角色头、用户、系统和工具返回。token 目标标记左移一位成为 DSpark 预测位置，末尾 block_size 个位置清零。这里的候选位置数量不是每步实际采样的 anchor 数量。

这与 MindSpeed 的整轨迹、多轮监督思路相同，但不是逐字复制其转换器：使用教师自己的模板处理连续 tool 消息；中途 user 保持原角色，不折叠进工具返回；畸形工具参数或非 assistant 结尾显式报错，不静默修改数据。当前仅支持文本内容。正式使用前应核对中途 user 和历史思考策略与部署输入的关系。

运行示例（输出目录必须不存在）：

```bash
cd /home/t00972278/speculators-single-trajectory
python3 scripts/prepare_ascendc_dspark_trajectory.py \
  --input /home/t00972278/dataset/ascendc-kernelgen-data/dataset/agent_dataset/train-00000.jsonl \
  --model /home/docker/Qwen3.6-35B-A3B-agentical-ascendc-hf-4t-bf16 \
  --split /home/t00972278/Dspark_sft_docs/待删除/20260929/trajectory_split.jsonl \
  --output /home/t00972278/Dspark_sft_docs/trajectory-data-new \
  --max-length 262112 --block-size 8 --overlength error
```

`--limit N` 仅用于检查前 N 条，不保证同时包含训练和验证样本。正式生成不传 limit，生成后核对两侧样本数。

超长策略：error（默认，停止并报告原始行号）；skip（整条跳过并记账）；truncate（保留完整渲染序列的前 max-length 个 token，可能截断回答）。truncate 不等同于 MindSpeed 按轮分配剩余输入/回答长度的算法，必须显式选择。不会自动拆分为多条样本。

产物：data_arrow/、manifest.json（参数、数量和来源摘要）、audit.jsonl（逐条长度、目标数、丢弃数及划分）、first_trajectory.txt（首条完整渲染文本）、first_trajectory_target_mask.json（首条保留部分的 token 目标标记）。中途失败不会生成成功 manifest；保留失败目录供检查，重试使用新的输出目录。

预处理无需教师服务或 NPU 前向，不修改已有数据。当前未修改 batch sampler、max_anchors 或启动脚本。此脚本只准备新格式数据，不代表正式训练已完成兼容性验证。
