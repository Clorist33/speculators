# VAT 动态槽位权重：新增与修改的代码路径

以下路径均为绝对路径。VAT 权重**不归一化**；首次不匹配槽位的权重为 **1**，只有其后的槽位开始衰减。

## 功能代码

| 文件 | 改动内容 |
| --- | --- |
| `/home/t00972278/speculators/src/speculators/losses/utils.py` | 新增 `vat_loss_decay()`：按 block 找首次贪心不匹配位置，计算动态槽位权重；处理无效槽位和 mask 空洞。 |
| `/home/t00972278/speculators/src/speculators/losses/__init__.py` | 导出 `vat_loss_decay()`。 |
| `/home/t00972278/speculators/src/speculators/models/dspark/metrics.py` | 在 DSpark 中选择 `vat` 模式；CE/TV 共用同一组动态权重，confidence 保持固定衰减；新增首次不匹配位置及权重诊断指标。 |
| `/home/t00972278/speculators/src/speculators/train/config/schema.py` | 将 `vat` 加入 `--per-position-loss-weight` 可选值，并校验 DSpark 类型、从 anchor 开始预测及正数 `γ`。 |
| `/home/t00972278/speculators/src/speculators/train/trainer.py` | VAT 模式下记录梯度裁剪前的范数。 |

## 实验脚本与说明

| 文件 | 用途 |
| --- | --- |
| `/home/t00972278/speculators/examples/train/dspark_qwen3_6_35b_ascendc_online_anchor512_lr1e-3_vat.sh` | VAT 实验入口；anchor=512、学习率=1e-3、gamma=8，并在运行名称中标记 VAT 与 gamma。 |
| `/home/t00972278/speculators/VAT_DYNAMIC_SLOT_WEIGHT_PLAN.md` | 设计、权重公式、边界处理与验证方案。 |

## 测试代码

| 文件 | 覆盖内容 |
| --- | --- |
| `/home/t00972278/speculators/tests/unit/models/test_dspark_metrics.py` | 首次不匹配位置、mask 边界、损失分母、CE/TV 权重和 confidence 原有权重。 |
| `/home/t00972278/speculators/tests/unit/train/test_cli_args.py` | `vat` 参数传递，以及不支持配置的报错。 |

当前实现只修改训练时的损失计算，不修改数据缓存格式和模型权重配置；尚未启动 NPU 训练。
