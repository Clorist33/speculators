# 论文版 LK 损失（`lk_hybrid_slot`）代码清单

以下行号以本文件编写时的工作区为准。区间行数按首尾两行均计入；后续编辑文件时行号可能变化。原有的逐 token 损失 `lk_hybrid` 保持独立，新增的 `lk_hybrid_slot` 按当前本地训练批次中同一预测槽位的有效 token 汇总接受概率，再计算该槽位共享的 λ。

## 实现与调用路径

| 文件（绝对路径） | 位置 | 行数 | 作用 |
| --- | --- | ---: | --- |
| [`src/speculators/losses/utils.py`](/home/t00972278/speculators/src/speculators/losses/utils.py:6) | 6 | 1 | 引入 `partial`，用于区分后端并绑定 LK 的上下文参数。 |
| [`src/speculators/losses/utils.py`](/home/t00972278/speculators/src/speculators/losses/utils.py:205) | 205–238 | 34 | `lk_hybrid_slot_loss`：计算 KL、TV，按槽位汇总有效位置的 α，停止 λ 的梯度，并生成逐 token LK 损失。 |
| [`src/speculators/losses/utils.py`](/home/t00972278/speculators/src/speculators/losses/utils.py:249) | 249、261 | 2 | 分别注册 fused、eager 版本，CLI 名称均为 `lk_hybrid_slot`。 |
| [`src/speculators/losses/utils.py`](/home/t00972278/speculators/src/speculators/losses/utils.py:272) | 272–273 | 2 | 补充损失解析器的文档说明。 |
| [`src/speculators/losses/utils.py`](/home/t00972278/speculators/src/speculators/losses/utils.py:325) | 325–326、342–345 | 6 | `compound_loss` 将掩码、block size 和 η 传给新损失；随后沿用现有位置加权和损失归约。 |
| [`src/speculators/losses/__init__.py`](/home/t00972278/speculators/src/speculators/losses/__init__.py:13) | 13、32 | 2 | 导出新损失函数。 |
| [`src/speculators/models/dspark/core.py`](/home/t00972278/speculators/src/speculators/models/dspark/core.py:105) | 105、116、136、211 | 4 | 将 `lk_eta` 从训练配置传入 DSpark 的训练与验证前向。 |
| [`src/speculators/models/dspark/metrics.py`](/home/t00972278/speculators/src/speculators/models/dspark/metrics.py:75) | 75、117–124 | 9 | 将 block size 和 η 送入复合损失；CE、LK、confidence 仍按原框架路径组合。 |
| [`src/speculators/train/config/schema.py`](/home/t00972278/speculators/src/speculators/train/config/schema.py:340) | 340、343–348、808–812 | 12 | 在 CLI 中说明新损失，添加 `--lk-eta`（默认 3，正有限数），并限定 `lk_hybrid_slot` 用于 DSpark。 |

损失关系：`LK_slot = λ_slot × KL(p‖q) + (1 − λ_slot) × TV(p,q)`；`λ_slot = exp(−η × stop_gradient(同槽位有效 α 的平均值))`。这是**当前本地批次**的槽位平均值，未跨数据并行进程额外同步。

## 训练脚本

| 文件（绝对路径） | 位置 | 行数 | 作用 |
| --- | --- | ---: | --- |
| [`examples/train/dspark_qwen3_6_35b_ascendc_online_anchor512_lr1e-3_lk_hybrid_slot.sh`](/home/t00972278/speculators/examples/train/dspark_qwen3_6_35b_ascendc_online_anchor512_lr1e-3_lk_hybrid_slot.sh:1) | 1–227 | 227 | 独立训练脚本，沿用原有教师、数据缓存和训练启动流程。 |

关键参数在脚本 **189–199 行**：block size 8；`--loss-fn '{"ce":0.1,"lk_hybrid_slot":0.9}'`；`--lk-eta 3`；`--confidence-head-alpha 1.0`；`--per-position-loss-weight vat`；VAT 的 `--dflash-decay-gamma 8`。Anchor 512、学习率 1e-3 和前 4 卡推理/后 8 卡训练在 **30–37 行**；含 `lk_hybrid_slot_vat_eta3` 的运行目录和 run name 在 **41、209 行**。VAT 只作用于 CE/LK 的槽位权重，confidence 沿用固定位置衰减。

## 测试

| 文件（绝对路径） | 位置 | 行数 | 覆盖内容 |
| --- | --- | ---: | --- |
| [`tests/unit/models/test_dspark_metrics.py`](/home/t00972278/speculators/tests/unit/models/test_dspark_metrics.py:7) | 7–13 | 7 | 引入新损失及对照测试需要的函数。 |
| [`tests/unit/models/test_dspark_metrics.py`](/home/t00972278/speculators/tests/unit/models/test_dspark_metrics.py:26) | 26–103 | 78 | 共享 λ、无效位置排除、与旧版逐 token LK 的区别、梯度、CE/confidence 组合、全掩码零损失。 |
| [`tests/unit/train/test_cli_args.py`](/home/t00972278/speculators/tests/unit/train/test_cli_args.py:134) | 134–161 | 28 | 新名称和 η 进入训练/验证参数，并拒绝非 DSpark 配置。 |

相关单元测试共 92 项通过；新脚本通过 `bash -n` 检查。尚未进行 NPU 训练实测。
