# DSpark 接入 VAT 动态槽位权重：开发方案

## 目标和范围

在现有 `fixed-exp-decay`、`dpace` 之外，增加可选的 `vat` 槽位权重模式。首版只迁移 VAT 的“根据当前 block 的首次贪心拒绝位置，动态调整后续槽位权重”这一部分；保持 DSpark 的模型结构、CE/TV 损失项和 confidence head 目标不变。默认模式仍为 `fixed-exp-decay`。

**本实验不对 VAT 权重做总和归一化。**继续沿用 DSpark 当前的损失分母——有效槽位数。这样首次拒绝之前以及首次拒绝槽位的权重可以保持为 1；整体损失和梯度可能变大，这是本实验有意接受的效果。它与 VAT 原项目的“按权重之和作分母”也不同，因此应称为“DSpark 的 VAT 式槽位加权”，不能把数值结果当成 VAT 原实现的直接复现。

实现状态（2026-11-04）：已接入 `vat` 模式、VAT 专用实验脚本和诊断指标；本文件保留设计与验证依据。尚未启动 NPU 训练，真实推理收益待实验确认。

实验入口：[`examples/train/dspark_qwen3_6_35b_ascendc_online_anchor512_lr1e-3_vat.sh`](examples/train/dspark_qwen3_6_35b_ascendc_online_anchor512_lr1e-3_vat.sh)。当前脚本为 `anchor=512、lr=1e-3、block-size=8、CE/TV=0.1/0.9、γ=8`。相对现有 `γ=4` 的固定衰减脚本，槽位权重模式和 γ 都发生了变化；比较结果时不能把差异仅归因于 VAT 动态权重。

## 权重定义

DSpark 设置 `sample_from_anchor=True` 时，block 的槽位 `k = 0, 1, …, B−1` 都是预测位，其中 `k=0` 是第一个预测 token。使用已经加上 Markov bias 的草稿 logits，与对应教师 logits 分别取 `argmax`，得到逐槽位的贪心匹配结果。权重计算不参与反向传播。

在每个 block 的连续有效槽位内，令 `r` 为第一个 `argmax(draft) ≠ argmax(teacher)` 的槽位：

```text
VAT 权重 w[k] = exp(−max(0, k−r) / γ)
```

- 首次不匹配之前，以及首次不匹配的槽位，权重都是 1；从下一个槽位开始按 `γ` 衰减。
- 如果连续有效槽位内全部匹配，这些槽位的权重都是 1。
- `loss_mask=0` 的位置不参与“首次拒绝”判断，也不产生损失。一个 block 内如果遇到 mask 空洞，贪心验证前缀到此结束；空洞之后仍有效的槽位退回现有固定衰减权重，避免把跨对话片段的槽位误认为同一次连续验证。若第 0 槽位无效，该 block 的有效槽位全部退回固定衰减。
- 首版仅允许 `speculator_type=dspark` 且 `sample_from_anchor=True` 使用 `vat`；其他组合在参数校验时报错，不静默走固定衰减。

例如 `B=4`、`γ=4`、第 3 个槽位（`r=2`）首次不匹配，且 4 个槽位都有效：

| 方案 | 槽位 0～3 的权重 | 权重和 | 损失分母 |
| --- | --- | ---: | ---: |
| 现有固定衰减 | `1, 0.779, 0.607, 0.472` | `2.858` | `4` |
| VAT 式动态权重 | `1, 1, 1, 0.779` | `3.779` | `4` |

如果每个槽位的未加权损失恰好都是 1，以上两组加权平均分别约为 `0.714` 和 `0.945`。这说明新模式既改变了位置分配，也提高了整体损失尺度；**不乘 `2.858/3.779` 之类的归一化系数，也不把分母改成权重和**。如果首次不匹配发生在槽位 0，新旧权重相同；如果全部匹配，新权重全为 1。

这里的“首次拒绝”是**贪心 `argmax` 不匹配的代理指标**，不是训练时抽样得到的一次真实拒绝事件，也不等于推理侧拒绝采样的接受率。现有 DSpark 的 `accept_rate = 1 − TV`、`accept_len` 和 confidence 目标继续保留原定义。
此外，训练中的后续槽位使用数据里的真实历史 token 作为 Markov 条件；实际推理会使用草稿自己生成的历史 token。两者可能不同，因此是否提高真实接受率必须通过推理评测确认。

## 代码改动

1. **配置入口。**在 [`src/speculators/train/config/schema.py`](src/speculators/train/config/schema.py) 的 `per_position_loss_weight` 可选值中增加 `vat`；校验 DSpark 类型、`sample_from_anchor=True` 和 `γ>0`。`fixed-exp-decay`、`dpace` 的默认值与限制不变。训练命令通过 `--per-position-loss-weight vat` 选择新模式，继续用 `--dflash-decay-gamma` 控制首次拒绝后的衰减速度。
2. **权重计算。**在 [`src/speculators/models/dspark/metrics.py`](src/speculators/models/dspark/metrics.py) 或 [`src/speculators/losses/utils.py`](src/speculators/losses/utils.py) 新增按 block 计算 VAT 权重的函数。复用现有的草稿/教师 `argmax`；输入含 block size、`loss_mask`、`γ`，输出与槽位位置对齐的权重张量。计算使用 `no_grad`/`detach`，检查形状和有效位。禁止未知模式落入当前 `else → fixed-exp-decay` 分支。
3. **接入 CE/TV。**同一个 block 的权重只计算一次，并复用到 `compound_loss` 中的 CE、TV 两项；不要让两项各自独立判断首次拒绝。保留各项原有系数，例如基线 `CE=0.1、TV=0.9`。现有 [`src/speculators/losses/utils.py`](src/speculators/losses/utils.py) 的损失分母继续是有效槽位数，**不做权重总和归一化**。
4. **confidence head。**首版继续用固定指数衰减计算 confidence BCE，保持其软目标 `1−TV`、`confidence_head_alpha` 和既有校准指标不变。这可以先观察 CE/TV 动态加权的效果，避免同时改变 confidence 训练。`vat` 不新增 VAT 论文中的独立验证头或硬标签 BCE。
5. **训练与验证参数。**[`src/speculators/models/dspark/core.py`](src/speculators/models/dspark/core.py) 已将同一组训练参数传给 train/val；新模式需在两侧都生效。VAT 实验脚本沿用选定的 DSpark 基线数据、权重初始化、block size、anchor、学习率和损失系数；按当前实验要求将 γ 从 4 调整为 8，并在运行名中标记。不要直接覆盖现有固定衰减实验脚本。
6. **观察指标。**增加每个 block 的首次不匹配位置分布、无不匹配比例、平均权重和、相对固定衰减的平均权重比；继续记录 CE、TV、confidence、`accept_len`、`eal`，并确认训练日志能记录梯度范数。真实推理接受率需另行实测。梯度范数与训练稳定性尤其重要，因为不归一化会提高整体训练力度。

## 验证与实验

- 单元测试覆盖：首次不匹配在槽位 0／中间／最后、全部匹配、部分 mask、mask 空洞、全空 block、多个 block、`γ` 边界；核对权重和实际 CE/TV 数值，确认分母仍为有效槽位数且权重无梯度。
- 回归测试覆盖：不指定 `vat` 时原固定衰减输出不变；`dpace` 不变；confidence BCE 仍使用固定衰减；训练与验证读取相同模式；不支持的配置明确报错。
- 用少量真实轨迹做前向和反向冒烟测试，检查 eager/fused 损失路径、有限损失、梯度与日志；再进行完整训练。对照实验必须保持基线的其他参数一致，并分别比较验证集指标和**相同真实推理任务**的接受率。
- 解释结果时同时看权重和、梯度范数和接受率：如有提升，不能单凭这组实验断定原因仅是“更好的槽位分配”，因为不归一化也改变了整体训练强度。后续若要拆分这两个因素，再单独补一组归一化消融实验。

只改损失权重，不改教师 hidden states、trajectory cache 的格式或模型预测槽位数，因此首版不需要重建预处理缓存，也不需要修改 DSpark 权重配置。
