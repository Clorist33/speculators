# 完整轨迹 DSpark 训练入口

本分支先使用 `scripts/prepare_ascendc_dspark_trajectory.py` 生成数据，再运行 `examples/train/dspark_qwen3_6_35b_ascendc_online.sh`。预处理说明见 [dspark_trajectory_preprocessing.md](dspark_trajectory_preprocessing.md)。

训练脚本只读取已准备好的 Arrow，不能直接传原始轨迹或 JSONL。先修改脚本顶部 DATA_PATH；默认指向 `/home/t00972278/Dspark_sft_docs/trajectory-data-new/data_arrow`，该目录尚待正式生成验证。不会自动使用旧的回合展开数据。训练和验证必须都非空。

启动辅助 `scripts/ascendc_dspark_launch.py` 负责检查数据、模型和设备配置，以及创建本次运行的草稿配置，不再转换数据。

原有训练参数仍保留：每卡 token 预算 262112、最多 128 个预测起点、5 个 epoch、4 卡教师和 4 卡训练。尚未改成固定轨迹条数的 batch-size，尚未完成新数据的 NPU 训练验证。运行前按实际设备和验证计划修改顶部参数。本次整理没有启动训练。

脚本根据自身位置选择工作目录，并设置本工作目录的 PYTHONPATH。原 `/home/t00972278/speculators` 的 main 工作目录不变。

运行时保存启动脚本、教师和训练命令、数据来源、provenance 和日志；保留外部教师服务，仅清理自身启动的服务。

旧的全量展开、小样本转换、预算筛选、旧版启动和准备后自动训练脚本已从本分支移除，可在 `单条轨迹拆分成多条训练样本` 分支的 `6f992b2` 提交查看。框架通用 prepare-data 功能保持原样。
