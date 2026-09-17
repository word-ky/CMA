# CMA：煤矿视觉引用分割与工具调度

恢复自 CMLLM 项目的方法源码，供代码审阅和研究讨论。核心任务是在煤矿图像中，使用上一轮矿工掩码作为视觉引用，分割对应安全帽，并结合增强、区域聚焦和多步工具执行。

## 建议阅读顺序

1. [完整方法说明：需求、Framework、动机、贡献与流程](research_log/20260917_method_framework_for_chatgpt.md)
2. [REF 分割模型](cmllm_remote/third_party/LISA/model/LISA.py)：引用如何影响掩码生成。
3. [多模态输入](cmllm_remote/third_party/LISA/model/llava/model/llava_arch.py)与[语言模型](cmllm_remote/third_party/LISA/model/llava/model/language_model/llava_llama.py)：REF 输入通路。
4. [引用数据集](cmllm_remote/third_party/LISA/utils/mr_ref_seg_dataset.py)与[反事实数据构建](cmllm_remote/scripts/build_mr_ref_counterfactual_train.py)。
5. [任务增强器](cmllm_remote/scripts/train_task_enhancer_v3_dual_direct_loss.py)。
6. [工具实现](cmllm_remote/scripts/stage3_rule_controller_v3_seg_local_enhance.py)与[策略执行器](cmllm_remote/scripts/stage3_policy_v2_rollout.py)。
7. [SFT](cmllm_remote/scripts/train_controller_policy_sft.py)、[DPO](cmllm_remote/scripts/train_controller_policy_dpo.py)及[偏好数据构造](cmllm_remote/scripts/build_controller_policy_rollout_preferences_v4.py)。
8. [复现指标与解释](research_log/REPRODUCTION.md)。

## 目录与运行边界

- `cmllm_remote/`：恢复的项目源代码、配置、训练及评测脚本，保留原目录结构。
- `cmllm_remote/third_party/LISA/`：恢复到的 LISA 修改文件，**不是完整上游仓库**。保留第三方文件原有声明；本仓库未为第三方代码另行授予许可证。
- `research_log/`：方法说明、数值结果、环境记录和发布记录。

这是一份源码阅读快照。没有提交模型权重、图像、伪掩码、API 密钥或服务器连接配置。要运行实验，还需原项目使用的完整 LISA 依赖树、模型与数据，并调整历史脚本中的本地路径。`environment_freeze.txt` 记录实际恢复环境，不是经过验证的跨平台安装清单。当前不声称 `git clone` 后即可训练或评测。

## 当前实测结果与限制

2026-09-17，固定 w15 + SFT-v2 + v3-lowseg + target15_b 配置，在重建的 30 条历史评测样本上得到 14/30 accepted，目标 mean IoU 0.4014485360；历史对应值为 14/30、0.4013989366。

解释代码或撰写方法时，请保留以下事实：

- 执行器使用 GT IoU 更新最佳候选、接受与回退；策略输入过滤 IoU 不等于整个系统不依赖 GT。
- 30 个样本执行相同的 14 步序列，未验证内容自适应规划或提前停止。
- 该小样本集曾用于开发和退化标定；伪掩码本次重新生成，不是原掩码逐字节恢复。
- 本次恢复的是 SFT-v2；没有重跑 DPO，也没有证据表明 DPO 优于 SFT。

## 给 ChatGPT 的阅读任务

Cycle 001 已新增 [共享 bridge 更新](CHATGPT_CODEX_BRIDGE.md#codex-update-001)、[REF 状态流](research_log/MEMORY_STATE_FLOW.md)、[离线 memory 评测与 API 说明](research_log/COUNTERFACTUAL_MEMORY_EVAL.md)、[verifier 信号清单](research_log/ORACLE_FREE_VERIFIER_SIGNALS.md)。新增代码可用小型合成掩码在 CPU 上测试；它尚未接入原执行器，也没有新增模型性能结论。

[Cycle 002](research_log/CYCLE002.md) 增加冻结 CF 评测器的原始掩码导出、clean/target15_b 固定对照及 identity margin。代码已部署并通过 19 项 CPU 测试；固定 30 组的输入资产不完整，尚未产生真实退化指标。缺失清单和恢复后运行命令已记录。

[Cycle 003](research_log/CYCLE003.md) 已补齐同一批 30 组并完成真实对照：clean→target15_b 的 mIoU 为 0.9431→0.6311，CMSA 为 0.9667→0.4667，IER 均为 0。使用明确记录的重建伪掩码，不能视为历史标签精确恢复；目前尚未分离一般分割退化与记忆外观损伤。查看[逐组结果及六组对照图](research_log/cycle003/RESULT.md)。

请先读方法说明，再沿上述文件核对引用信息、损失函数、增强器梯度路径和执行器状态流。分别评价已实现机制、实验支持程度与后续需要验证的主张。将建议与现有实现分开，不把 GT 辅助结果选择写成无标注自主判断，也不把固定轨迹写成已验证的自适应规划。
