# 当前memory到底包含什么？

当前被验证的memory是 `M_i = (miner mask R_i, miner bbox b_i, crop C(I_condition,R_i,b_i))`。这是**外部提供、已经定位的矿工状态**，不是仅一个“A/B”文本ID。

Cycle032补充：crop+bbox在真实REF槽注入后可影响后续SEG语义，但当前输出ref_hidden_fcs读取pre-REF前缀，不直接含该槽注入身份；输出context的明确身份输入是mask/bbox几何。冻结50组A/B前缀相同，详见Cycle032对齐审计。

- crop：当前条件图像内，mask外置黑，bbox裁剪、左上补方形、CLIP预处理。退化图对应退化crop，不是另存的一张干净历史帧。依据 `cmllm_remote/scripts/eval_mr_ref_counterfactual_v0.py:84、233`。
- mask/bbox：告诉系统参考矿工在哪、形状如何；clean/degraded比较保留这些定位信息。不能说实验还验证了矿工mask受损后的自动恢复。依据同文件`:242、264、291`。
- crop与bbox经输入侧REF进入语言模型；mask/bbox还形成输出侧几何context。bbox不是本冻结runner直接传给SAM的box prompt。依据 `LISA.py:764、894、513`，完整路径见walkthrough。
- 输出虽然是另一对象helmet，worker定位依然缩小关系选择范围：例如区分两个近邻矿工各自头部装备。但有位置不等于已给出精确helmet轮廓，关系与像素预测仍需完成。这个解释是机制动机，不是已经消融证明的收益。
- 当前没有自主写memory、跨帧追踪、检索、置信度更新或回滚。固定输入调用结束后不会把输出重新写成下一步自选memory；依据 `research_log/run_cma_cycle025.py:50–69`。

## 哪些目标信息允许出现？

训练可以使用helmet目标算mask loss与rank；离线评分可以在预测冻结后读取它。推理若读取真实helmet mask构造特征、按helmet GT IoU选择候选/动作、或用其决定accept/rollback，就越过当前任务输入边界。

Cycle025先 `read_targets=False`，不读helmet raster；但构造器仍把已提供的miner mask放进主监督列表的miner行。因此runner还把整个 `masks_list` 清零，并由hook核查全零且inference=True（`research_log/run_cma_cycle025.py:43、52、60`）。独立的ref_masks/ref_bboxes继续保留，这正是明示允许的输入。清零占位不等于删除参考，也不是声称前期伪标签资产准备从未接触目标。

模型返回字段名gt_masks只是传入占位（`LISA.py:603–610`），本runner的真实helmet标签不在该字段里。Cycle025原有预测冻结/读取记录支持这一路径；本轮没有复跑推理。来源与标签仍是既有伪标签体系，50组不是完整公共benchmark，未知历史训练暴露也没有因本次代码讲解消失。
