# 可训练路径：代码配方与历史事实分开

来源：`cmllm_remote/third_party/LISA/train_ds.py`（T）、`model/LISA.py`（L，位于同一LISA目录）、`cmllm_remote/scripts/run_lisa_finetune_mr_ref_v2b_rank.sh`（launcher）、Cycle028 w15 config、Cycle023历史日志摘录。源版本c251a1c；本轮不实例化模型，不读取训练权重或optimizer。

**精确历史逐参数requires_grad/optimizer快照未恢复，统一为UNKNOWN。** 下表回答现有历史训练入口规定的状态及证据强度，不把它冒充实测历史状态。merged checkpoint配置不保存原始requires_grad或完整LoRA训练配置；不能从merged模型推断整LLM全参数训练。

| 模块 | 文档化训练入口的设置 | 支持与历史限制 |
|---|---|---|
| LLM注意力适配 | LoRA默认r8、alpha16、dropout0.05、q_proj/v_proj；其余PEFT基座通常冻结 | T:47、106–108、220–261；launcher未显式覆盖这些默认。精确w15实际LoRA参数/PEFT状态UNKNOWN |
| token embedding、lm_head | 最后的命名循环显式requires_grad=True；没有仅训练REF行的行级限制 | T:273–292；精确实际optimizer成员UNKNOWN |
| CLIP vision tower | 显式冻结；恢复视觉wrapper forward还有no_grad | T:210–211；当前配置unfreeze_mm_vision_tower=false；原始运行快照UNKNOWN |
| mm_projector | 显式冻结 | T:212–213；w15 freeze_mm_mlp_adapter=true、tune_mm_mlp_adapter=false；无独立历史参数快照 |
| ref_input_bbox_fcs、ref_input_fcs | 显式可训练，不放入LoRA目标 | T:238–239、285–292；L:169–188 |
| 输入ref_input_scale | 固定float0.5，无可学习gate参数 | L:配置；llava_llama.py:129–135使用float；不能称可训练可靠性gate |
| text_hidden_fcs | 显式可训练 | T:282、292；L:125–137 |
| ref_hidden_fcs | 显式可训练，但输入是pre-REF共享上下文 | T:283、292；L:433；可训练不等于带身份 |
| ref_visual_fcs | 显式可训练 | T:284、292；L:153–164 |
| ref_embedding_scale | nn.Parameter，显式可训练；初始化0.2 | L:165–167、T:287、292；最终学习值未知 |
| SAM image encoder | 初始冻结；图像特征计算no_grad | L:118–120、295–304；不反传到其参数 |
| SAM prompt encoder | SAM整体冻结后未被最终命名列表解冻 | L:118–124、T:273–292；冻结参数不等于阻断对输入prompt的梯度 |
| SAM mask decoder | 解冻并train；最终命名循环再次可训练 | L:121–124、T:281、292；w15 train_mask_decoder=true |
| 辅助reference重建 | 复用text_hidden_fcs与SAM prompt/mask decoder，没有单独一个新ref-SAM decoder | L:474、565；与主分割共享参数；ref_hidden_fcs不是这条重建投影 |

launcher rank默认0.5，不是w15的1.5；w15配置与历史摘要支持1.5。类似地，不能把默认LoRA参数或CE/BCE/Dice系数直接称为已恢复的精确训练调用。历史摘录能支持分支/训练存在，不能补出缺失的逐参数状态。

模块会否获得非零梯度还取决于实际图连接、valid行、hinge是否激活、clamp及数据；此表只是可训练设置。下一文件分别说明连接，不报告测量的重要性或梯度值。
