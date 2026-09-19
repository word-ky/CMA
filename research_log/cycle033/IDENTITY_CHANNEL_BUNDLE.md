# A/B切换究竟改变了什么？

代码依据：`cmllm_remote/scripts/eval_mr_ref_counterfactual_v0.py:202`（E），`research_log/run_cma_cycle025.py:43–62`（R），`cmllm_remote/third_party/LISA/model/LISA.py`（L）。不是通道消融。

| 输入/字段 | A/B关系 | 首次进入计算的位置与作用 | 使用阶段 |
|---|---|---|---|
| 主RGB与条件/seed | 同组相同 | CLIP主图、SAM图像编码；共享场景 | 训练/推理 |
| 问题、角色、前缀及整个token序列 | 相同，50组真实token已核验 | 语言模型；前缀不编码pair_id | 训练/推理 |
| 矿工appearance crop | 随身份切换 | E:84先用该miner mask置黑、bbox裁剪；L:764视觉编码池化→REF输入 | 训练/推理 |
| REF输入bbox | 随身份切换 | L:169、765的4→4096 bbox MLP；与crop特征相加 | 训练/推理 |
| supplied miner mask | 随身份切换 | E:84同时改变crop内容；L:894的16×16几何编码 | 训练/推理；另作参考重建训练目标 |
| 输出几何bbox | 随身份切换，和输入bbox是同一定位来源 | L:899，与256维mask几何拼接成260维，再映射256维context | 训练/推理 |
| REF/SEG特殊ID、valid模式、模型权重 | 相同；每组valid=[0,1,0,1] | 控制位置/行选择；不是A/B语义输入 | 训练/推理 |
| masks_list监督占位 | 推理全零，A/B无目标差异 | R:60及hook核查；不用于答案选择 | 推理接口形状 |
| miner监督mask | 训练随身份不同；推理主监督列表清零 | 主miner BCE/Dice、辅助参考BCE/Dice | 训练；独立ref字段推理仍保留 |
| helmet目标mask | 随身份不同，但不作为推理输入 | L:644定位loss、L:652交叉soft-IoU；预测冻结后的独立scorer | 训练/离线评分 |

“变化”表示切换对应身份的输入来源，不保证每一维数值都不同。矿工mask影响crop与几何两个地方，bbox也走两个入口；不能把crop简单等同于无几何的纯外观语义。

最终helmet可受两条明确身份路径影响：crop+bbox→真实REF→后续SEG；mask+bbox→输出geometry context→SAM。pre-REF分量由相同前缀产生，不是当前A/B身份入口。几何是经MLP与prompt融合显式到达SAM，不是本冻结路径原生SAM box/mask prompt（L:513的boxes/masks为None）。

**可识别性结论：当前counterfactual实验识别的是联合提供的memory bundle，不是任何单一通道的独立因果贡献。** 没有匹配消融，不能判定外观必要、几何主导或各自解释多少Cycle025差距。
