# 教学流程：先指定谁，再找属于他的装备

```mermaid
flowchart TD
  I[同一条件图像 I + 固定关系问题 q] --> L[LLM forward：关系条件化，不是agent循环]
  M[切换外部给定矿工 memory A/B：mask+bbox+当前图crop] --> V[crop视觉token平均 + bbox编码]
  V --> R[4096维REF向量：实际REF槽加0.5倍]
  R --> L
  M --> G[mask缩至16×16 + bbox：260维输入转256维几何]
  X[共享主图与REF之前的token] --> H[共享pre-REF hidden：不读取后面的REF注入；投影256维]
  L --> S[SEG-associated prompt：256维]
  H --> C[几何融合context：norm cap20 × learned scale × valid]
  G --> C
  C --> P[加到SEG prompt，clamp正负50]
  S --> P
  I --> E[SAM图像编码]
  E --> D[SAM单mask解码与原图后处理]
  P --> D
  D --> O[helmet A/B logits；推理阈值大于0]
  O --> T[仅训练：sigmoid预测A/B × helmet训练目标A/B]
  T --> J[2×2 soft-IoU矩阵]
  J --> K[正确对角 vs 错误列：margin0.05 hinge，平均后乘1.5]
  L --> A[实际REF槽：text投影后辅助miner mask解码]
  A --> B[仅训练：参考mask BCE/Dice]
```

索引要点：H不是注入REF槽本身的hidden；辅助A采用未移位槽。当前代码即使推理也执行辅助解码，但只有训练计算其损失。图中“LLM关系条件化”是功能路径说明，不是独立测得的推理能力归因。主mask BCE/Dice与token CE同样属于训练，图为避免拥挤未单列。

Cycle032已用源码提取测试确认偏移，并对冻结50组确认A/B前缀相同。H属于共享前缀路径，R可以影响后续SEG路径；不能画出R→H的反向时间依赖。详细带边类型的实际因果图见 `../cycle032/CURRENT_W15_CAUSAL_GRAPH.md`。当前简述：REF条件化SEG语义 + worker几何prompt + counterfactual排序。

离线另做：冻结二值预测→读取评分目标→binary IoU→Fidelity/CMSA/IER，不把这些指标反馈给推理。代码锚点及shape依据全部见 METHOD_WALKTHROUGH.md；这不是新增网络或论文图。
