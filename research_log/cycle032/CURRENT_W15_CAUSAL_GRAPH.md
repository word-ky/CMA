# 当前w15实际因果图

边标签：`D` = direct explicit input（代码显式传递/变换）；`C` = causally possible by decoder order（按因果decoder顺序允许，不等于已证明有效利用）；`T` = training-only（只在训练使用该监督关系）。每条边均标注。

```mermaid
flowchart TD
  V[参考appearance crop+bbox] -->|D| R[真实REF输入槽：e+0.5r]
  Q[共享主图+相同对话前缀] -->|C| H[pre-REF hidden：同前缀上下文]
  Q -->|C| U[真实REF位置hidden]
  R -->|C| U
  R -->|C| S[后续第二轮SEG-associated hidden]
  Q -->|C| S
  H -->|D| F[ref_hidden_fcs：256维共享前缀分量]
  G[当前矿工mask+bbox] -->|D| J[ref_visual_fcs：身份定位几何分量]
  F -->|D| C[相加、norm cap20、learned scale、valid]
  J -->|D| C
  S -->|D| P[SEG投影+context、clamp]
  C -->|D| P
  I[SAM主图特征] -->|D| M[SAM：helmet logits]
  P -->|D| M
  U -->|D| A[text_hidden_fcs+辅助SAM参考重建]
  A -->|T| B[参考miner mask BCE/Dice]
  YR[训练miner mask] -->|T| B
  M -->|T| K[helmet BCE/Dice与组内soft-IoU rank]
  Y[训练helmet目标A/B] -->|T| K
```

**不存在“真实REF输入→pre-REF hidden”的边。** 未移位辅助重建可读取真实REF槽，但它不是当前输出context中的 `ref_hidden_fcs` 向量。辅助解码计算本身仍在inference返回之前，所以U→A标D而不是T；A输出的监督loss才标T。

依据：输入注入 `llava_llama.py:91`；移位context抽取 `LISA.py:433`；SEG `:346`；几何 `:894`；融合 `:918、487`；辅助 `:474、565、979`；训练loss `:644、652、670`。完整路径与源哈希见 alignment_receipt.json。离线Fidelity/CMSA/IER没有连回模型；它们不是训练反馈，也不是推理控制器。

当前最准确的简述是：**REF条件化的后续SEG语义 + 显式矿工几何prompt + counterfactual身份排序训练**。另保留输出context中的共享pre-REF前缀分量。不能将该分量讲成对注入身份的第二次语义读取。
