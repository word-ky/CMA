# REF 对齐验证：不加载模型的确定性审计

源码版本33c8b72。运行 `python research_log/cycle032/audit_ref_alignment.py`；CPU PyTorch用于布尔索引，没有模型、权重、tokenizer、图片、训练或scorer。结果 PASS，详见 alignment_receipt.json。

## 方法与证据

脚本直接从源码AST提取并执行两个 `_build_expanded_token_mask`，不是重新手写它们。输出侧REF与SEG则执行 `LISA.py::model_forward` 中各自的三条原始赋值；唯一设备替换为 `.cuda()→.cpu()`。辅助重建核查 `get_ref_token_embeddings` 实际传入 `shifted=False`，并执行它使用的未移位mask helper。没有执行辅助SAM解码。

来源：`cmllm_remote/third_party/LISA/model/llava/model/language_model/llava_llama.py:55、91`；`cmllm_remote/third_party/LISA/model/LISA.py:346、433、926、968`。源文件哈希已写入receipt。

## 代表布局：位置从0开始

原始序列 `[BOS, IMAGE, text11, SEG, text12, REF, text13, SEG, EOS]`；其中普通text ID只是占位，REF=32001、SEG=32000来自w15配置，IMAGE=-200来自源码约定。这不是实际tokenizer输出。

IMAGE展开为256个patch后：BOS=0，image=1…256，text11=257，第一SEG=258，text12=259，REF=260，text13=261，第二SEG=262，EOS=263。

| 路径 | 实际选中位置 | 含义 |
|---|---:|---|
| 输入真实REF槽 | 260 | appearance+bbox向量注入位置 |
| 当前输出ref_hidden_fcs mask | 259 | REF之前一个hidden |
| 当前SEG mask | 257、261 | 各SEG前一位置；第二个在REF之后 |
| 辅助重建未移位REF mask | 260 | 真正REF槽 |

脚本断言 `output_REF == injected_REF-1`、两个未移位helper一致、SEG准确选中257/261。**偏移被证实，没有推翻Cycle031静态判断。** 因果语言模型的pre-REF状态不能读取稍后的REF向量；第二轮SEG-associated状态可以。SEG沿用next-token表征约定，不因此被判为同一问题。

## 结论依赖什么布局？

当前手工mask写死前置255，未逐位置展开图像，因此上述一位偏移成立的条件是：单个图像token在相关REF/SEG前、展开为256patch、REF等特殊token未被截掉，且token与hidden按这一路径对齐。并非对任意序列的定理。

| 限定性反例（非真实模型试验） | true REF | manual REF | 结果 |
|---|---:|---:|---|
| 单图但196patch | 200 | 259 | mask长度264与真实204不符 |
| IMAGE在REF后 | 2 | 256 | 不是pre-REF位置 |
| 两个IMAGE在REF前，各256patch | 514 | 258 | mask长度263与真实518不符 |

这三个仅用于界定断言适用条件；没有给生产代码新增兼容分支，也不暗示Cycle025实际使用它们。Cycle025源码的手工偏移假定256patch；本轮没有重新加载视觉塔实测P。

## A/B前缀结论：PREFIX_IDENTICAL_BEFORE_REF

执行实际 `build_multiround_conversation` 与 `build_item` 内conversation列表构造的AST，使用消息记录器保存序列化前的角色/正文。对 `research_log/cycle025/selected_source_groups.jsonl` 已冻结50组逐项检查，**50/50组的两条conversation消息完全相同**。没有读取图像或目标mask。示例消息结构为：

1. USER：`<image>` + 换行 + `Segment the miner whose helmet should be segmented next.`
2. ASSISTANT：`[SEG].`
3. USER：`[REF] ` + 该组的 `same_round2_query`
4. ASSISTANT：`[SEG].`

代码 `eval_mr_ref_counterfactual_v0.py:161、276` 对每个pair_id重复同样参数，没有把pair_id/矿工描述放进前缀。loader `run_mcr_train_predictions.py:25` 对所有行使用同一llava_v1模板和tokenizer；`LISA.py:359`复制同一主图CLIP输入。第一轮ASSISTANT是固定SEG文本，不是把A/B预测mask重新编码写入前缀。

所以同一确定性模板/tokenizer必产生相同A/B token序列，主图视觉前缀也相同。身份差异是独立REF embedding与输出几何字段，不是REF之前的文本。**本轮没加载实际tokenizer，也不声称拿到了运行时token IDs**；原始token相等由相同输入与同一确定性转换推得，特殊token位置用上面的源码测试验证。

在相同前缀输入、因果注意力及确定性eval的数学计算下，pre-REF hidden没有区分A/B的输入来源；本轮未比较实际BF16 hidden数值，不对设备舍入做额外等同性承诺。它可以包含共享场景/对话前缀语义，但不能归因为当前REF注入的身份语义。
