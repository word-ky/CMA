# CMA与SegLLM：同源输入不等于相同表示

冻结协议Cycle025；当前核查版本1e7b1e5。CMA=w15，SegLLM源码4593a069f09628ce3a5b46e657f5417fefd7be46、权重revision095e0637fcba015a02c0686f67b848c79c3cc80b。没有运行任一模型。

记号：**S**同一原始来源；**D**派生/序列化不同；**P**存在额外直接通道。E=`cmllm_remote/scripts/eval_mr_ref_counterfactual_v0.py`，L=`cmllm_remote/third_party/LISA/model/LISA.py`，B=`cmllm_remote/external_baselines/segllm/`。

| 信息 | CMA base-w15 | pinned SegLLM | 判断与锚点 |
|---|---|---|---|
| 当前RGB | 原图/target15_b经自身CLIP、SAM预处理 | 对应相同RGB经native CLIP/HIPIE预处理 | S+D；CMA runner:57逐像素断言，B/prepare_inputs.py:16，内部张量不声称相同 |
| 关系问题 | 原shared round2 query | 原语义记录保留，实际转成`Segment the mining helmet worn by instance 1.[REF:1]` | 同一目标语义，文本/模板D；不是两系统token字面相同；B/run_smoke.py:45–65 |
| supplied miner mask | 用于crop，并有独立下采样几何路径 | 用于native masked crop构造 | S+P；E:242、L:894；B/memory_state.py:19 |
| supplied miner bbox | 原图归一化后进入REF与输出几何两路 | native resize、整数crop边界后归一化bbox编码 | S+D；E:73、L:765、899；B/memory_state.py:23–34 |
| masked appearance crop | 原图mask外置黑、bbox裁剪、左上补方形、CLIP处理 | 先native图像/mask处理，再裁剪补方形、native CLIP处理 | S+D；E:84，B/memory_state.py:20–33；不宣称像素或特征一致 |
| 完整空间mask/下采样geometry | H×W supplied field缩16×16，与bbox拼接260维输入 | 没有独立supplied全幅/16×16 mask字段在crop后作为该推理接口输入 | P；L:894–924；B/seed_memory只返回appearance,bbox,receipt |
| bbox坐标表示 | `[x1/W,y1/H,x2/W,y2/H]` | native裁剪整数坐标`[y0,x0,y1,x1]/1024` | D；归一化、轴顺序、量化与空间不同 |
| token/history | 两条固定多轮conversation，真实REF注入；每条第二SEG选helmet | native history slot0，mask-encode+bbox-encode replacement；选择最后decode输出 | D；E:276、342；B/run_smoke.py:60–78、95–149 |
| helmet目标 | 推理read_targets=False且主监督占位全零 | preparation/runtime引用INFERENCE/NULL，无helmet输入 | 相同排除契约；CMA runner:43、52、60；B/run_smoke.py:60、125；旧freeze audit |

SegLLM没有单独的下游mask几何输入，并不表示它“完全没有空间信息”：bbox直接编码定位，masked crop也携带轮廓/背景置黑结构。CMA的crop同样依赖mask/bbox，不能称完全无定位的纯外观语义。

结论：同组提供对应的外部矿工定位/身份来源，支持native完整系统比较；派生表示、直接通道、训练经历均不匹配，无法将结果差距归因于单个语义memory机制。
