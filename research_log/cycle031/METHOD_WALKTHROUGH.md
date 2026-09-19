# CMA 方法逐步教学：从两名矿工到所属安全帽

依据：仓库 `7763f3b`，冻结 w15 配置见 `../cycle028/W15_CONFIG_SNAPSHOT.json`。以下是静态代码核查，不是新推理或实验。路径均相对仓库根目录；行号对应此次源文件。先读本文件，再读 TOY_COUNTERFACTUAL_EXAMPLE.md。

## 1. 实际需求 → 思路

一张图有矿工 A/B 及其安全帽 A/B。“找到一个安全帽”不够：用户要的是**这个矿工的安全帽**。即使 mask 外观像安全帽，选错所属矿工仍是错误。给定相同图像和相同关系问题，只更换已定位的矿工参考，输出应跟着切换。

这里不是让模型自己找到矿工。输入已经给了矿工 mask、bbox 和从当前图像提取的 crop。解决的是“利用指定矿工状态选出关联安全帽”。对应 `eval_mr_ref_counterfactual_v0.py:202`（下文简称 E，完整路径 `cmllm_remote/scripts/`）。

| 动机 / 可能失败 | 对应实现 | 能声称到哪里 |
|---|---|---|
| 问题相同却应指向不同人 | crop+bbox注入输入REF | 身份条件进入语言路径；未独立证明这一路贡献多大 |
| 输出分割需要位置与形状信息 | mask+bbox几何与REF-associated表示融合SAM prompt | 显式提供worker空间条件；不是直接提供helmet答案 |
| REF表示可能缺少对象约束 | 参考miner mask辅助重建 | 训练监督设计；没有证明其单独提升性能 |
| 两个条件都输出显眼的同一顶帽子 | 同组correct-vs-wrong soft-IoU hinge | 显式身份间隔目标；仍需像素定位loss |
| 平均IoU隐藏身份切换失败 | 固定图/query的离线身份矩阵 | 可以诊断identity control；本身不是可微训练模块 |

这些是机制设计意图与实现对应，不能写成已完成的组件因果实验。锚点逐项见下文。

## 2. 先区分三个数量

- 一组只含一张原始图，大小为 H×W×3；H/W 随样本变化。
- 两个身份生成两条固定多轮 conversation，每条包含 miner 和 helmet 两个 SEG 输出，因此一组有四个 mask 行，顺序 `[miner A, helmet A, miner B, helmet B]`。
- REF 元数据按四个 mask 行组织，valid 为 `[0,1,0,1]`。只有两张真实参考 crop 参与输入侧编码。不是四个独立身份。

依据：E:276–337；`cmllm_remote/third_party/LISA/utils/dataset.py:30` 的 collate/offset。以下 L 指 `cmllm_remote/third_party/LISA/model/LISA.py`。

## 3. 推理路径与张量

| 步骤 | 做什么、为什么 | 可核验形状与来源 |
|---|---|---|
| 当前观察 | clean 或固定 target15_b 处理后得到 RGB；主图与参考使用相同条件像素 | H×W×3；E:226–239；`research_log/run_cma_cycle025.py:50` |
| 参考定位 | 读取 supplied miner mask 与 bbox；bbox 除以原图宽高归一化 | 每身份 mask H×W，bbox 4；E:73、242 |
| 参考 crop | mask 外置黑，再裁 bbox；补成正方形，内容放左上角，不是居中；再 CLIP 预处理 | 原始 crop 边长依 bbox；CLIP 输出 3×h_c×w_c，处理器配置未在本轮恢复，h_c/w_c 不猜；E:84 |
| 主图两路 | CLIP 供 LLM；SAM 路最长边 resize1024，再标准化/补边 | SAM 输入 3×1024×1024；CLIP 尺寸同样取决于处理器；E:312；loader:21 |
| 输入参考编码 | CLIP 视觉塔后经过 LLaVA mm_projector，随后沿 token 维平均 | 两个有效参考 `[2,P,4096] → [2,4096]`；编码前视觉宽度配置1024，投影后4096；L:764、llava_arch.py:102。P 本轮无运行时确认 |
| bbox 编码 | `Linear(4,4096) → GELU → Linear(4096,4096)` | `[2,4] → [2,4096]`；L:169 |
| REF 向量 | pooled crop+bbox 后 LayerNorm 和 Linear | `[2,4096]`；L:176、766；按 conversation REF 次序打包为 `[2,1,4096]`（本例每条1个REF）；L:780 |
| 输入注入 | 图像 token 展开后，找到实际 REF token 槽；`e_REF += 0.5*r` | LLM embedding `[2,T,4096]`；T 含图像展开与文本，样本相关；`llava/model/language_model/llava_llama.py:91`。gated_add 是固定标量，不是可靠性网络 |
| LLM forward | 同图复制到两条 conversation，直接 forward；读取最终 hidden | `[2,T,4096]`；L:359–420；不是采样生成的 agent 对话 |
| SEG 路投影 | `4096→4096→256`，按 SEG-associated 索引抽取 | 本组 `[4,256]`；L:125、423。SEG 使用移位 token mask，见下节 |
| REF-associated 路 | 独立 `ref_hidden_fcs` 投影，取每条最后一个匹配位置 | `[2,256]`，再映射到 `[4,256]` 的有效helmet行；L:429–472、902–917 |
| 几何路 | supplied mask 双线性缩小到16×16、flatten，与4维bbox拼接 | `[4,H,W]→[4,256]→[4,260]→[4,256]`；L:153、894–903 |
| SAM context | REF-associated 表示+几何向量，范数最多20，乘可学习scale和valid | `[4,256]`；scale配置初值0.2，当前学习值本轮未读取；L:918–924 |
| 融合输出 prompt | `SEG + context`，元素截断到[-50,50]，增加token维 | `[4,1,256]`；L:487–518。没有把矿工bbox作为SAM的原生box prompt输入；本runner的该可选参数为None |
| SAM 解码 | 主图 SAM 特征与上述prompt进入单mask decoder，再映射回原图 | 后处理 `[4,1,H,W]`，保存 `[4,H,W]`；L:522、557。SAM内部低分辨率特征的具体空间尺寸未由本轮本地源/config确认，不写成实测值 |
| 最终选择 | logits>0；只取第1、3行的helmet | `[2,H,W]` 二值mask；E:342–358。不按GT IoU挑选候选 |

loader 完整路径：`cmllm_remote/scripts/run_mcr_train_predictions.py:21`；llava_arch 完整路径：`cmllm_remote/third_party/LISA/model/llava/model/llava_arch.py:102`。数字4096/1024/256/16/0.5/20/50均有w15配置或Linear定义支持；以上批维是针对两身份例子从构造逻辑推导，未运行模型测量。

## 4. 重要索引细节：不是所有“REF hidden”都在同一位置

输入注入在真实REF槽。输出 context 路 `L:431` 使用 `input_ids[:,1:] == ref_token_idx`，补尾并前置255个False；在代码假定“前方单个图像token展开成256个patch”的布局下，它选择的是**REF之前一个位置**的hidden。SEG路 `L:346` 也采用这种next-token移位约定。

辅助重建则不同：`get_ref_token_embeddings` 在 `L:979` 明确 `shifted=False`，抽取真实REF位置经过 `text_hidden_fcs` 的表示（不是上述独立ref_hidden_fcs路径）。不能把两条路径画成同一个向量。

因果注意：对于标准因果注意力，REF之前的hidden不会读取随后REF槽中的注入向量。因此不能声称输出context中的这一hidden已直接包含该REF槽的crop注入信息。后面的SEG-associated位置可以利用此前注入信息，几何路也仍明确提供身份定位。此处是索引与因果结构的静态推论，未测量实际影响，也不据此改变冻结代码或推翻已有整系统结果。

## 5. 训练路径：多了什么监督？

构造器 `cmllm_remote/scripts/build_mr_ref_counterfactual_train.py:54` 将同图配对共享第二轮问题；dataset `cmllm_remote/third_party/LISA/utils/mr_ref_seg_dataset.py:209` 组织交替miner/helmet目标。确切w15历史训练manifest仍未知。

1. LLM token CE：L:613，训练时传labels，监督文本输出。
2. 主mask BCE/Dice：L:620–698，训练时监督miner/helmet像素；不是只训练排序。
3. 参考重建：真实REF位置的投影向量经SAM重建miner mask；L:474、565、670。有效参考BCE/Dice乘配置权重1.0。
4. 配对rank：helmet行 `1::2`，sigmoid概率与所有helmet训练目标形成soft-IoU矩阵；L:84、652。每行要求正确列高于最强错误列至少0.05；平均hinge后乘1.5；L:711。算例见同目录文件。
5. 总loss为CE+主mask loss+参考loss+rank loss；L:720。历史CE/BCE/Dice具体override未知，不能拿launcher默认值冒充。

**训练专属的是监督损失及其目标使用，不是辅助解码计算本身。** L:565–601 的辅助参考mask解码在 `if inference` 返回（L:606）之前，所以当前推理也计算它，但返回的主预测不包含该辅助mask，它不参与训练loss计算。不要称推理已经优化掉该分支。

## 6. 离线评分：推理之后才看安全帽标签

Cycle025先以 `read_targets=False` 构造，再把整个 `masks_list` 清零，hook核查全零且inference=True（runner:43、52、60）。miner参考仍留在独立ref字段。mask监督占位用于接口形状，不是helmet答案。

预测冻结后，离线 `cmllm_remote/scripts/eval_counterfactual_memory_fidelity.py:15、37、103` 用二值mask计算IoU矩阵、Fidelity、CMSA、IER、margin和mIoU。训练soft-IoU hinge不是这些离散指标；低rank loss不自动等于高CMSA。Cycle031没有执行该scorer。
