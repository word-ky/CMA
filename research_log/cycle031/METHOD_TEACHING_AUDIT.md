# Cycle031 自查

依据静态源码与w15配置；未加载权重、图像或模型。源版本7763f3b。

- 输入/crop/REF注入/LLM/几何/SAM/训练/离线指标都有具体文件和函数行号；source_receipt.json记录源码哈希。
- 数字维度来自config与Linear定义；本例行数来自builder/collate逻辑。CLIP处理尺寸、P、SAM低分辨率网格及学习后的scale未实测，明确未确认。输入token512上限不冒充展开后序列T。
- 监督目标与推理输入分离：helmet训练目标仅用于loss或离线评分；推理清零监督列表，supplied miner参考仍是允许输入。
- soft-IoU训练hinge与binary离线指标分开。三组2×2数字明确是同一场景教学假设；逐行手算，平均后乘1.5。未调用实际scorer，未把soft矩阵冒充binary结果。
- supplied mask/bbox及same-condition crop没有隐藏；没有宣称干净历史记忆、自主写入/追踪/回滚或成功agent。
- 输入REF位置、输出移位REF-associated位置与辅助未移位REF位置分开讲。对于因果模型，移位位置不直接读取后一个REF槽的向量，是静态推论而非新实验。保留冻结代码，不擅自修正索引或重跑。
- 辅助参考mask解码在inference返回前执行；仅其监督loss训练专属。教学没有把该计算误称为推理不存在。
- 上游精确diff未建立。本地恢复导入5678c17已经有REF扩展；不以third_party目录或首次导入当作原创证明。仅整系统结果获支持，无组件独立增益断言。
- 未修改任何论文、Related Work、title/abstract、格式或实验资产；不运行训练、推理、scoring、ablation、新baseline或Layer2。
- 仅检查文档diff、配置字段与源码锚点，未为文档变更新增测试框架。

Exactly one next recommendation: 用户沿两身份例子逐步核对REF注入、移位hidden抽取和rank损失，先澄清方法理解。
