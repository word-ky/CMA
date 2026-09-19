# 哪些复用、哪些改动、哪些只用于评估？

源版本7763f3b。这里的“CMA改动”表示本仓库恢复方法具有的扩展功能，不等于文献首创。精确原版LISA上游commit及完整逐行diff **not established**；本仓库首个导入提交5678c17已经包含REF扩展，不能用这个提交证明首次发明。当前只配置origin为CMA。

| Inherited/base behavior | CMA modification/addition in this repository | Evaluation-only machinery |
|---|---|---|
| LLaVA图像塔、mm_projector、语言模型与图像token展开；`llava_arch.py:102` | crop平均池化+bbox编码→4096维REF向量，REF槽固定0.5加法；`LISA.py:169、732`、`llava_llama.py:91` | 同一图与问题下切换两组 supplied identity；runner:50 |
| LISA式SEG投影、SAM图像特征和mask输出；`LISA.py:125、423、522` | 独立REF-associated投影+16×16 mask/bbox几何→256维context，add融合；`LISA.py:138、153、429、828` | 二值IoU矩阵与Fidelity/CMSA/IER；`eval_counterfactual_memory_fidelity.py:15、37、103` |
| 文本CE、mask BCE/Dice；`LISA.py:613–698`；属于已知基础训练范式，精确源码谱系未建立 | 真实REF位置辅助miner重建、valid reference loss；`LISA.py:565、670、968` | 先冻结预测再评分、输入读取/占位记录；`run_cma_cycle025.py:28、43、69` |
| 分割模型根据语言/视觉输出区域；关联对象能力也存在于SegLLM（见Cycle030来源） | 同图同query成组训练、helmet交叉soft-IoU rank，margin0.05、weight1.5；builder:54、dataset:209、`LISA.py:652` | 同图同条件系统级SegLLM比较；不能证明某模块单独带来增益 |

完整路径可由 METHOD_WALKTHROUGH.md 与 `../cycle028/METHOD_TRACEABILITY.md` 定位。w15配置以 `../cycle028/W15_CONFIG_SNAPSHOT.json` 为准。

本地git证据：`git log -- LISA.py`（使用其完整路径）包含5678c17恢复导入，之后ad34adc、127005c加入可选空间/局部特征实验。Cycle025未传spatial_boxes/local_evidence，不能将这些后续分支算入冻结方法。没有为本轮拉取新上游或实现任何baseline。

三点边界：1）文件在third_party中不代表所有行都是上游原样；2）文件在本仓库中不代表机制新颖；3）Cycle025只支持整系统性能，没有匹配组件消融来分摊贡献。
