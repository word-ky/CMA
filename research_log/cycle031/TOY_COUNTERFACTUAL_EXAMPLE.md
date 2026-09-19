# 一个场景、三种行为：手算counterfactual rank

**全部数字是教学假设，不是实验结果、不是本轮scorer输出。** 同一张图只有矿工A/B及其helmet A/B。问题固定为“分割参考矿工的安全帽”。矩阵行是memory A/B下的预测，列是训练目标helmet A/B。

代码：`cmllm_remote/third_party/LISA/model/LISA.py:84` 对logits做sigmoid后计算soft IoU；`:652`取helmet `1::2`；`:667`计算hinge；`:711`归一化乘权重。w15配置margin m=0.05，rank weight=1.5。

`h_A=max(0,0.05-S_AA+S_AB)`；`h_B=max(0,0.05-S_BB+S_BA)`。
本两身份例子的rank项是 `1.5*(h_A+h_B)/2`（忽略代码分母的1e-8数值稳定项）。

| 行为 | 2×2 soft-IoU矩阵，行顺序A/B | A的hinge | B的hinge | mean hinge | 加权rank项 |
|---|---|---|---|---:|---:|
| 忽略memory，总选helmet A | `[[0.90,0.02],[0.90,0.02]]` | max(0,0.05−0.90+0.02)=0 | max(0,0.05−0.02+0.90)=0.93 | 0.465 | 0.6975 |
| 能区分身份，但定位差 | `[[0.20,0.05],[0.05,0.20]]` | max(0,0.05−0.20+0.05)=0 | max(0,0.05−0.20+0.05)=0 | 0 | 0 |
| 身份对且定位好 | `[[0.90,0.02],[0.02,0.90]]` | max(0,0.05−0.90+0.02)=0 | max(0,0.05−0.90+0.02)=0 | 0 | 0 |

第一种行为每次都输出一个像样的安全帽，但B的请求错了。即使平均看对角soft IoU，也有(0.90+0.02)/2=0.46，掩盖A好/B坏的结构。这里只是soft对角平均，**不是把它称作实际二值mIoU**。真实二值mIoU同样会把各身份平均，不能单独展示“换memory是否换目标”。若类别任务根本不指定所属矿工，总选A甚至可能符合那个较弱任务。

第二种和第三种rank loss都为0：rank只要求相对身份间隔，不强制覆盖完整helmet。它不能替代像素BCE/Dice；也不能从rank=0推导CMSA成功。普通分割loss会惩罚错目标，并非对身份完全无感；rank的作用是显式比较同组正确与错误身份目标，而不是宣称常规监督在数学上无法学身份。

实际离线指标必须先阈值化真实预测，再重新计算binary IoU；不能从这三张soft矩阵推算其Fidelity/CMSA/IER数值。CMSA另外要求同组两身份都正确且各自binary IoU≥0.5（`eval_counterfactual_memory_fidelity.py:37`）。
