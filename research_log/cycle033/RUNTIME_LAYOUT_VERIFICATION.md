# 实际恢复环境的token/视觉布局核验

结果 **PASS**。`verify_runtime_layout.py` 在A6000恢复环境执行，仅加载配置与tokenizer；不加载模型权重、图片，不执行forward/训练/scorer。详见 `runtime_layout_receipt.json`，包含50组完整prompt/token证据及源码hash。

## 视觉配置

实际 `shared/models/clip-vit-large-patch14/config.json`：image_size=224，patch_size=14。视觉塔 `num_patches` 属性按 `(image_size//patch_size)**2` 计算，故P=256。预处理配置resize/crop=224。因此Cycle031原未确认的CLIP图像张量尺寸现可补为3×224×224，patch序列长度256；mm_projector后宽度4096。

视觉config SHA256 `8a09b467700c58138c29d53c605b34ebc69beaadd13274a8a2af8ad2c2f4032a`，预处理config SHA256 `910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd`。它们是本次恢复环境读取值，未将其描述成原始历史训练全环境证明。

## tokenizer与序列化

按照冻结loader：AutoTokenizer(use_fast=False, model_max_length=512, padding_side=right)，pad=unk，注册SEG/REF；使用恢复环境真实llava_v1 Conversation与实际 `build_multiround_conversation` 函数AST，并应用collate的 `<image>→<im_start><image><im_end>` 替换，再执行真实 `tokenizer_image_token` AST。

w15的config、added_tokens、special_tokens_map、tokenizer.model、tokenizer_config哈希全部与Cycle025 prediction freeze记录一致。REF ID=32001，SEG ID=32000；图像占位=-200。

首次脚本错误地假定单独编码`[REF]`只返回一个ID，断言失败。实际返回 `[29871,32001]`（SEG为`[29871,32000]`）。已改成调用冻结源码的 `get_added_token_id`，通过convert_tokens_to_ids取得添加token ID。这是审计脚本修正，生产代码没有改动。

## 一个实际冻结组的位置（0起算）

组 `cf_778571dc8020b5df`；原始长度106，image位置39，展开长度361。image在系统文字之后、相关REF/SEG之前；“前方单图”并非指一定在序列第0或第1位。

| 对象 | raw位置 | 展开真实位置 | 当前hidden抽取位置 |
|---|---:|---:|---:|
| REF | 72 | 327 | 输出REF分支326；辅助重建327 |
| 第一轮SEG | 66 | 321 | 320 |
| 第二轮SEG | 103 | 358 | 357 |

255恰好是本布局图像展开增加的token数。manual mask长度与展开长度一致，但其REF内容选择**有意按源码测试所见偏前一位**，不能把“偏移量符合runtime”误写成“输出REF已经对齐真实REF槽”。第二轮SEG抽取在REF之后，可以读取注入身份；第一轮SEG在REF之前。

50/50冻结组的A/B真实token序列完全相同，均满足单个图像占位在相关token前、P=256及同样的相对位置关系。`PREFIX_IDENTICAL_BEFORE_REF` 现在由真实tokenization确认，不再仅依靠相同消息的推论。没有比较hidden值或预测性能。

512是tokenizer设置；collate仅对训练分支执行`model_max_length-255`截断。本次冻结推理配置不走该训练截断，不能把512解释为已实测的统一展开后长度。数据来源仍是既有50组清单，没有新增选样。
