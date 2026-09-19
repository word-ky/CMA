# 已有Cycle025输入契约证据核对

运行 `audit_existing_receipts.py`，结果 **PASS_METADATA_ONLY**；只读JSON与元数据，不打开图片/权重/预测数组，不加载模型、不执行scorer。汇总及各源JSON SHA256见 `input_contract_check.json`。

## 同源与配对

- 50组、每方法200个(group,condition,identity)键完全对应。
- 对全部prepared记录检查source spec中的miner路径/hash/bbox一致；clean与target15_b的identity几何记录相同。
- SegLLM200条runtime记录的main RGB hash与prepared像素hash一致，supplied bbox与spec一致，实际注入appearance/bbox hash与构造hash一致。
- CMA旧freeze的`same_condition_pixels_as_segllm=true`来自runner实际像素比较，`zero_target_hook_passed=true`来自forward hook。本轮读取这些旧证据，没有重新打开图像复验。
- 退化改变主图、masked crop像素及appearance tensor；100个对应身份的native bbox hash在clean/degraded间固定、appearance hash变化。miner mask与bbox坐标保持固定，CMA的16×16几何也因此保持固定（源逻辑推导）。

## 目标是否在预测冻结前被读取？必须分阶段回答

**推理runtime没有读取helmet目标栅格**：CMA允许读取原图/miner参考，helmet主监督占位全零；SegLLM读取prepared图/miner参考，用INFERENCE/NULL decode槽。原scoring审计中两方法记录的runtime raster reads都包含在不含helmet目标的allowlist，本轮复核该集合关系。

**但资产恢复/准备阶段确实在预测之前接触helmet伪标签。** `protocol_receipt.json`明确`target_access_here=asset validity/hash freeze only`；伪mask恢复与结构检查不是推理。这份证据不支持“整个pipeline直到预测完成才第一次见到目标”的说法。

旧时间链：CMA预测冻结1789754448.7782187；SegLLM冻结1789754795.864617；scoring开始1789754800.283603。两个完整freeze都早于评分；源`research_log/score_cycle025.py`在检查冻结/输入读取后才打开评分目标。没有推理GT candidate selection或以GT挑选最终mask的记录。

## 精确引用

- `../cycle025/protocol_receipt.json` SHA256 `b1b234244a18a245a93ab50b95fb45a1e04d5b2cd4161c6b4740f4d15a8dce7a`
- `../cycle025/cma_prediction_freeze.json` SHA256 `d2c62fb9af4d005afb76b77913d1cd8da97adfa4c1415882702356bb7fd5339f`
- `../cycle025/segllm_prediction_freeze.json` SHA256 `4472a00291d8cab6440357a07f0ad857c312956dc4a6b7485ae43e0f8c771b40`
- `../cycle025/scoring/prediction_freeze_audit.json` SHA256 `3b6ef4197b275280835f3d14bfb32ecbae6b21baa8740cd1cd6d1870490387cf`

原`local_verification.json`已记录400个预测hash回放通过，本轮没有重复读取预测或计算任何指标。目标排除是上述已记录读取路径/冻结链范围内的结论，不是新增全系统追踪证明。伪标签质量、历史暴露未知与小样本限制保持不变。
