# Runtime estimate — planning only

The complete frozen set requires 1,408 predictions per method. Cycle025 measured load-to-freeze durations for 200 predictions on the same A6000 environment: CMA 95.03 seconds and SegLLM 342.43 seconds (../cycle025/HANDOFF.md). Linear scaling by 352/50 gives approximately **11.15 minutes CMA** and **40.18 minutes SegLLM**, or **51.33 minutes sequentially**. These are estimates, not fresh timings; scaling also multiplies fixed loading overhead, and image shapes/storage/GPU sharing can change actual cost.

CMA previously recorded peak CUDA allocation 17,322,769,920 bytes (~16.13 GiB). A6000 has 48 GiB-class memory. No matched fresh SegLLM peak or baseline speed ranking is claimed here.

RAS released HF metadata reports about 30.14 GB repository storage and 15.07 billion safetensors parameters. A 48 GiB card is a feasibility candidate, not a demonstrated fit: proposals, activations, generation and caches require additional memory. Runtime/peak are UNKNOWN until an actual wiring pilot; account for proposer cost separately and in end-to-end cost.

LISA released HF metadata reports about 32.24 GB repository storage. Storage is not inference memory; runtime/peak are UNKNOWN. VRP-SAM/ProSAM runtime is UNKNOWN while the task/checkpoint blockers remain. This cycle downloaded only small official documents/metadata, not new baseline weights, and performed no baseline training or task inference.

SAM-B asset restoration plus prepared-image freeze ran from 01:11:53 to 01:17:30 (+08), about 5m37s, exit 0. This one-time label/input preparation is not CMA inference time and not a performance result.
