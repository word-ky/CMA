# Cycle015 progress

- 2026-09-18 13:53 +08: reviewed b082c95, only pinned SegLLM four-trial supplied-memory smoke authorized. Deadline approximately 14:53 +08. Native prompt frozen before any score: `Segment the mining helmet worn by instance 1.[REF:1]`.
- Separate remote Python3.10 venv created under external_baselines/segllm-venv; CMA environment untouched.
- Direct HuggingFace connection timed out. hf-mirror config response reported exact pinned revision. Git clone pinned HEAD still downloading at 13:59; no upstream file modified.
- Setup run 20260918-135634-cma-cycle015-setup installed base torch2.3.1/torchvision0.18.1/numpy2.0 and exited1 at HEAD check because clone was unfinished. This is setup sequencing, not model failure. Resume only after clone completion; no inference has run.
- CUDA12.1 compiler exists in conda package cache (nvcc12.1.66); full headers/link environment still unverified.

- 14:07: source clone on server failed HTTP2 EOF. Local HTTP1.1 clone succeeded at the exact pinned commit; transferred full checkout as source archive and verified remote HEAD. No source patches.
- 14:08: resumed environment build run 20260918-140835-cma-cycle015-build. Built isolated CUDA prefix from cached nvcc12.1.66, cccl12.1.55, cudart-dev12.1.55; solver installed cudart12.4.127. Torch wheel runtime remains cu121. Native CUDA extension compilation not yet completed.
- Checkpoint sequential download was deliberately interrupted to parallelize three distinct shards; first parallel attempt failed HTTP416 on already-complete small metadata. Re-fetch metadata normally, resume weights; active run 20260918-140358-cma-cycle015-download-resume. No revision changed.
- Official SegLLM INSTALL GoogleDrive link returns404 locally and connection refused remotely. Official HIPIE README now links author HF KonstantinosKK/HIPIE; r50_parts same named released ResNet50 model pinned to2bde18e with SHA256 in ancillary_sources.json. Active ancillary download run20260918-140528-cma-cycle015-ancillary. This is a documented distribution endpoint change, not a SegLLM checkpoint switch.
- Frozen smoke_inputs.json contains only source image, miner mask/bbox, semantic/native prompt and opaque identity IDs; no helmet target fields. Source-only history map in external_baselines/segllm/README.md is not runtime evidence.

- 14:11 detectron2 compilation failed on missing cusparse.h; included installed NVIDIA wheel headers through CPATH, unchanged native code. Retry run20260918-141237-cma-cycle015-build-headers reached linking, failed -lcudart due cached cudart-dev12.1.55 symlink expecting12.1.55 but solver chose runtime12.4.127.
- 14:15 installed cached runtime12.1.105 matching Torch and repaired only isolated prefix libcudart.so -> libcudart.so.12. Retrying build run20260918-141501-cma-cycle015-build-link. Changed invocation of first native op from make.sh --user install to equivalent setup.py build install into venv (no user-site mutation); native source unchanged.

- 14:16 native detectron2 and both deformable extensions built/imported (build-link exit0). Official float/double CUDA forward checks both True, max absolute errors4.66e-10 and8.67e-19. These are operator health tests only, not segmentation results.
- 14:19 ancillary acquisition completed exit0; r50_parts SHA256 verified. BERT config/vocab/weights retrieved at pinned revision in ancillary_sources.json.
- 14:22 server SAM git fetch remained stalled; stopped only verified cycle015-runtime process group2474430. Downloaded official source archive at exact required6fdee8 commit locally, SHA256990f35a2...cad396, uploaded and installed via archive. Reattempt inference deps run20260918-142223-cma-cycle015-runtime-archive. No dependency/model revision substituted.
- Added load_native.py for native baseline load health (syntax passed; execution pending imports and checkpoint completion). No seeded-memory inference adapter is claimed ready. Review014 Tier A wording clarified in BASELINE_PROTOCOL.md without changing frozen inputs or metrics.

- Closeout heartbeat 15:01 +08: checkpoint final acquisition finished14:46:30 exit0, all3 shard hashes passed. No loader/forward occurred before14:53 deadline; do not extend the cycle. No active cycle015/SegLLM process found. This is incomplete execution, not a checkpoint failure.
- Runtime deps installed after exact source archive/Cython repairs. Missing upstream HIPIE datasets directory restored from official pinned HIPIE; native imports now pass. Actual native memory preprocessing passed, runtime history/injection/output mapping remain untested. Full environment freeze and provenance retained.
- Closeout: source syntax checks only for new loader/runner; no predictions, scores or inference resource measurements. Report and handoff written; awaiting review.
