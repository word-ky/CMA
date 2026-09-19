# THREE-CONTRIBUTION EXECUTION PLAN

## 1. Problem / Benchmark
Task: degraded identity-grounded relational perception.

Core intervention:
same scene + same query + switch miner identity memory -> associated helmet target must switch.

Benchmark unit: counterfactual identity pair/group rather than independent masks.

Final evidence: locked large-scale documented-disjoint set shared by every method.

## 2. CMA model
Current validated mechanisms:
- appearance crop + miner mask + bbox memory;
- REF-conditioned later relational/SEG semantics;
- explicit SAM-side mask/bbox geometry prompting;
- counterfactual identity ranking.

Cycle038:
- clean mIoU 88.28 -> 39.99 and CMSA 92 -> 4 when crop feature is zeroed;
- target15_b mIoU 67.53 -> 34.28 and CMSA 58 -> 0.

Meaning: frozen w15 strongly functionally depends on the crop channel beyond retained localization. This is not pure-semantic or retrained-necessity proof.

## 3. Degradation-Aware Agentic Memory
Agent manages memory state, not only output.

Loop:
ASSESS -> REPAIR/ENHANCE -> CMA -> WRITE CANDIDATE -> COMMIT/ROLLBACK.

Enhancement:
repair memory-observation mismatch, preferably local reference-memory/ROI enhancement before full-frame enhancement.

Rollback:
restore prior stable memory state and prevent unreliable updates from contaminating identity memory.

MCR:
may be reused as one identity-reliability feature, not as a universal localization-confidence oracle.

## Large-scale baseline policy
Headline results require multiple baselines on one frozen manifest.

Priority executable set:
CMA, SegLLM, RAS/ORES, plus at least one valid reasoning/reference method such as LISA or VRP-SAM/ProSAM.

Target >=200 counterfactual groups when available; otherwise all valid documented-disjoint groups. Full evaluation is preferred when practical.

## Order of execution
Cycle039: freeze large benchmark + baseline contracts + sequence audit.
Cycle040: implement/test transactional agentic-memory state + action-headroom audit.
Cycle041+: integrate first additional baseline(s) and run large-scale comparisons.
Then: agentic memory-repair/rollback pilot on locked system-level benchmark.
Paper writing remains paused until all three contributions have empirical support.
