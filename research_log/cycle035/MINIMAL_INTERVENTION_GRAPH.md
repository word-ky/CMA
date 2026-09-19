# One fixed-localization intervention

Current computation (source: LISA.py:732–924; llava_llama.py:91; counterfactual evaluator:84,202):

```mermaid
flowchart TD
 I[Current full RGB] --> V[Full-image visual tokens and SAM image features]
 I --> C[Masked crop, bbox extraction, padding]
 M[Supplied miner mask] --> C
 B[Supplied miner bbox] --> C
 C --> E[CLIP plus mm_projector, mean pool: c]
 E --> X[Full: c / ablated: zeros_like c]
 B --> BB[REF bbox MLP: b]
 X --> SUM[c or zero + b]
 BB --> SUM
 SUM --> R[ref_input_fcs; true REF injection]
 R --> S[Later SEG semantics]
 V --> S
 M --> G[16x16 mask plus bbox geometry]
 B --> G
 P[Shared pre-REF context] --> G2[Output context addition]
 G --> G2
 S --> D[SAM prompt and mask decoder]
 G2 --> D
 V --> D
 D --> L[Training losses including counterfactual rank]
 L -. training pressure .-> R
 L -. training pressure .-> D
```

The only proposed difference is c -> 0 immediately after pooled crop features at LISA.py:764, before bbox addition. Keep crop construction and encoding executed for an otherwise identical computational path; do not replace crop pixels with a black image, which can yield nonzero encoder features. Preserve bbox MLP, REF projection, REF injection scale, full mask geometry, shared pre-REF output branch, auxiliary reconstruction and all losses. Apply intervention in both training and evaluation. No REF-index correction.

Fixed means paired raw RGB/query/mask/bbox/targets, preprocessing, architecture and starting parameters/settings. After learning, arm-specific parameters and derived hidden states may differ; holding final learned tensors equal would defeat the training comparison. Main-image appearance, bbox identity/localization and mask-shape geometry remain available in the zero arm. Removing crop features does not remove all appearance information or produce a pure bbox-only system.
