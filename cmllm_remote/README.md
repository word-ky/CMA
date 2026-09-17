# CMLLM DsDPM66 Pipeline

This project builds a small, reproducible DsDPM 66 subset and generates
spatial-relation safety expressions with rule templates plus DeepSeek rewriting.

Runtime location on H100:

```text
/home/wjq/cmllm
```

The DeepSeek API key must be supplied through the process environment as
`DEEPSEEK_API_KEY`. Do not write it into this repository.

Default model:

```text
deepseek-v4-flash
```

Run:

```bash
bash scripts/run_pipeline.sh
```

Status:

```bash
bash scripts/check_status.sh
```

