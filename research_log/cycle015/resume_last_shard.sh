#!/usr/bin/env bash
set -euo pipefail
ROOT="$1"
DEST="$ROOT/shared/models/segllm_095e0637/all_data_checkpoint"
date -Is
curl -fsSL --connect-timeout 15 --max-time 600 -C - https://hf-mirror.com/Marlo-Z/SegLLM/resolve/095e0637fcba015a02c0686f67b848c79c3cc80b/all_data_checkpoint/model-00001-of-00003.safetensors -o "$DEST/model-00001-of-00003.safetensors"
echo "490e9a7c216bb8b257fd10cf63eefff98c288ccd647ede1379105644bfc6cb2f  $DEST/model-00001-of-00003.safetensors" | sha256sum -c -
echo "c38c4d9637b2b47b1f48e30033e575cf7c9ff694024f18633fa458120441b6f8  $DEST/model-00002-of-00003.safetensors" | sha256sum -c -
echo "5e67f5ab544a014dd6fe7c1a7982ea870b2447e926e3bb31f512ca02ae625cf7  $DEST/model-00003-of-00003.safetensors" | sha256sum -c -
date -Is
