#!/usr/bin/env bash
set -euo pipefail
ROOT="$1"
DEST="$ROOT/shared/models/segllm_ancillary"
mkdir -p "$DEST/bert-base-uncased"
date -Is
curl -fsSL --connect-timeout 15 --max-time 1800 https://hf-mirror.com/KonstantinosKK/HIPIE/resolve/2bde18e63ba0db6afdd2e3d0ec90d3d53c3de6cc/r50_parts.pth -o "$DEST/r50_parts.pth"
echo "8c2e22a0cb2101cc1250552691b992b322be8a81c2c0a527da023b76c7d900a8  $DEST/r50_parts.pth" | sha256sum -c -
curl -fsSL --connect-timeout 15 --max-time 300 https://hf-mirror.com/google-bert/bert-base-uncased/resolve/86b5e0934494bd15c9632b12f734a8a67f723594/config.json -o "$DEST/bert-base-uncased/config.json"
curl -fsSL --connect-timeout 15 --max-time 300 https://hf-mirror.com/google-bert/bert-base-uncased/resolve/86b5e0934494bd15c9632b12f734a8a67f723594/pytorch_model.bin -o "$DEST/bert-base-uncased/pytorch_model.bin"
curl -fsSL --connect-timeout 15 --max-time 300 https://hf-mirror.com/google-bert/bert-base-uncased/resolve/86b5e0934494bd15c9632b12f734a8a67f723594/vocab.txt -o "$DEST/bert-base-uncased/vocab.txt"
date -Is
