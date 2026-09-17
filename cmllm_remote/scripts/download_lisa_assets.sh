#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p models/LISA-7B-v1 models/LISA_Plus_7b models/sam outputs
echo "START $(date -Is)" > outputs/lisa_download.log

hf download xinlai/LISA-7B-v1 \
  --local-dir models/LISA-7B-v1 >> outputs/lisa_download.log 2>&1
echo "LISA_DONE $(date -Is)" >> outputs/lisa_download.log

hf download Senqiao/LISA_Plus_7b \
  --local-dir models/LISA_Plus_7b >> outputs/lisa_download.log 2>&1
echo "LISA_PLUS_DONE $(date -Is)" >> outputs/lisa_download.log

if [[ ! -f models/sam/sam_vit_h_4b8939.pth ]]; then
  aria2c -x8 -s8 -k1M \
    -d models/sam \
    -o sam_vit_h_4b8939.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
    >> outputs/lisa_download.log 2>&1
fi
echo "SAM_H_DONE $(date -Is)" >> outputs/lisa_download.log

echo "DONE" > outputs/lisa_download_done.txt
