#!/usr/bin/env bash
set -euo pipefail
ROOT="$1"
ENV="$ROOT/external_baselines/segllm-venv"
SRC="$ROOT/external_baselines/segllm"
export PIP_CACHE_DIR="$ROOT/shared/cache/segllm-pip"
export MAX_JOBS=4
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST=8.6
export CUDA_HOME="$ROOT/external_baselines/segllm-cuda121"
export PATH="$CUDA_HOME/bin:$ENV/bin:$PATH"
date -Is
python -m pip install --upgrade 'pip==24.2' 'setuptools==69.5.1' wheel
python -m pip install 'torch==2.3.1' 'torchvision==0.18.1' 'ninja==1.11.1.1' 'numpy==2.0.0'
export CPATH="$(python -c 'import glob,sys; print(":".join(glob.glob(sys.prefix+"/lib/python3.10/site-packages/nvidia/*/include")))')"
cd "$SRC"
test "$(git rev-parse HEAD)" = 4593a069f09628ce3a5b46e657f5417fefd7be46
python -m pip install --no-build-isolation -e ./uninext-segm
cd uninext-segm/projects/HIPIE/hipie/models/deformable_detr/ops
python setup.py build install
cd ../../maskdino/pixel_decoder/ops
python setup.py build install
python -c 'import torch, MultiScaleDeformableAttention; from detectron2 import _C; print("Native extensions imported", torch.__version__)'
