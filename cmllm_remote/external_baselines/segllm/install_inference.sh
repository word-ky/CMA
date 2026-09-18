#!/usr/bin/env bash
set -euo pipefail
ROOT="$1"
ENV="$ROOT/external_baselines/segllm-venv"
SRC="$ROOT/external_baselines/segllm"
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME="$ROOT/external_baselines/segllm-cuda121"
export PATH="$CUDA_HOME/bin:$ENV/bin:$PATH"
export CPATH="$(python -c 'import glob,sys; print(":".join(glob.glob(sys.prefix+"/lib/python3.10/site-packages/nvidia/*/include")))')"
export PIP_CACHE_DIR="$ROOT/shared/cache/segllm-pip"
export MAX_JOBS=4
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=http.version
export GIT_CONFIG_VALUE_0=HTTP/1.1
python -m pip install 'Cython==3.0.10'
echo "990f35a2acae6e6d609e6563d69504cbebfedce07a9679b3a51af78d45cad396  $ROOT/shared/source/sam_6fdee8.tgz" | sha256sum -c -
python -m pip install --no-build-isolation "$ROOT/shared/source/sam_6fdee8.tgz"
echo "cf1692e5df4b009082daddedc0147a1195979556f0c8d2ca1463dd7df408bdcb  $ROOT/shared/source/coco_772b85.tgz" | sha256sum -c -
python -m pip install --no-build-isolation "file://$ROOT/shared/source/coco_772b85.tgz#subdirectory=PythonAPI"
python -m pip install --no-build-isolation -r "$ROOT/research_log/cycle015/requirements-inference.txt"
cd "$SRC"
export PYTHONPATH="$SRC"
python -c 'from llava.train.inference_cli import build_conversation, inference; from llava.model.segmentator.hipie_utils import Preprocessor; print("Native inference imports passed")'
python -m pip freeze > "$ROOT/research_log/cycle015/environment_freeze.txt"
