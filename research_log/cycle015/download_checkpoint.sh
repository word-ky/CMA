#!/usr/bin/env bash
set -euo pipefail
ROOT="$1"
DEST="$ROOT/shared/models/segllm_095e0637/all_data_checkpoint"
mkdir -p "$DEST"
date -Is
pids=()
curl -fsSL --connect-timeout 15 --max-time 2100 "https://hf-mirror.com/Marlo-Z/SegLLM/resolve/095e0637fcba015a02c0686f67b848c79c3cc80b/all_data_checkpoint/added_tokens.json" -o "$DEST/added_tokens.json"
curl -fsSL --connect-timeout 15 --max-time 2100 "https://hf-mirror.com/Marlo-Z/SegLLM/resolve/095e0637fcba015a02c0686f67b848c79c3cc80b/all_data_checkpoint/config.json" -o "$DEST/config.json"
curl -fsSL --connect-timeout 15 --max-time 2100 "https://hf-mirror.com/Marlo-Z/SegLLM/resolve/095e0637fcba015a02c0686f67b848c79c3cc80b/all_data_checkpoint/generation_config.json" -o "$DEST/generation_config.json"
curl -fsSL --connect-timeout 15 --max-time 2100 -C - "https://hf-mirror.com/Marlo-Z/SegLLM/resolve/095e0637fcba015a02c0686f67b848c79c3cc80b/all_data_checkpoint/model.safetensors.index.json" -o "$DEST/model.safetensors.index.json"
curl -fsSL --connect-timeout 15 --max-time 2100 "https://hf-mirror.com/Marlo-Z/SegLLM/resolve/095e0637fcba015a02c0686f67b848c79c3cc80b/all_data_checkpoint/special_tokens_map.json" -o "$DEST/special_tokens_map.json"
curl -fsSL --connect-timeout 15 --max-time 2100 "https://hf-mirror.com/Marlo-Z/SegLLM/resolve/095e0637fcba015a02c0686f67b848c79c3cc80b/all_data_checkpoint/tokenizer.json" -o "$DEST/tokenizer.json"
curl -fsSL --connect-timeout 15 --max-time 2100 "https://hf-mirror.com/Marlo-Z/SegLLM/resolve/095e0637fcba015a02c0686f67b848c79c3cc80b/all_data_checkpoint/tokenizer.model" -o "$DEST/tokenizer.model"
curl -fsSL --connect-timeout 15 --max-time 2100 "https://hf-mirror.com/Marlo-Z/SegLLM/resolve/095e0637fcba015a02c0686f67b848c79c3cc80b/all_data_checkpoint/tokenizer_config.json" -o "$DEST/tokenizer_config.json"
curl -fsSL --connect-timeout 15 --max-time 2100 "https://hf-mirror.com/Marlo-Z/SegLLM/resolve/095e0637fcba015a02c0686f67b848c79c3cc80b/all_data_checkpoint/trainer_state.json" -o "$DEST/trainer_state.json"
curl -fsSL --connect-timeout 15 --max-time 2100 -C - "https://hf-mirror.com/Marlo-Z/SegLLM/resolve/095e0637fcba015a02c0686f67b848c79c3cc80b/all_data_checkpoint/model-00001-of-00003.safetensors" -o "$DEST/model-00001-of-00003.safetensors" &
pids+=("$!")
curl -fsSL --connect-timeout 15 --max-time 2100 -C - "https://hf-mirror.com/Marlo-Z/SegLLM/resolve/095e0637fcba015a02c0686f67b848c79c3cc80b/all_data_checkpoint/model-00002-of-00003.safetensors" -o "$DEST/model-00002-of-00003.safetensors" &
pids+=("$!")
curl -fsSL --connect-timeout 15 --max-time 2100 -C - "https://hf-mirror.com/Marlo-Z/SegLLM/resolve/095e0637fcba015a02c0686f67b848c79c3cc80b/all_data_checkpoint/model-00003-of-00003.safetensors" -o "$DEST/model-00003-of-00003.safetensors" &
pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done
test "$(stat -c%s "$DEST/added_tokens.json")" = 253
test "$(stat -c%s "$DEST/config.json")" = 1362
test "$(stat -c%s "$DEST/generation_config.json")" = 145
test "$(stat -c%s "$DEST/model-00001-of-00003.safetensors")" = 4939075464
echo "490e9a7c216bb8b257fd10cf63eefff98c288ccd647ede1379105644bfc6cb2f  $DEST/model-00001-of-00003.safetensors" | sha256sum -c -
test "$(stat -c%s "$DEST/model-00002-of-00003.safetensors")" = 4947390880
echo "c38c4d9637b2b47b1f48e30033e575cf7c9ff694024f18633fa458120441b6f8  $DEST/model-00002-of-00003.safetensors" | sha256sum -c -
test "$(stat -c%s "$DEST/model-00003-of-00003.safetensors")" = 4716774288
echo "5e67f5ab544a014dd6fe7c1a7982ea870b2447e926e3bb31f512ca02ae625cf7  $DEST/model-00003-of-00003.safetensors" | sha256sum -c -
test "$(stat -c%s "$DEST/model.safetensors.index.json")" = 187830
test "$(stat -c%s "$DEST/special_tokens_map.json")" = 438
test "$(stat -c%s "$DEST/tokenizer.json")" = 1844695
test "$(stat -c%s "$DEST/tokenizer.model")" = 499723
echo "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347  $DEST/tokenizer.model" | sha256sum -c -
test "$(stat -c%s "$DEST/tokenizer_config.json")" = 2897
test "$(stat -c%s "$DEST/trainer_state.json")" = 3163419
date -Is
