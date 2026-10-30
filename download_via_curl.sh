#!/bin/bash
# 用 curl 从 hf-mirror 下载 OmniVL-Guard-2B 全部文件(绕过 huggingface_hub)
cd /Users/a1234/video_moderation_demo || exit 1
mkdir -p omni_vl_guard_2b
cd omni_vl_guard_2b || exit 1

BASE="https://hf-mirror.com/SJJ0854/OmniVL-Guard-2B/resolve/main"

FILES=(
  added_tokens.json
  chat_template.jinja
  config.json
  generation_config.json
  merges.txt
  model.safetensors
  preprocessor_config.json
  special_tokens_map.json
  tokenizer.json
  tokenizer_config.json
  video_preprocessor_config.json
  vocab.json
)

for f in "${FILES[@]}"; do
  echo "下载 $f ..."
  curl -sL -o "$f" "$BASE/$f" || echo "❌ $f 下载失败"
done

echo "===== 下载完成,文件清单 ====="
ls -lh
