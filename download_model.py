#!/usr/bin/env python3
"""下载 OmniVL-Guard-2B 视觉安全审核模型(走国内镜像)"""
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "SJJ0854/OmniVL-Guard-2B"
SAVE_DIR = "omni_vl_guard_2b"

print(f"开始下载模型 {MODEL_ID} ...")
print("(约 4GB,首次下载需要几分钟,请耐心等待)")

model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)

model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)

print(f"✅ 模型已下载并保存到 {SAVE_DIR}/")
