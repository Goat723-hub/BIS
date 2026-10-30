#!/usr/bin/env python3
"""OCR 识别 + 敏感词判断
对抽出来的每一帧:OCR 提取画面文字 → 敏感词匹配 → 输出疑似违规
"""
import os
from rapidocr_onnxruntime import RapidOCR

FRAMES_DIR = "frames"
# 示例敏感词库(真实业务请替换成你们自己的词表)
SENSITIVE_WORDS = ["赌博", "博彩", "加微信", "看片", "代开发票", "色情", "私聊", "约炮"]

# 初始化 OCR 引擎(首次会加载模型,稍慢)
print("初始化 OCR 引擎...")
engine = RapidOCR()

files = sorted(os.listdir(FRAMES_DIR))
print(f"待处理帧数: {len(files)}\n")

hits = []
texts_all = []

for f in files:
    path = os.path.join(FRAMES_DIR, f)
    result, _ = engine(path)          # OCR 识别这一帧
    if result:
        for box, text, score in result:
            texts_all.append((f, text, score))
            # 机审:敏感词判断
            for w in SENSITIVE_WORDS:
                if w in text:
                    hits.append((f, text, w))

print("=== OCR 识别到的文字(全部) ===")
for f, text, score in texts_all:
    print(f"  {f}: 「{text}」 (置信度 {score:.2f})")

print("\n=== 疑似违规(命中敏感词) ===")
if hits:
    for f, text, w in hits:
        print(f"  ⚠️ {f}: 文字「{text}」命中敏感词「{w}」")
else:
    print("  未发现敏感词(该视频是正常预告片,符合预期)")
