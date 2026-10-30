#!/usr/bin/env python3
"""机审打分 + 阈值分档
流程:OCR 提取文字 → 命中敏感词 → 累计风险分 → 按阈值分三档处置
(真实业务请换成自己的敏感词表、权重和阈值)
"""
import os
from rapidocr_onnxruntime import RapidOCR

FRAMES_DIR = "frames"

# 敏感词库 + 风险权重(权重越高越危险)
SENSITIVE_WORDS = {
    "赌博": 90, "博彩": 85, "色情": 90, "约炮": 80,
    "加微信": 50, "看片": 60, "代开发票": 70, "私聊": 40,
}

HIGH = 80   # 风险分 >= 80:直接拦截
LOW = 50    # 风险分 <= 50:放行;中间:转人工(灰色地带)

print("初始化 OCR 引擎...")
engine = RapidOCR()

files = sorted(os.listdir(FRAMES_DIR))
max_score = 0
hits = []

for f in files:
    result, _ = engine(os.path.join(FRAMES_DIR, f))
    if not result:
        continue
    for box, text, score in result:
        for word, weight in SENSITIVE_WORDS.items():
            if word in text:
                hits.append((f, text, word, weight))
                max_score = max(max_score, weight)

print("\n=== 命中详情 ===")
if hits:
    for f, text, word, w in hits:
        print(f"  ⚠️ {f}: 「{text}」命中「{word}」(权重 {w})")
else:
    print("  无敏感词命中")

print("\n=== 审核结论 ===")
print(f"视频最高风险分: {max_score}")
if max_score >= HIGH:
    print(f"处置:【拦截】风险分 {max_score} >= {HIGH},直接拦截")
elif max_score > LOW:
    print(f"处置:【转人工】风险分 {max_score} 落在 {LOW}~{HIGH} 灰色地带,交人工复核")
else:
    print(f"处置:【放行】风险分 {max_score} <= {LOW},正常放行")
