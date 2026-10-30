#!/usr/bin/env python3
"""合成违规视频:在原始视频上叠加中文违规字幕(用于演示审核处置)
字幕内容为演示用示例敏感词,非真实违规素材。
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC = "test_video.mp4"
OUT = "violate_video.mp4"
FONT = "/System/Library/Fonts/STHeiti Medium.ttc"

# 违规字幕(演示用示例敏感词)
BOTTOM_TEXT = "加微信"

cap = cv2.VideoCapture(SRC)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

font_size = max(24, h // 8)
font = ImageFont.truetype(FONT, font_size)

writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

n = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    # 转 PIL 画中文(带半透明黑条背景,保证 OCR 好识别)
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    # 底部字幕条
    draw.rectangle([0, h - font_size - 40, w, h], fill=(0, 0, 0, 200))
    draw.text((20, h - font_size - 25), BOTTOM_TEXT, font=font, fill=(255, 255, 255, 255))
    img = Image.alpha_composite(img, overlay)
    writer.write(cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR))
    n += 1

cap.release()
writer.release()
print(f"✅ 违规视频已生成: {OUT} (共 {n} 帧, {w}x{h}, {fps:.0f}fps)")
