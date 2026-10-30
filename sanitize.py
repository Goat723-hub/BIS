#!/usr/bin/env python3
"""审核 + 处置:识别违规字幕位置 → 打马赛克 → 输出合规视频"""
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

SRC = "violate_video.mp4"
OUT = "clean_video.mp4"

SENSITIVE_WORDS = ["加微信"]

# ① 先探一帧,OCR 定位违规文字区域
cap = cv2.VideoCapture(SRC)
ok, frame = cap.read()
fps = cap.get(cv2.CAP_PROP_FPS) or 25
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cv2.imwrite("/tmp/frame_probe.jpg", frame)

print("OCR 定位违规字幕中...")
engine = RapidOCR()
result, _ = engine("/tmp/frame_probe.jpg")

violate_boxes = []
if result:
    for box, text, score in result:
        if any(word in text for word in SENSITIVE_WORDS):
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))
            violate_boxes.append((x1, y1, x2, y2))
            print(f"  ⚠️ 命中「{text}」→ 位置 ({x1},{y1})~({x2},{y2})")

if not violate_boxes:
    print("  未识别到违规字幕,退出")
    cap.release()
    raise SystemExit(1)

# ② 重头读视频,对所有帧打马赛克
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

n = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    for x1, y1, x2, y2 in violate_boxes:
        roi = frame[y1:y2, x1:x2]
        roi = cv2.GaussianBlur(roi, (51, 51), 0)
        frame[y1:y2, x1:x2] = roi
    writer.write(frame)
    n += 1

cap.release()
writer.release()
print(f"✅ 合规视频已生成: {OUT} (打码 {len(violate_boxes)} 处, 共 {n} 帧)")
