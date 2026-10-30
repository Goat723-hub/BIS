#!/usr/bin/env python3
"""抽帧脚本:用两种策略从视频里抽帧
1) 等间隔抽帧(打底,防止漏)
2) 场景切换检测(画面变化剧烈就额外抽)
"""
import cv2
import os

VIDEO = "test_video.mp4"
OUT_DIR = "frames"
INTERVAL_SEC = 2.0        # 规则1:每隔 2 秒抽一帧
SCENE_THRESHOLD = 30.0    # 规则2:帧差均值超过该值,判定为场景切换

os.makedirs(OUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise SystemExit(f"无法打开视频: {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
interval_frames = int(fps * INTERVAL_SEC)

prev_gray = None
frame_idx = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    reason = None

    # 规则1:等间隔抽帧
    if frame_idx % interval_frames == 0:
        reason = "等间隔"

    # 规则2:场景切换检测(和上一帧比,差异够大就抽)
    if prev_gray is not None:
        diff = float(cv2.absdiff(gray, prev_gray).mean())
        if diff > SCENE_THRESHOLD:
            reason = f"场景切换(diff={diff:.1f})"

    if reason:
        out = os.path.join(OUT_DIR, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(out, frame)
        saved += 1

    prev_gray = gray
    frame_idx += 1

cap.release()
print(f"完成: {VIDEO}")
print(f"  - 总帧数: {frame_idx}, FPS: {fps:.2f}")
print(f"  - 抽取帧数: {saved}")
print(f"  - 保存目录: {OUT_DIR}/")
