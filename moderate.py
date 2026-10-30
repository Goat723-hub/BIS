#!/usr/bin/env python3
"""
视频审核系统(完整版)
用法: python3 moderate.py <视频文件路径>

流程: 抽帧 → OCR → 敏感词打分 → 阈值分档 → 输出审核报告
"""
import argparse
import os
import tempfile

import cv2
from rapidocr_onnxruntime import RapidOCR

import blacklist

# ==================== 配置区(集中管理,方便改) ====================
SENSITIVE_WORDS = {
    "赌博": 90, "博彩": 85, "色情": 90, "约炮": 80,
    "加微信": 50, "看片": 60, "代开发票": 70, "私聊": 40,
}
HIGH = 80          # 风险分 >= 80:直接拦截
LOW = 50           # 风险分 <= 50:放行;中间:转人工(灰色地带)
INTERVAL_SEC = 2.0     # 抽帧:每 2 秒抽一帧
SCENE_THRESHOLD = 30.0 # 抽帧:场景切换阈值
# ==================================================================


def extract_frames(video_path, out_dir):
    """抽帧:等间隔 + 场景切换检测"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = int(fps * INTERVAL_SEC)
    prev_gray = None
    idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        save = (idx % interval == 0)
        if prev_gray is not None and cv2.absdiff(gray, prev_gray).mean() > SCENE_THRESHOLD:
            save = True
        if save:
            cv2.imwrite(os.path.join(out_dir, f"frame_{idx:05d}.jpg"), frame)
            saved += 1
        prev_gray = gray
        idx += 1
    cap.release()
    return idx, saved, fps


def ocr_frames(frames_dir, engine):
    """OCR:遍历所有帧,提取画面文字"""
    texts = []
    for f in sorted(os.listdir(frames_dir)):
        result, _ = engine(os.path.join(frames_dir, f))
        if result:
            for box, text, score in result:
                texts.append((f, text, score))
    return texts


def score_texts(texts):
    """机审打分:命中敏感词,取最高权重"""
    hits = []
    max_score = 0
    for f, text, conf in texts:
        for word, weight in SENSITIVE_WORDS.items():
            if word in text:
                hits.append((f, text, word, weight))
                max_score = max(max_score, weight)
    return max_score, hits


def judge(score):
    """阈值分档:拦截 / 转人工 / 放行"""
    if score >= HIGH:
        return "拦截"
    if score > LOW:
        return "转人工(灰色地带)"
    return "放行"


def main():
    parser = argparse.ArgumentParser(description="视频审核系统")
    parser.add_argument("video", help="视频文件路径")
    args = parser.parse_args()

    video = args.video
    if not os.path.exists(video):
        raise SystemExit(f"视频不存在: {video}")

    # ① 抽帧
    frames_dir = tempfile.mkdtemp(prefix="moderation_")
    print("① 抽帧中...")
    total, saved, fps = extract_frames(video, frames_dir)
    print(f"   总帧 {total} 帧,抽出 {saved} 帧 (临时目录: {frames_dir})")

    # ② 黑库比对(第一层机审)
    print("② 黑库比对中...")
    blacklist_db = blacklist.build_blacklist("blacklist")
    blacklist_hits = []
    if blacklist_db:
        for f in sorted(os.listdir(frames_dir)):
            hit = blacklist.check_frame(os.path.join(frames_dir, f), blacklist_db)
            if hit:
                blacklist_hits.append((f, hit))
    print(f"   黑库条目 {len(blacklist_db)} 个,命中 {len(blacklist_hits)} 处")

    if blacklist_hits:
        print("\n" + "=" * 44)
        print("  视频审核报告")
        print("=" * 44)
        print(f"  视频文件 : {video}")
        print(f"  处置结论 : 【拦截】命中黑库(同源违规内容)")
        for f, name in blacklist_hits:
            print(f"    ⚠️ 帧 {f} 与黑库「{name}」相似")
        print("=" * 44)
        return

    # ③ OCR
    print("③ OCR 识别中...")
    engine = RapidOCR()
    texts = ocr_frames(frames_dir, engine)
    print(f"   识别到 {len(texts)} 条文字")

    # ④ 打分
    print("④ 打分中...")
    max_score, hits = score_texts(texts)

    # ⑤ 报告
    action = judge(max_score)
    print("\n" + "=" * 44)
    print("  视频审核报告")
    print("=" * 44)
    print(f"  视频文件 : {video}")
    print(f"  时长     : {total/fps:.1f} 秒")
    print(f"  抽帧数   : {saved}")
    print(f"  识别文字 : {len(texts)} 条")
    print(f"  命中敏感 : {len(hits)} 处")
    if hits:
        for f, text, word, w in hits:
            print(f"    ⚠️ {f}: 「{text}」命中「{word}」(权重 {w})")
    print(f"  最高风险 : {max_score}")
    print(f"  处置结论 : 【{action}】")
    print("=" * 44)


if __name__ == "__main__":
    main()
