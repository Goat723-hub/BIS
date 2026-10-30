#!/usr/bin/env python3
"""
视频审核系统(完整版)
用法: python3 moderate.py <视频文件路径>

流程: 抽帧 → OCR → 敏感词打分 → 阈值分档 → 输出审核报告
"""
import argparse
import os
import re
import tempfile

import cv2
from rapidocr_onnxruntime import RapidOCR

import blacklist
import semantic

# ==================== 审核规则(参考抖音 / 行业标准) ====================
# 分类敏感词库(黑词库):按违规大类 + 权重
SENSITIVE_CATEGORIES = {
    "涉政违法": {"weight": 100, "words": ["赌博", "博彩", "毒品", "枪支"]},
    "低俗色情": {"weight": 90,  "words": ["色情", "约炮", "看片", "裸聊"]},
    "虚假宣传": {"weight": 80,  "words": ["最", "第一", "国家级", "销量第一", "最佳", "最牛"]},
    "广告导流": {"weight": 60,  "words": ["加微信", "加V", "私聊", "代开发票"]},
}

# 白词库:命中即放行(即使含敏感词),防误杀
WHITELIST = ["微信支付", "官方微信", "微信公众号"]

# 组合关键词:多个词同时出现才判违规(如 "加"+"微信")
COMBINATIONS = [
    (["加", "微信"], "广告导流", 60),
    (["加", "V"], "广告导流", 60),
]

# 正则变体匹配:抓字面匹配抓不到的变体(谐音/拆字/空格/字母替换)
VARIANT_PATTERNS = [
    (r"加\s*[微薇威Vv芯]\s*信", "广告导流", 60),  # 加微信 / 加威信 / 加V信 / 加微芯
    (r"[微薇威Vv芯]\s*信", "广告导流", 60),        # 微信 / 威信 / V信(单独出现)
]

# 两级阈值(参考百度智能审核:确认阈值 + 疑似阈值)
CONFIRM_THRESHOLD = 80   # 分数 >= 80:确认违规,直接拦截
SUSPECT_THRESHOLD = 50   # 分数 >= 50:疑似违规,转人工;低于 50:放行

INTERVAL_SEC = 2.0       # 抽帧:每 2 秒抽一帧
SCENE_THRESHOLD = 30.0   # 抽帧:场景切换阈值
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


def score_texts(texts, semantic_matcher=None):
    """机审打分:白词豁免 → 黑词 → 组合 → 正则变体 → embedding 语义,取最高权重"""
    hits = []
    max_score = 0
    for f, text, conf in texts:
        # ① 白词豁免:只扣掉白词本身,不放行整条(剩余文本继续查黑词)
        cleaned = text
        for w in WHITELIST:
            cleaned = cleaned.replace(w, "")
        # ② 分类黑词匹配(对 cleaned 文本)
        for category, info in SENSITIVE_CATEGORIES.items():
            for word in info["words"]:
                if word in cleaned:
                    hits.append((f, text, word, category, info["weight"]))
                    max_score = max(max_score, info["weight"])
        # ③ 组合关键词匹配(对 cleaned 文本)
        for words, category, weight in COMBINATIONS:
            if all(w in cleaned for w in words):
                hits.append((f, text, "+".join(words), category, weight))
                max_score = max(max_score, weight)
        # ④ 正则变体匹配(对 cleaned 文本,抓谐音/拆字/空格)
        for pattern, category, weight in VARIANT_PATTERNS:
            if re.search(pattern, cleaned):
                hits.append((f, text, f"变体:{pattern}", category, weight))
                max_score = max(max_score, weight)
        # ⑤ embedding 语义匹配(抓字面/正则都抓不到的语义变体,权重有加成)
        if semantic_matcher and cleaned.strip():
            for seed_text, category, weight, sim in semantic_matcher.match(cleaned):
                hits.append((f, text, f"语义:{seed_text}(相似度{sim:.2f})", category, weight))
                max_score = max(max_score, weight)
    return max_score, hits


def judge(score):
    """两级阈值分档:确认违规(拦截)/ 疑似(转人工)/ 放行"""
    if score >= CONFIRM_THRESHOLD:
        return "拦截"
    if score >= SUSPECT_THRESHOLD:
        return "转人工(疑似)"
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

    # ④ 打分(含 embedding 语义检索)
    print("④ 打分中(含语义检索)...")
    try:
        matcher = semantic.SemanticMatcher()
    except Exception as e:
        print(f"   语义模型加载失败,跳过语义检索: {e}")
        matcher = None
    max_score, hits = score_texts(texts, matcher)

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
        for f, text, word, category, w in hits:
            print(f"    ⚠️ {f}: 「{text}」命中「{word}」[{category}] (权重 {w})")
    print(f"  最高风险 : {max_score}")
    print(f"  处置结论 : 【{action}】")
    print("=" * 44)


if __name__ == "__main__":
    main()
