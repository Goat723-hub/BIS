#!/usr/bin/env python3
"""智谱 GLM-4V 视觉审核(云 API 方案)

不用下载模型,直接调智谱的多模态 API 做「观感审核」。
输入:图片 + 绿源观感审核 prompt → 输出:文字判断
"""
import base64
import os

import requests

# 绿源观感审核标准(来自《内容质量检查表》)
AUDIT_PROMPT = (
    "你是绿源集团的内容安全审核员。请检查这张图片,判断是否存在以下观感违规:\n"
    "1. 性暗示 / 擦边动作 / 挑逗\n"
    "2. 窥视感镜头(对人物身体部位的长时间特写)\n"
    "3. 人物着装暴露\n"
    "4. 低俗、含性暗示的字幕或台词\n"
    "5. 令人不适的暧昧、低俗氛围\n\n"
    "请严格输出 JSON,格式:\n"
    '{"违规": "是/否", "维度": "性暗示/窥视感/着装暴露/低俗字幕/无", '
    '"问题描述": "...", "严重程度": "高/中/低"}'
)

API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL = "glm-4v-flash"  # 免费视觉模型


def _load_key():
    """从 .env 读 key(不依赖 python-dotenv,直接解析)"""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("ZHIPU_API_KEY="):
                    val = line.split("=", 1)[1].strip()
                    if val and "在这里" not in val:
                        return val
    return os.getenv("ZHIPU_API_KEY", "")


def audit_image(image_path, prompt=AUDIT_PROMPT):
    """调智谱 GLM-4V 审核一张图片,返回文字判断"""
    api_key = _load_key()
    if not api_key:
        raise SystemExit("❌ 未找到 ZHIPU_API_KEY,请先在 .env 里填你的 key")

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    resp = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        },
        timeout=90,
    )
    data = resp.json()
    if "choices" in data:
        return data["choices"][0]["message"]["content"]
    return f"❌ 调用失败: {data}"


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python3 zhipu_audit.py <图片路径>")
        sys.exit(1)
    print(audit_image(sys.argv[1]))
