#!/usr/bin/env python3
"""视觉审核:多模态 LLM 审画面(观感审核层)

用 OmniVL-Guard-2B(专门的视觉安全审核模型)对抽出的帧做审核。
输入:图片 + 审核 prompt → 输出:文字判断(JSON)

对应论文/BIS 里的「观感审核」——擦边、窥视感、性暗示这类
"无法量化、只能靠视觉理解"的违规。
"""
from PIL import Image

# 审核 prompt(绿源观感审核标准,来自《内容质量检查表》)
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


class VisualAuditor:
    """加载多模态模型,对图片做观感审核"""

    def __init__(self, model_dir="omni_vl_guard_2b"):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        print("加载视觉审核模型...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_dir, torch_dtype="auto")
        self.processor = AutoProcessor.from_pretrained(model_dir)

    def audit(self, image_path):
        """审核一张图片,返回模型的文字判断"""
        from qwen_vl_utils import process_vision_info

        image = Image.open(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": AUDIT_PROMPT},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        output = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True, clean_up_tokenization=False,
        )[0]
        return output
