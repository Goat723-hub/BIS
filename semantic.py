"""embedding 语义检索:抓字面/正则都抓不到的语义变体(谐音/近义)

原理:把「违规语义锚点」和待审文本都映射成向量,用余弦相似度判断
"意思像不像",而不是"字面像不像"。

- "菠菜" 能被 "博彩赌博" 锚点抓到(语义相同,字面完全不同)
- "加威信" 能被 "加微信" 锚点抓到(谐音)

权重加成:语义级判断比字面规则更可信,所以权重 = 字面权重 + 加成。
"""
import numpy as np

# 语义种子库:违规语义锚点(短词,精确)(种子文本, 违规类别, 字面权重)
SEMANTIC_SEEDS = [
    ("博彩赌博", "涉政违法", 100),
    ("下注", "涉政违法", 100),
    ("看片约炮", "低俗色情", 90),
    ("加微信", "广告导流", 60),
    ("私聊", "广告导流", 60),
]

# 语义相似度阈值(余弦相似度 > 阈值 判为语义命中)
SIM_THRESHOLD = 0.72

# 语义命中权重加成:语义级判断更可信,权重比字面规则更高
SEMANTIC_BONUS = 15


def _cosine(a, b):
    """余弦相似度"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class SemanticMatcher:
    """加载中文语义模型,把待审文本和违规锚点做语义相似度匹配"""

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        # 预计算所有锚点的向量(只算一次)
        self.seeds = []
        for seed_text, category, weight in SEMANTIC_SEEDS:
            vec = self.model.encode(seed_text)
            self.seeds.append((seed_text, vec, category, weight))

    def match(self, text):
        """返回命中的 [(种子文本, 类别, 加权后权重, 相似度)]"""
        text_vec = self.model.encode(text)
        hits = []
        for seed_text, seed_vec, category, weight in self.seeds:
            sim = _cosine(text_vec, seed_vec)
            if sim > SIM_THRESHOLD:
                hits.append((seed_text, category, weight + SEMANTIC_BONUS, sim))
        return hits
