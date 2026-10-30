#!/usr/bin/env python3
"""黑库比对(相似度检索 · 第一层机审)

原理(pHash 感知哈希):
  图片 → 缩到 32x32 → DCT 变换 → 取低频 8x8 → 和均值比 → 生成 64 位 0/1 指纹

比对:汉明距离(两个指纹有多少位不同),距离越小越像
"""
import os

import cv2
import numpy as np

THRESHOLD = 10  # 汉明距离 <= 10 判为"同一内容"(0 = 完全相同)


def phash(img_path, hash_size=8):
    """算一张图的感知哈希指纹,返回 64 位 0/1 字符串"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (32, 32))       # 1. 缩到 32x32
    dct = cv2.dct(np.float32(img))        # 2. DCT 变换
    low = dct[:hash_size, :hash_size]     # 3. 取左上 8x8 低频
    avg = low.mean()                       # 4. 算均值
    bits = (low > avg).flatten()           # 5. 比均值,生成 0/1
    return ''.join('1' if b else '0' for b in bits)


def hamming(a, b):
    """汉明距离:两个指纹有多少位不同"""
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def build_blacklist(blacklist_dir):
    """把黑库目录里所有图片算好指纹,返回 {文件名: 指纹}"""
    db = {}
    if not os.path.isdir(blacklist_dir):
        return db
    for f in sorted(os.listdir(blacklist_dir)):
        fp = phash(os.path.join(blacklist_dir, f))
        if fp:
            db[f] = fp
    return db


def check_frame(frame_path, blacklist_db, threshold=THRESHOLD):
    """单帧和黑库比对,命中返回黑库文件名,否则 None"""
    fp = phash(frame_path)
    if fp is None:
        return None
    best_name, best_dist = None, 999
    for name, bfp in blacklist_db.items():
        d = hamming(fp, bfp)
        if d < best_dist:
            best_name, best_dist = name, d
    if best_dist <= threshold:
        return best_name
    return None
