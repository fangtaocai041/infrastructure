#!/usr/bin/env python3
"""
中文生态学术语 NLP 集成 — HanLP + Jiagu + Synonyms

提供三项能力:
  1. 分词 + 词性标注 — 将中文文献摘要解析为结构化 token
  2. 命名实体识别 (NER) — 提取 物种名 / 地名 / 研究方法
  3. 同义词匹配 — "刀鲚" = "长江刀鱼" = "Coilia nasus"

依赖: pip install hanlp jiagu synonyms

用法:
    python infrastructure/chinese_nlp.py segment "刀鲚是长江三鲜之首..."
    python infrastructure/chinese_nlp.py ner "2024年长江安庆段刀鲚资源调查"
    python infrastructure/chinese_nlp.py synonym "刀鲚"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent

# ── 生态学术语自定义词典 ──
ECOLOGY_DICT: Dict[str, str] = {
    # 物种名
    "刀鲚": "SPECIES",
    "短颌鲚": "SPECIES",
    "凤鲚": "SPECIES",
    "鳤": "SPECIES",
    "珠星三块鱼": "SPECIES",
    "江豚": "SPECIES",
    "中华鲟": "SPECIES",
    "白鲟": "SPECIES",
    "胭脂鱼": "SPECIES",
    # 生态学术语
    "洄游": "ECO_PROCESS",
    "产卵场": "HABITAT",
    "索饵场": "HABITAT",
    "越冬场": "HABITAT",
    "耳石": "METHOD",
    "Sr/Ca比": "METHOD",
    "微化学": "METHOD",
    # 地名
    "长江": "LOCATION",
    "安庆段": "LOCATION",
    "鄱阳湖": "LOCATION",
    "洞庭湖": "LOCATION",
}


def segment(text: str) -> List[Tuple[str, str]]:
    """中文分词 + 词性标注。"""
    try:
        import jiagu
    except ImportError:
        print("⚠️  pip install jiagu")
        return []

    words = jiagu.segment(text)
    pos_tags = jiagu.pos(words)

    results = list(zip(words, pos_tags))
    for word, pos in results:
        print(f"  {word:12s} {pos}")
    return results


def ner(text: str) -> List[Dict[str, str]]:
    """命名实体识别 — 提取物种、地名、方法。"""
    try:
        import jiagu
    except ImportError:
        print("⚠️  pip install jiagu")
        return []

    words = jiagu.segment(text)
    # 基于自定义词典的规则匹配
    entities = []
    i = 0
    while i < len(words):
        word = words[i]
        # 检查多字词组合
        for n in range(3, 0, -1):
            combo = "".join(words[i:i + n])
            if combo in ECOLOGY_DICT:
                entities.append({"text": combo, "type": ECOLOGY_DICT[combo], "pos": i})
                i += n
                break
        else:
            if word in ECOLOGY_DICT:
                entities.append({"text": word, "type": ECOLOGY_DICT[word], "pos": i})
            i += 1

    for e in entities:
        print(f"  [{e['type']:12s}] {e['text']}")
    return entities


def synonym_search(word: str) -> List[str]:
    """查找生态学术语同义词。"""

    # 首先检查自定义映射 (不依赖 synonyms 包)
    CUSTOM_SYNONYMS = {
        "刀鲚": ["长江刀鱼", "Coilia nasus", "刀鱼", "长颌鲚"],
        "江豚": ["长江江豚", "Neophocaena asiaeorientalis", "江猪"],
        "鳤": ["Ochetobius elongatus", "鳤鱼"],
    }

    if word in CUSTOM_SYNONYMS:
        results = CUSTOM_SYNONYMS[word]
    else:
        try:
            from synonyms import nearby
        except ImportError:
            print("⚠️  pip install synonyms")
            return []
        results = nearby(word)[:10]

    print(f"  {word} → {', '.join(results)}")
    return list(results)


def main() -> int:
    parser = argparse.ArgumentParser(description="中文生态学 NLP")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("segment", help="分词").add_argument("text")
    sub.add_parser("ner", help="命名实体识别").add_argument("text")
    sub.add_parser("synonym", help="同义词搜索").add_argument("word")
    sub.add_parser("dict", help="显示自定义词典")

    args = parser.parse_args()

    if args.cmd == "segment":
        segment(args.text)
    elif args.cmd == "ner":
        ner(args.text)
    elif args.cmd == "synonym":
        synonym_search(args.word)
    elif args.cmd == "dict":
        for k, v in sorted(ECOLOGY_DICT.items()):
            print(f"  {k:12s} → {v}")
    else:
        # 默认演示
        print("═══ 中文生态学 NLP 演示 ═══\n")
        demo = "2024年长江安庆段刀鲚资源调查显示洄游群体数量回升"
        print(f"📝 原文: {demo}\n")
        print("── 分词 ──")
        segment(demo)
        print("\n── NER ──")
        ner(demo)
        print("\n── 同义词 ──")
        synonym_search("刀鲚")
    return 0


if __name__ == "__main__":
    sys.exit(main())
