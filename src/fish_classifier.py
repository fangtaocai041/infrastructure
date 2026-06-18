#!/usr/bin/env python3
"""
Hugging Face 鱼类识别集成 — Fish-Vista + 60fishmodel + dinov2

提供三个层级的鱼类图像分类能力:
  L1: 60fishmodel — 60 种常见鱼类快速分类（零配置，开箱即用）
  L2: Fish-Vista — 1900 种鱼类 + 9 种性状分割（需下载数据集）
  L3: DINOv2 — 自监督特征提取（适合少样本长江特有种）

用法:
    python infrastructure/fish_classifier.py classify "fish.jpg"
    python infrastructure/fish_classifier.py --model fishvista --download
    python infrastructure/fish_classifier.py --benchmark
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent

# ── 模型注册表 ──
MODELS = {
    "60fish": {
        "hf_id": "NeroZ02/60fishmodel",
        "type": "image-classification",
        "species": 60,
        "description": "60种常见鱼类分类器（含鲤科）— 一行代码可用",
        "install": ["transformers", "torch", "Pillow"],
    },
    "fishvista": {
        "hf_id": "imageomics/fish-vista",
        "type": "dataset+model",
        "species": 1900,
        "description": "60K图片 + 1900物种 + 9种性状分割",
        "install": ["datasets", "transformers", "torch", "Pillow"],
    },
    "dinov2": {
        "hf_id": "facebook/dinov2-small",
        "type": "feature-extraction",
        "description": "自监督视觉特征 — 适合少样本长江特有种",
        "install": ["transformers", "torch"],
    },
}


def check_install(packages: List[str]) -> bool:
    """检查包是否已安装。"""
    import importlib
    missing = []
    for pkg in packages:
        try:
            importlib.import_module(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"⚠️  缺少依赖: {', '.join(missing)}")
        print(f"   pip install {' '.join(missing)}")
        return False
    return True


def classify_60fish(image_path: str) -> List[Tuple[str, float]]:
    """
    使用 60fishmodel 对鱼类图片进行分类。
    返回: [(物种名, 置信度), ...] 前5个预测
    """
    if not check_install(MODELS["60fish"]["install"]):
        return []

    from PIL import Image
    from transformers import pipeline

    print(f"🔍 加载模型: {MODELS['60fish']['hf_id']} ...")
    classifier = pipeline("image-classification", model=MODELS["60fish"]["hf_id"])

    image = Image.open(image_path).convert("RGB")
    results = classifier(image, top_k=5)

    for rank, r in enumerate(results, 1):
        print(f"  {rank}. {r['label']:30s} {r['score']:.2%}")
    return [(r["label"], r["score"]) for r in results]


def extract_features_dinov2(image_path: str) -> Optional[List[float]]:
    """
    使用 DINOv2 提取图像特征向量（768维）。
    可用于少样本长江特有种的相似度匹配。
    """
    if not check_install(MODELS["dinov2"]["install"]):
        return None

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    print(f"🔍 加载模型: {MODELS['dinov2']['hf_id']} ...")
    processor = AutoImageProcessor.from_pretrained(MODELS["dinov2"]["hf_id"])
    model = AutoModel.from_pretrained(MODELS["dinov2"]["hf_id"])

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
    features = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

    print(f"✅ 提取特征: {len(features)} 维向量")
    print(f"   前5维: {[round(f, 4) for f in features[:5]]}...")
    return features


def download_fishvista() -> None:
    """下载 Fish-Vista 数据集。"""
    if not check_install(MODELS["fishvista"]["install"]):
        return
    from datasets import load_dataset
    print(f"📥 下载 {MODELS['fishvista']['hf_id']} ...")
    ds = load_dataset(MODELS["fishvista"]["hf_id"], trust_remote_code=True)
    print(f"✅ 下载完成: {ds}")


def benchmark() -> None:
    """运行简单的模型可用性基准测试。"""
    print("═══ HuggingFace 模型可用性检查 ═══\n")
    for name, info in MODELS.items():
        icon = "✅" if check_install(info["install"]) else "⚠️"
        print(f"  {icon} {name:12s} {info['hf_id']:40s} {info['description']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="HuggingFace 鱼类识别集成")
    sub = parser.add_subparsers(dest="cmd")

    classify_p = sub.add_parser("classify", help="分类鱼类图片")
    classify_p.add_argument("image", help="图片路径")
    classify_p.add_argument("--model", default="60fish", choices=["60fish", "dinov2"])

    sub.add_parser("download", help="下载 Fish-Vista 数据集")
    sub.add_parser("benchmark", help="检查模型可用性")
    sub.add_parser("list", help="列出可用模型")

    args = parser.parse_args()

    if args.cmd == "classify":
        if args.model == "60fish":
            classify_60fish(args.image)
        elif args.model == "dinov2":
            extract_features_dinov2(args.image)
    elif args.cmd == "download":
        download_fishvista()
    elif args.cmd == "benchmark":
        benchmark()
    elif args.cmd == "list":
        for name, info in MODELS.items():
            print(f"  {name:12s} {info['hf_id']:40s} ({info['species']}种)" if "species" in info else f"  {name:12s} {info['hf_id']}")
    else:
        parser.print_help()
        # 默认运行 benchmark
        benchmark()
    return 0


if __name__ == "__main__":
    sys.exit(main())
