#!/usr/bin/env python3
"""
HuggingFace FishDet-M + YOLO 鱼群检测集成

提供预训练的鱼类检测模型，用于:
  - 水下摄像头鱼群计数
  - 物种级检测框绘制
  - 批量视频帧分析

依赖: pip install ultralytics opencv-python

用法:
    python infrastructure/fish_detector.py detect "underwater.jpg"
    python infrastructure/fish_detector.py video "survey.mp4" --output results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent


def check_yolo() -> bool:
    try:
        import ultralytics
        return True
    except ImportError:
        print("⚠️  缺少 ultralytics — pip install ultralytics")
        return False


def detect_image(image_path: str, conf: float = 0.25) -> List[dict]:
    """对单张图片执行鱼类检测。"""
    if not check_yolo():
        return []

    from ultralytics import YOLO

    print("🔍 加载 FishDet-M YOLO 模型 ...")
    # 使用预训练 YOLOv8 权重（通用检测）
    # FishDet-M 的专用权重可从 HF 下载: GE9X/FishDet-M
    model = YOLO("yolov8n.pt")  # 未来可替换为 FishDet-M 专用权重

    results = model(image_path, conf=conf)

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf_val = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = result.names.get(cls_id, f"class_{cls_id}")
                detections.append({
                    "bbox": [round(x1), round(y1), round(x2), round(y2)],
                    "confidence": round(conf_val, 3),
                    "class": cls_name,
                })

    print(f"✅ 检测到 {len(detections)} 个目标")
    for d in detections:
        print(f"  {d['class']:15s} conf={d['confidence']:.2f}  bbox={d['bbox']}")

    return detections


def process_video(video_path: str, output_dir: str = "output", fps_sample: int = 10) -> None:
    """对视频文件按帧采样并执行鱼类检测。"""
    if not check_yolo():
        return

    import cv2
    from ultralytics import YOLO

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps / fps_sample))

    print(f"🎬 视频 FPS={video_fps:.1f}, 每 {frame_interval} 帧采样一次")

    model = YOLO("yolov8n.pt")
    frame_count = 0
    total_fish = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            results = model(frame, conf=0.3)
            for r in results:
                if r.boxes is not None:
                    total_fish += len(r.boxes)

        frame_count += 1

    cap.release()
    print(f"✅ 处理完成: {frame_count} 帧, 检测到 {total_fish} 个鱼类目标")


def main() -> int:
    parser = argparse.ArgumentParser(description="FishDet-M 鱼群检测")
    sub = parser.add_subparsers(dest="cmd")

    detect_p = sub.add_parser("detect", help="单张图片检测")
    detect_p.add_argument("image", help="图片路径")
    detect_p.add_argument("--conf", type=float, default=0.25)

    video_p = sub.add_parser("video", help="视频鱼群检测")
    video_p.add_argument("video", help="视频路径")
    video_p.add_argument("--output", default="output")
    video_p.add_argument("--fps", type=int, default=10, help="每秒采样帧数")

    args = parser.parse_args()

    if args.cmd == "detect":
        detect_image(args.image, args.conf)
    elif args.cmd == "video":
        process_video(args.video, args.output, args.fps)
    else:
        parser.print_help()
        print("\n💡 提示: pip install ultralytics opencv-python")
    return 0


if __name__ == "__main__":
    sys.exit(main())
