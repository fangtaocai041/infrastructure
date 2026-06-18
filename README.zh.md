# infrastructure — 统一涌现检测引擎

> 将 Reasonix 生态系统中三个核心项目的涌现检测能力融合到一个可组合的代码库中。

## 模块

| 模块 | 能力 |
|:-----|:-----|
| **unified_emergence** | 实时 Z-score 监控 · D₁→D₃ 维度追踪 · 三层分析 (异常→突变→理论) · 自组织领域发现 |
| **fish_classifier** | HuggingFace 鱼类识别: 60fishmodel (60 种) / Fish-Vista (1900 种) / DINOv2 特征提取 |
| **chinese_nlp** | 中文生态学术 NLP: HanLP + Jiagu 分词/词性标注 · NER 实体识别 · 同义词匹配 |
| **fish_detector** | FishDet-M + YOLO 鱼群检测: 水下摄像头鱼群计数 · 物种级监测 · 视频帧批量分析 |

## 作用

infrastructure 不直接处理物种数据或执行文献搜索。它为所有项目提供共享的涌现检测和分析能力。任何 Pn agent 均可 `import infrastructure` 获得涌现感知。

## 测试

```bash
cd D:\Reasonix\infrastructure
python -m pytest tests -v
```
57 项测试覆盖涌现检测、分类器、NLP 和检测器模块。

## 依赖

- Python >= 3.10
- torch, transformers (可选, 用于 fish_classifier)
- opencv-python (可选, 用于 fish_detector)
- hanlp, jiagu (可选, 用于 chinese_nlp)
