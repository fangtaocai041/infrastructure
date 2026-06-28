"""
holland/internal_models.py — 内部模型原理 (Internal Models)
=============================================================
Holland 原理⑥: 主体构建环境预测模型, 通过反馈竞争进化。
"Internal models allow agents to anticipate the consequences
of their actions. Better models confer adaptive advantage."

工程实现:
  - 预测模型集合: 多模型并行竞争
  - 贝叶斯模型选择: 后验概率 → 优胜劣汰
  - 涌现检测: 新模型的预测精度超过旧模型 → 认知涌现

Holland 原文映射:
  "An internal model is a mechanism that mimics some aspect
   of the environment, allowing the agent to try out actions
   internally before acting."
   → 工程: 模型 = 假设函数 h(x) ≈ f(x_env)
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class InternalModel:
    """Holland 内部模型: 主体对环境的预测器。

    Holland: "An agent's internal model maps situations to
              predicted outcomes."
    """
    name: str
    predict_fn: Callable[[dict], float] = field(repr=False)
    accuracy: float = 0.0          # 历史预测精度
    usage_count: int = 0
    last_error: float = 0.0
    birth_step: int = 0

    def predict(self, inputs: dict) -> float:
        self.usage_count += 1
        return self.predict_fn(inputs)

    def update_accuracy(self, predicted: float, actual: float):
        """滑动平均更新精度。"""
        error = abs(predicted - actual) / max(abs(actual), 1e-9)
        self.last_error = error
        alpha = 1.0 / max(self.usage_count, 1)
        self.accuracy = (1 - alpha) * self.accuracy + alpha * (1.0 - min(error, 1.0))


class ModelCompetition:
    """Holland 模型竞争: 多预测模型并行, 优胜劣汰。

    Holland: "Competition among internal models is a key
              adaptive mechanism in CAS."
    """

    def __init__(self, max_models: int = 10, min_accuracy: float = 0.3):
        self.max_models = max_models
        self.min_accuracy = min_accuracy
        self.models: dict[str, InternalModel] = {}
        self._generation = 0
        self._fitness_history: dict[str, list[float]] = defaultdict(list)

    def register(self, model: InternalModel):
        """注册新模型。"""
        model.birth_step = self._generation
        self.models[model.name] = model

    def predict_best(self, inputs: dict) -> tuple[str, float]:
        """选择最优模型预测。

        Holland: "Better models drive out worse through selection."
        """
        if not self.models:
            return ("none", 0.0)

        best_name, best_model = max(
            self.models.items(), key=lambda x: x[1].accuracy
        )
        prediction = best_model.predict(inputs)
        return (best_name, prediction)

    def update_all(self, inputs: dict, actual: float):
        """用真实反馈更新所有模型。

        Holland: "Feedback from the environment is the ultimate
                  arbiter of model quality."
        """
        to_remove = []
        for name, model in self.models.items():
            predicted = model.predict(inputs)
            model.update_accuracy(predicted, actual)
            self._fitness_history[name].append(model.accuracy)

            if model.accuracy < self.min_accuracy and model.usage_count > 10:
                to_remove.append(name)

        for name in to_remove:
            del self.models[name]

        self._generation += 1

        # 修剪: 保留 top-k
        if len(self.models) > self.max_models:
            survivors = sorted(
                self.models.items(), key=lambda x: x[1].accuracy, reverse=True
            )[:self.max_models]
            self.models = {k: v for k, v in survivors}

    def check_cognitive_emergence(self) -> list[dict]:
        """检测认知涌现: 新模型精度持续超过旧模型。

        Holland: "When a new internal model outperforms existing ones,
                  a cognitive emergence has occurred."
        """
        emergences = []
        for name, history in self._fitness_history.items():
            if len(history) < 5:
                continue
            recent = history[-5:]
            old = history[:-5] if len(history) > 5 else history[:1]
            old_avg = sum(old) / max(len(old), 1)
            recent_avg = sum(recent) / len(recent)

            if recent_avg > old_avg * 1.5:  # 50% 精度提升
                emergences.append({
                    "type": "cognitive_emergence",
                    "model": name,
                    "accuracy_old": round(old_avg, 4),
                    "accuracy_new": round(recent_avg, 4),
                    "improvement": round(recent_avg / max(old_avg, 1e-9), 2),
                })

        return emergences

    def fitness_landscape(self) -> dict[str, list[float]]:
        """返回适应度景观: {model_name: [accuracy_history]}"""
        return dict(self._fitness_history)
