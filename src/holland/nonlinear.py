"""
holland/nonlinear.py — 非线性原理 (Nonlinearity)
==================================================
Holland 原理③: 小扰动可以在 CAS 中产生不成比例的大效应。
"Nonlinearity means that the whole of the interactions
is greater than the sum of the parts."

工程实现:
  - 敏感度探针: 微小输入变化 → 输出变化幅度
  - Lyapunov 指数近似: 相邻轨迹的发散速度
  - 涌现阈值检测: 非线性跳变 = 相变信号

Holland 原文映射:
  "A hallmark of emergence is nonlinear interaction: small
   changes in one part of the system can cascade into large
   effects elsewhere."
   → 工程: 敏感度 = ∂Output/∂Input | 小扰动
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class SensitivityResult:
    """非线性敏感度分析结果。"""
    variable: str
    base_output: float
    perturbed_output: float
    sensitivity: float          # ∂Output/∂Input (归一化)
    is_emergent: bool = False   # 是否触发涌现阈值
    lyapunov_approx: float = 0.0  # 近似 Lyapunov 指数


class SensitivityAnalyzer:
    """全局敏感度分析器。

    Holland: "To understand a CAS, we must understand its
              nonlinearities — where and when small actions
              produce large effects."
    """

    def __init__(self, emergence_threshold: float = 2.0):
        """
        Args:
            emergence_threshold: 敏感度超过此值视为涌现
                (输出变化/输入变化 > threshold)
        """
        self.threshold = emergence_threshold
        self._history: list[SensitivityResult] = []

    def analyze(self, variable: str, base_input: float,
                perturbed_input: float,
                base_output: float, perturbed_output: float
                ) -> SensitivityResult:
        """分析单个变量的非线性敏感度。

        Hollands 涌现判据:
          |ΔOutput/ΔInput| / |base_output/base_input| > emergence_threshold
        """
        delta_in = max(abs(perturbed_input - base_input), 1e-9)
        delta_out = abs(perturbed_output - base_output)

        # 归一化敏感度
        norm_sensitivity = (delta_out / delta_in)
        if abs(base_output) > 1e-9:
            norm_sensitivity /= abs(base_output / max(base_input, 1e-9))

        # Lyapunov 近似: ln(|Δy|/|Δx|) / ln(|x|)
        lyap = 0.0
        if delta_in > 1e-9 and delta_out > 1e-9:
            lyap = math.log(delta_out / delta_in) / max(
                math.log(abs(base_input) + 1), 1.0
            )

        result = SensitivityResult(
            variable=variable,
            base_output=base_output,
            perturbed_output=perturbed_output,
            sensitivity=round(norm_sensitivity, 4),
            is_emergent=norm_sensitivity > self.threshold,
            lyapunov_approx=round(lyap, 4),
        )
        self._history.append(result)
        return result

    def emergent_variables(self) -> list[SensitivityResult]:
        """返回所有触发涌现的变量"""
        return [r for r in self._history if r.is_emergent]


class NonlinearProbe:
    """非线性探针: 对复杂系统的"轻推"实验。

    Holland: "Nonlinear interactions make it possible to
              explore CAS by probing — a small nudge reveals
              the structure of nonlinear couplings."

    算法: 对输入的每个维度施加 ε 扰动, 测量输出响应。
    高响应维度 → 非线性敏感 → 涌现候选。
    """

    def __init__(self, epsilon: float = 0.01):
        self.epsilon = epsilon
        self.analyzer = SensitivityAnalyzer()

    def probe(self, system_fn, input_data: dict[str, float]
              ) -> list[SensitivityResult]:
        """探针扫描: 逐个扰动输入维度, 收集敏感度。

        Args:
            system_fn: callable(input_dict) → output_dict
            input_data: 多变量输入

        Returns:
            敏感度结果列表, 按敏感度降序
        """
        base = system_fn(input_data)
        results = []

        for var, value in input_data.items():
            perturbed = dict(input_data)
            perturbed[var] = value * (1 + self.epsilon)

            output = system_fn(perturbed)

            for out_key, out_val in output.items():
                base_val = base.get(out_key, out_val)
                result = self.analyzer.analyze(
                    variable=f"{var}→{out_key}",
                    base_input=value,
                    perturbed_input=perturbed[var],
                    base_output=base_val,
                    perturbed_output=out_val,
                )
                results.append(result)

        results.sort(key=lambda r: r.sensitivity, reverse=True)
        return results
