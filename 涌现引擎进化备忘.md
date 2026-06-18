# 涌现引擎 — 数据投喂进化备忘

> 当你手上有真实数据时，先回答下面三个问题，再决定激活哪个模块。

---

## 一、数据类型 → 决定算法

| 你的数据长什么样？ | 需要用到的算法 | 对应模块 |
|---|---|---|
| 逐年/逐月时间序列（生物量、多样性、体长等） | Z-score 异常检测 + CUSUM 突变点 | `EmergenceEngine.detect_anomalies()` + `detect_change_points(method="cusum")` |
| 多变量表格（多个采样点的物种组成） | IQR 异常检测 + 理论模式匹配 | `detect_anomalies(method="iqr")` + `match_theory()` |
| 高维/稀疏数据（eDNA 数据、宏条形码） | 先降维，再用 Z-score | 需要 sklearn (PCA/UMAP) 预处理后接入 |
| 空间分布数据（不同湖泊/河段的对比） | 连通性效应、中度干扰模式 | `KNOWN_PATTERNS` 中对应理论 |
| 文本/文献元数据 | `emerge_domains()` 领域发现 | 自动聚类共现数据库 |

---

## 二、分析场景 → 决定理论模式

| 你想看什么？ | 激活的理论模式 | 需要提供的数据变量 |
|---|---|---|
| 禁捕效果评估 | 非对称恢复 (+ K策略者悖论) | `body_size_slope`, `diversity_slope` |
| 物种恢复模式 | 连通性效应 | `connected_lake_recovery`, `isolated_lake_recovery` |
| 污染物/水文影响 | 自然流态断裂 | `hydrologic_alteration`, `community_change` |
| 群落稳态转换 | 降维打击 | `state_transition_detected` (布尔/概率) |
| 干扰梯度效应 | 中度干扰 | `H_diversity`, `disturbance_level` |

---

## 三、介入阶段 → 决定在线/离线/混合

| 你什么时候用？ | 建议模式 | 使用哪个类 |
|---|---|---|
| 数据采集时实时看 | **在线监控** | `EmergenceMonitor` (record → check_emergence) |
| 数据到手后一次性分析 | **离线分析** | `EmergenceEngine` (scan 全部数据) |
| 先跑离线 → 再部署在线 | **混合模式** | 先用 `EmergenceEngine.scan()` 跑历史数据 → 用结果初始化 `EmergenceMonitor` 的阈值 |

---

## 四、将来投喂数据的步骤

```
1. 确定数据类型 → 查第一节，选算法
2. 确定分析场景 → 查第二节，选理论模式
3. 确定介入阶段 → 查第三节，选在线/离线
4. 调用 engine.scan(data=你的数据, species="物种名")
5. 看结果中有没有 theory_match 条目
6. 把新发现的模式追加到 KNOWN_PATTERNS (通过配置，无需改代码)
```
