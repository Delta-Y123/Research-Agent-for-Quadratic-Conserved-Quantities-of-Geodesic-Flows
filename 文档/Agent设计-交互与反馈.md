# Agent 设计：交互方式、反馈协议与决策规则

本文档回答一个问题：**这个研究 Agent 具体怎么用、怎么和它交互、它反馈什么、凭什么可信。**

一句话定位：

> Agent 不是"会做数学的模型"，而是**把确定性符号工具包装成"实验—诊断—决策"闭环的调度器**。
> 数学结论由证书（certificate）决定，不由模型决定。

---

## 1. 三层架构

| 层 | 名称 | 由什么实现 | 职责 | 是否含模型 |
|---|---|---|---|---|
| L1 | 工具层 | 纯 Python + SymPy | 生成方程、精确求解、回代验证 | 否 |
| L2 | 决策层 | 规则表（+ 可选 LLM 提议） | 读诊断签名 → 选下一个动作 | 可选 |
| L3 | 交互层 | CLI / JSON | 意图 → 计划；证据 → 报告 | 否 |

**关键约束：** L1 的每个函数都是纯函数，输入输出都可 JSON 序列化，可以脱离 Agent 单独运行。
现有 `程序/二维/`、`程序/四维/` 下的脚本就是 L1 的"手工调用示例"，同时充当回归测试。

---

## 2. L1：工具契约

每个工具返回的不是"成功/失败"，而是一个**诊断签名**（第 3 节）。

| 工具 | 输入 | 输出字段 |
|---|---|---|
| `build_background` | 度规 spec、参数值、坐标 | `g, g_inv, H0, Q0, known_invariants, signature` |
| `killing_search` | `g`、动量次数、系数基底、对称性 | `rank, nullity, kernel_basis, known_dim, gap` |
| `verify_killing` | `g`、候选 `K` | `pass, first_nonzero, residual_terms` |
| `perturb` | 度规族、展开阶数 | `h, H1, exact_or_truncated` |
| `first_order_solve` | `h` 基、`Q1` 基、`Q0`、`H0` | `solvable, allowed_dim, constraints, per_direction, obstruction` |
| `classify` | 参数轴、取值表 | `allowed_dim_by_value, transition_points` |
| `abstract_profile` | 具体 `f(r)`、替换为 `F(r)` | `identity_holds, holds_for_arbitrary_F` |
| `numeric_drift` | `K`、初始轨道 | `drift_curves, max_drift, integrator_error` |

设计要点：

- **`killing_search` 返回 `gap = nullity − known_dim`**，而不是只返回 nullity。`gap` 才是"有没有新东西"的信号。
- **`first_order_solve` 返回 `per_direction`**（每个扰动方向的 ALLOWED/EXCLUDED），而不是只返回"有解/无解"。这就是"分类"。
- **`verify_killing` 与 `killing_search` 是两条独立路径**（泊松括号 vs Killing 方程），互相核对。

---

## 3. 诊断签名：反馈的核心

Agent 读的不是自然语言，而是这个结构：

```json
{
  "kind": "existence | nonexistence_in_span | classification | artifact | inconclusive",
  "rank": 600,
  "nullity": 9,
  "known_dim": 5,
  "gap": 4,
  "obstruction": {
    "momentum_power": [0, 1, 0, 2],
    "metric_component": "h_ru",
    "left_null_basis_dim": 12
  },
  "certificate": {
    "type": "symbolic_identity | left_null_witness | grid_convergence",
    "verified": true,
    "grid_history": [{"points": 16, "kernel": 12, "artifacts": 3},
                     {"points": 30, "kernel": 9,  "artifacts": 0}]
  },
  "scope": {"M": 1, "a": "1/2", "basis": "16-dim h, 138-dim Q1"},
  "cost": {"seconds": 72.8, "columns": 154}
}
```

`kind` 的五个取值直接对应五种后续动作，这是整个设计的枢纽：

| `kind` | 含义 | 后续 |
|---|---|---|
| `existence` | 找到并验证了 `Q1` | 做结构解释（R5） |
| `nonexistence_in_span` | 指定空间内无解，且有左零见证 | 报告阻碍 / 换基（R3） |
| `classification` | 得到允许方向子空间 | 输出分类表（R7） |
| `artifact` | 点值核向量回代失败 | 自动加密网格（R4） |
| `inconclusive` | 超时 / 化简失败 | 降规模或提问（R9） |

---

## 4. 决策表（策略）

把"研究直觉"写成规则。每条规则都对应本项目真实出现过的情况。

**R1 退化背景检测。** `gap > 0` 时，先检查背景是否退化。
> 实例：Kerr `a≠0` 时 nullity=2（`H`、Carter），`a=0` 时 nullity=3（多出 `L²`）。
> 若不检查，会把"对称性变大"误报成"发现新 Killing 张量"。

**R2 无新解。** `gap == 0` → 输出"该空间内无额外解"，并**明确标注这不是不存在性定理**。

**R3 阻碍有结构时换基。** `kind = nonexistence_in_span` 且 `obstruction` 集中在少数分量 → 依次尝试：换基底（幂 → 指数 / 有理函数）、放松对称性、改问"分类"而非"存在"。
> 实例：Schwarzschild ℓ=2 九参数扰动下 `Q0 = L²` 无法延续，约束为 `A1=A2=A3=B1=B3=C1=C2=C3=0` 且 `B1−B2/2=0`。阻碍集中在 `P₂` 的三个径向扇区，于是改用"分类"提问。

**R4 网格伪解。** `kind = artifact`（残差形如 `∏(r−rᵢ)` 的倍数）→ 自动加密点值网格重算。
> 实例：本次分类在 16 点网格上得到 12 维核，其中 3 个向量残差正比于 `(r−3)(r−4)(r−5)(r−6)`；加密到 30 点后核降到 9 维且全部通过全域回代。

**R5 正解必须做结构解释。** 找到 `Q1` 后，把具体剖面 `f(r)` 换成抽象 `F(r)`，检验恒等式是否对任意 `F` 成立。
> 实例：三个幂次正例 → 任意 `f(r)` 全部通过，原因是 Stäckel 加法分离 `2S_F H_F = R_F(r) + W(u)`。这一步把"3 个例子"升级为"一整族"。

**R6 参数分岔。** 参数扫描出现 rank 跳变 → 标记分岔点，单独分析。
> 实例：二维模型 `ε = 1` 与 `ε = −1/4` 使 nullity 从 2 降到 1，但两者机理不同（一个来自 `x²y²` 项破坏可分离性，一个是使 `λ` 退化的特殊值）。

**R7 分类优先于存在性。** 若 `kind = classification`，输出"允许方向子空间 + 约束矩阵 + 逐方向判定"，并区分**保持结构关系的方向**与**破坏结构关系的方向**。

**R8 规范自由度提示。** 任何 ALLOWED 判定都要检查是否可能只是纯坐标变换。若基内含纯规范方向，必须单独标注，不得计为新守恒量。

**R9 预算。** 单次工具调用超过预算（时间 / 列数）→ 自动降规模，并在报告中标注"结论强度下降"。

---

## 5. 交互协议

### 5.1 三种模式

**（a）目标模式**——用户给研究意图，Agent 返回计划，确认后执行。

```
> 我想知道 Kerr 的 Carter 常数，在破坏式(54)配合关系的扰动下还能不能延续

< 计划（4 步，预计 90 s，需要固定背景参数）
  [1] build_background(kerr, M=1, a=1/2)
  [2] perturb(eq54 族, f ∈ {1, r⁻¹, r⁻², r⁻³}) → 4 个保持配合关系的方向
      + 6 个度规分量 × {r⁻², r⁻³} → 12 个破坏配合关系的方向
  [3] first_order_solve(h 基 = 16 维, Q1 基 = 138 维)
  [4] 证书检查：网格自证 + 全域回代
  风险：16 点网格可能不足，会自动加密
  确认执行？
```

**（b）单步模式**——用户直接指定某个工具或某个实验，跳过决策层。

**（c）审查模式**——Agent 停下，把需要人判断的点列出来，给选项。

```
> 结果：16 个方向里只有 4 个可延续，全部是保持式(54)配合关系的方向；
> 12 个破坏配合关系的方向全部被排除，约束矩阵在它们上面是单位阵。

> 两个点需要你判断：
>  1. 是否放开 M=1, a=1/2？放开后失去精确有理数，改用参数化符号，成本约 +3 倍。
>  2. 是否把破坏项的径向剖面从 {r⁻², r⁻³} 扩到 {r⁻²…r⁻⁵}？
>     列数 154 → 178，预计 +40% 时间。
> 我的建议：先做 (2)，因为它只加强"排除"结论；(1) 留到有第二个背景参数需求时再做。
```

### 5.2 反馈的三条硬规则

1. **每条结论必须绑定证书**：符号恒等式 / 左零见证 / 网格收敛历史，三者至少一项。
2. **每条结论必须绑定适用范围**：`M`、`a`、`h` 基维数、`Q1` 基维数、一阶还是精确。
3. **不得把"有限空间内无解"表述为"不存在"**。措辞模板固定为
   "在声明的 N 维候选空间内无解，见证见证书 X"。

---

## 6. 反馈产物

```
runs/<run_id>/
  plan.json           # 本次实验的计划（可复现）
  ledger.jsonl        # 每行一条实验记录
  certificates/       # 每个结论的证书
  report.md           # 自动汇总
```

`ledger.jsonl` 的一行：

```json
{"step": 3,
 "hypothesis": "破坏式(54)配合关系的扰动仍可能延续 Carter 常数",
 "action": "first_order_solve",
 "params": {"h_basis": 16, "q1_basis": 138, "grid": "auto"},
 "signature": {"kind": "classification", "allowed_dim": 4, "gap": 4},
 "interpretation": "允许空间恰为保持配合关系的 4 维子空间；破坏项被单位阵约束全部压为零",
 "decision": "R7 → 输出分类表；R2 → 提示可扩基",
 "cost": {"seconds": 72.8},
 "certificate": "certificates/classification_a1_2.json"}
```

`report.md` 固定三栏：**已证 / 未证 / 仅在有限基内成立**。

---

## 7. 效果评估（课程要求的对比实验）

同一时间预算下比较三种策略：

| 策略 | 描述 |
|---|---|
| 固定流程 | 按次数 2 → 4 → 6 顺序扩基，扩到预算为止 |
| 人工策略 | 人预先指定扩基顺序 |
| Agent | 按第 4 节规则决定下一步 |

指标：

- 到达同一结论所需**工具调用次数**；
- **无效扩基次数**（扩基后 nullity 不变）；
- 是否自动发现**结构解释**（R5）与**网格伪解**（R4）——这两项是 Agent 相对固定流程最明显的增益；
- **错误结论数**（被证书推翻的结论）；
- **人工干预次数**。

预期结论：Agent 的优势不在"算得更快"，而在**知道什么时候该换问题**——
R3（阻碍有结构 → 改问分类）和 R5（正解 → 求结构原因）这两条，
固定流程不会自动触发。

---

## 8. 仓库改动建议

```
程序/
  内核/            # L1：纯函数 + JSON schema + 证书检查（由现有脚本重构而来）
    tools.py
    certificate.py
  agent/           # L2 + L3
    policy.py      # 第 4 节规则表
    ledger.py
    cli.py
  二维/ 四维/      # 保留：作为内核的手工调用示例与回归测试
文档/
  第一阶段成果汇报.md
  第二阶段成果汇报.md
  第三阶段-一阶扰动与分类.md      # 待补
  第四阶段-Agent设计与评估.md      # 本文档
```

现有脚本**不废弃**：它们是最可信的部分（每个都带精确回代），
重构只是把其中的 `check_cubic`、`clean`、`evaluation_matrix` 等提取成可复用工具。

---

## 9. 模型到底在哪、不在哪

| 环节 | 由谁做 | 理由 |
|---|---|---|
| 把研究意图翻译成工具调用序列 | LLM | 自然语言 → 结构化计划 |
| 判断等式是否为零 | SymPy 精确零判定 | 模型不可靠 |
| 读出"残差 ∝ ∏(r−rᵢ) 说明网格不够" | LLM 提议 + 规则确认 | 模式识别 |
| 决定"换基还是扩基" | 规则表 | 可审计 |
| 给出最终数学结论 | 证书 | 模型不参与 |
| 写研究报告的文字 | LLM | 但数字全部来自 ledger |

这条边界必须写进报告：**Agent 的可信度来自"它不能自己下结论"，而不是来自"它很聪明"。**

---

## 10. 实现状态（已完成）

框架已全部实现，代码在 [`程序/agent/`](../程序/agent/)，说明见
[程序/agent/README.md](../程序/agent/README.md)。

| 本文档的设计 | 实现位置 | 状态 |
|---|---|---|
| L1 工具层 | `engine.py` `backgrounds.py` `families.py` `solvers.py` | 完成，`selftest.py` 验收 |
| 工具契约与 JSON schema | `tools.py` | 完成，7 个工具 + 原生 function calling |
| 诊断签名 | `policy.diagnose` | 完成（5 类 + `info`） |
| 决策表 R1–R9 | `policy.suggest` / `policy.check_claims` | 完成（离线规划器 + 守门） |
| 三种交互模式 | `cli.py`（`list` / `tool` / `ask --mode llm\|rules`） | 完成 |
| 台账与报告 | `ledger.py`（`ledger.jsonl` + `report.md`） | 完成 |
| 模型接入 | `llm.py` + `transport.py`（同济 MiMo，OpenAI 兼容） | 完成，实测可用 |

### 实测：两次完整闭环

**（一）二维模型**（8 步，11 次工具调用，6.4 s）

模型自主完成：`list` → `inspect`（ε=0、ε=1）→ `killing_search`（秩 16/17）
→ `verify_quantity`（确认 $Q_0$ 在 ε=1 下不守恒，首个非零系数在 $p_y^3$）
→ `scan_parameter`（分岔点 ε=1）→ `finish`。

**（二）Kerr 一阶分类**（6 步，7 次工具调用，146 s，47.8k tokens）

模型自主完成：`list` → `inspect` → `killing_search`（gap=0，只有
$p_t^2,p_tp_\phi,p_\phi^2$）→ `first_order_classification(family="components")`
（24 个方向全排除，作负对照）→ `first_order_classification(family="eq54")`
（4/4 允许，作正对照）→ `finish`。

**守门确实起作用**：第一次 `finish` 把一条 `verified` 结论挂在
`killing_search` 的证据上，被策略层判为 `unsupported` 打回；
模型读完问题列表后自行改正 status 并重新提交，第二次通过。

### 与规则模式的对照

同一问题下，规则模式走 `list → inspect → killing_search →
first_order_classification → finish`，5 次调用、130 s，结论与 LLM 模式一致；
但它**不会**自己想到"先做 components 负对照、再做 eq54 正对照"
这种把方法本身也检验一遍的设计。这正是第 7 节所说
"Agent 的增益在于知道什么时候该换问题"的具体体现。

