"""策略层：把"研究直觉"写成可审计的规则。

两个用途
--------
1. **诊断**：把工具返回的原始 JSON 压成五类 kind，并给出建议动作。
   这是 Agent 的"读结果"环节，也是离线（无 LLM）模式下的规划器。
2. **守门**：检查 LLM 提交的结论是否被证据覆盖、措辞是否越界。
   模型可以提议结论，但不能自己批准结论。
"""

from __future__ import annotations

import re

KINDS = ("existence", "nonexistence_in_span", "classification",
         "artifact", "inconclusive")

# 结论 status 与诊断 kind 是**两套**词表，不要混用：
#   kind   —— 工具内部对一次实验的判定（"existence"）
#   status —— 模型在 finish 里给结论打的标签（"existence_in_span"）
# finish 的 JSON schema 用的是后者，所以校验也必须用后者。
CLAIM_STATUSES = ("verified", "existence_in_span", "nonexistence_in_span",
                  "classification", "inconclusive")

OVERCLAIM_PATTERNS = [
    (re.compile(r"(证明|证实).{0,6}(不存在|没有)"), "不得声称证明了不存在"),
    (re.compile(r"\bno (killing tensor|conserved quantity) exists\b", re.I),
     "不得声称证明了不存在"),
    (re.compile(r"对所有.{0,8}(扰动|度量)都"), "不得把有限基结论推广到所有扰动"),
    (re.compile(r"(完全|彻底)(分类|解决)"), "有限基内不做完全分类"),
]

SCOPE_HINTS = ("基", "维", "M=", "a=", "span", "空间内", "有限")


def diagnose(result: dict) -> dict:
    """工具结果 -> (kind, 建议动作)。"""
    tool = result.get("_tool", "?")
    if result.get("error"):
        return {"kind": "inconclusive",
                "reason": f"{result.get('error')}: {result.get('message', '')}",
                "advice": ["检查参数拼写或降低计算规模后重试"]}

    if tool in ("list_backgrounds", "inspect_background"):
        return {"kind": "info", "reason": "背景信息已读取",
                "advice": ["接着做 killing_search 或 first_order_classification"]}

    if tool == "finish":
        return {"kind": "info",
                "reason": f"提交了 {len(result.get('claims') or [])} 条结论",
                "advice": ["由策略层检查证据与措辞"]}

    if tool == "killing_search":
        gap = result.get("gap", 0)
        nullity = result.get("nullity", 0)
        known = result.get("known_dim", 0)
        if gap > 0:
            return {"kind": "existence",
                    "reason": f"nullity={nullity} > known_dim={known}，存在超出已知不变量的解",
                    "advice": ["用 verify_quantity 独立回代每个新解",
                               "检查背景是否退化（例如 a=0 会额外产生 L^2）",
                               "若成立，尝试把结论抽象到一般参数"]}
        if nullity == known and nullity > 0:
            return {"kind": "nonexistence_in_span",
                    "reason": f"解空间 {nullity} 维，恰好由 {known} 个已知不变量张成",
                    "advice": ["提高 degree 或换函数基底再看",
                               "用 scan_parameter 找 nullity 跳变的特殊参数",
                               "注意：这只是在声明的候选空间内没有额外解"]}
        return {"kind": "inconclusive",
                "reason": f"nullity={nullity}, known_dim={known}，出现已知不变量无法表示的情形",
                "advice": ["用 inspect_background 核对已知不变量是否落在候选空间内"]}

    if tool == "verify_quantity":
        if result.get("poisson_zero"):
            return {"kind": "existence", "reason": "{Q,H}=0 精确成立",
                    "advice": ["该 Q 是精确守恒量，可记录为已验证"]}
        first = result.get("first_nonzero") or {}
        return {"kind": "nonexistence_in_span",
                "reason": f"{{Q,H}} 非零，首个非零系数在动量幂 {first.get('momentum_power')}",
                "advice": ["用 killing_search 看该背景下还剩几个独立守恒量",
                           "若 Q 只在某参数值守恒，用 scan_parameter 定位"]}

    if tool == "scan_parameter":
        changes = result.get("transition_indices") or []
        if changes:
            vals = [result["table"][i]["value"] for i in changes]
            return {"kind": "classification",
                    "reason": f"nullity 在 {vals} 处跳变（分岔点）",
                    "advice": ["对每个分岔点单独做 killing_search",
                               "检查分岔是否来自参数退化（如度规行列式为零）"]}
        return {"kind": "nonexistence_in_span",
                "reason": "扫描范围内 nullity 不变",
                "advice": ["扩大取值区间或提高 degree"]}

    if tool == "numeric_check":
        drift = result.get("rel_drift")
        carter = result.get("simulator_carter_rel_drift")
        if drift is None:
            return {"kind": "inconclusive", "reason": "数值检验没有返回漂移",
                    "advice": []}
        if drift < 1e-6:
            return {"kind": "existence",
                    "reason": f"数值轨道上 {result.get('label')} 相对漂移 "
                              f"{drift:.1e}，与符号结论一致",
                    "advice": ["可与符号恒等式并列作为第二条证据",
                               "注意：有限条轨道上近似不变不构成存在性证明"]}
        return {"kind": "nonexistence_in_span",
                "reason": f"数值漂移 {drift:.1e}"
                          + (f"（模拟器自身 Carter 漂移 {carter:.1e}）"
                             if carter is not None else ""),
                "advice": ["若符号层判为零，说明候选写错或积分精度不足",
                           "若候选本身就是故意扰动的，这正是预期结果"]}

    if tool == "first_order_classification":
        if result.get("artifacts"):
            return {"kind": "artifact",
                    "reason": f"{len(result['artifacts'])} 个点值核向量未通过全域回代",
                    "advice": ["加密网格重算（程序已自动加密到自证为止）"]}
        allowed = result.get("allowed_perturbation_dimension", 0)
        total = len(result.get("h_basis") or [])
        verdict = result.get("verdict") or {}
        n_ok = sum(1 for v in verdict.values() if v == "ALLOWED")
        if allowed == 0:
            return {"kind": "nonexistence_in_span",
                    "reason": f"{total} 个扰动方向全部被排除",
                    "advice": ["扩大 Q1 幂次范围后再看",
                               "确认增广权重是否覆盖已知正例"]}
        return {"kind": "classification",
                "reason": f"{n_ok}/{total} 个方向可延续，允许子空间 {allowed} 维",
                "advice": ["报告约束矩阵：它给出被排除方向的具体限制",
                           "区分'保持结构关系'与'破坏结构关系'的方向"]}

    return {"kind": "inconclusive", "reason": "未识别的工具", "advice": []}


# ------------------------------------------------------------------ 守门 ----
def check_claims(claims, ledger) -> list:
    """返回问题列表；空列表表示可以提交。"""
    problems = []
    steps = {entry.step: entry for entry in ledger.entries}
    if not claims:
        problems.append({"type": "empty", "message": "没有提交任何结论"})
        return problems

    for i, claim in enumerate(claims, 1):
        text = claim.get("text", "")
        status = claim.get("status", "")
        evidence = claim.get("evidence") or []
        scope = claim.get("scope", "")

        if status not in CLAIM_STATUSES:
            problems.append({"type": "status", "claim": i,
                             "message": f"status={status!r} 不在 {CLAIM_STATUSES} 中"})
        if not evidence:
            problems.append({"type": "no_evidence", "claim": i,
                             "message": "结论没有绑定任何工具调用"})
        missing = [e for e in evidence if e not in steps]
        if missing:
            problems.append({"type": "bad_evidence", "claim": i,
                             "message": f"引用了不存在的步骤 {missing}"})

        # 证据必须真的支持这个状态。
        # 除了看诊断 kind，还要看工具自己发的**证书**：例如 killing_search
        # 的 kind 是 nonexistence_in_span（该空间内没有额外解），但它解出来的
        # 每个 Q 都做过逐项回代，certificate.verified_all = True。此时断言
        # "这些解已被验证"是成立的，不该因为 kind 的名字被拒。
        supported = False
        for e in evidence:
            entry = steps.get(e)
            if not entry:
                continue
            kind = (entry.diagnosis or {}).get("kind")
            cert = (entry.result or {}).get("certificate") or {}
            certified = bool(cert.get("verified_all")
                             or cert.get("verified_all_kernel_vectors"))
            if status in ("verified", "existence_in_span") and kind == "existence":
                supported = True
            if status == "verified" and certified:
                supported = True
            if status == "nonexistence_in_span" and kind == "nonexistence_in_span":
                supported = True
            if status == "classification" and kind in ("classification", "existence"):
                supported = True
            # 分类器只要给出了逐方向 verdict，就是一次分类实验，
            # 哪怕结论恰好是"全部被排除"（此时 kind 记为 nonexistence_in_span）。
            if status == "classification" and (entry.result or {}).get("verdict") is not None:
                supported = True
            if status == "inconclusive":
                supported = True
        if evidence and not supported:
            problems.append({"type": "unsupported", "claim": i,
                             "message": "引用的步骤没有给出对应类型的证书"})

        for pattern, message in OVERCLAIM_PATTERNS:
            if pattern.search(text):
                problems.append({"type": "overclaim", "claim": i,
                                 "message": message, "text": text})

        if status in ("nonexistence_in_span", "classification") and not scope:
            problems.append({"type": "missing_scope", "claim": i,
                             "message": "这类结论必须写明适用范围（候选空间维数、固定参数）"})
        if status in ("nonexistence_in_span", "classification") and scope:
            # 具体的适用范围一定会写出某个数：维数、参数值、点数、网格……
            # 只认关键词会误伤 "ε=1"、"a=1/2" 这类合理写法，所以也接受数字。
            concrete = (any(hint in scope for hint in SCOPE_HINTS)
                        or re.search(r"\d", scope) is not None)
            if not concrete:
                problems.append({"type": "weak_scope", "claim": i,
                                 "message": f"适用范围 {scope!r} 太笼统，"
                                            "请写明候选空间维数或固定参数"})
    return problems


# ------------------------------------------------------------ 离线规划器 ----
def compose_claims(ledger) -> dict:
    """从台账直接拼出结论（规则模式与兜底用）。"""
    claims = []
    for entry in ledger.entries:
        tool, res, diag = entry.tool, entry.result or {}, entry.diagnosis or {}
        if res.get("error"):
            continue
        params = res.get("parameters") or {}
        ptext = ", ".join(f"{k}={v}" for k, v in params.items())

        if tool == "killing_search":
            gap = res.get("gap", 0)
            text = (f"{res['background']}（{ptext}）在 degree="
                    f"{res['ansatz']['degree']}、{res['unknowns']} 个未知系数的候选空间内："
                    f"秩 {res['rank']}，解空间 {res['nullity']} 维，"
                    f"其中已知不变量 {res['known_dim']} 维，gap={gap}。")
            if gap > 0:
                text += f" 存在 {gap} 个超出已知不变量的解。"
                status = "existence_in_span"
            elif gap == 0:
                text += " 没有超出已知不变量的解。"
                status = "nonexistence_in_span"
            else:
                text += (" 出现已知不变量无法在候选空间内表示的情形（gap<0），"
                         "该结果不可用于下结论。")
                status = "inconclusive"
            claims.append({"text": text, "status": status,
                           "evidence": [entry.step],
                           "scope": f"{res['unknowns']} 维候选空间，{ptext}，"
                                    f"系数次数 ≤ {res['ansatz']['degree']}"})

        elif tool == "verify_quantity":
            if res.get("poisson_zero"):
                claims.append({"text": f"Q 在 {res['background']}（{ptext}）下精确满足 {{Q,H}}=0。",
                               "status": "verified", "evidence": [entry.step],
                               "scope": f"{ptext}"})
            else:
                first = res.get("first_nonzero") or {}
                claims.append({
                    "text": f"Q 在 {res['background']}（{ptext}）下不守恒："
                            f"{{Q,H}} 在动量幂 {first.get('momentum_power')} 处非零。",
                    "status": "nonexistence_in_span", "evidence": [entry.step],
                    "scope": f"{ptext}；给定的单个 Q"})

        elif tool == "scan_parameter":
            table = res.get("table") or []
            changes = res.get("transition_indices") or []
            text = (f"扫描 {res['background']} 的 {res['parameter']}："
                    + "，".join(f"{r['value']}→nullity {r['nullity']}" for r in table))
            if changes:
                text += (f"。分岔点：{res['parameter']}="
                         + ", ".join(table[i]["value"] for i in changes))
            claims.append({"text": text, "status": "classification",
                           "evidence": [entry.step],
                           "scope": f"{len(table)} 个参数取值，degree=2 候选空间"})

        elif tool == "first_order_classification":
            verdict = res.get("verdict") or {}
            ok = [k for k, v in verdict.items() if v == "ALLOWED"]
            bad = [k for k, v in verdict.items() if v == "EXCLUDED"]
            text = (f"一阶分类（{res['background']}，{ptext}，Q0={res['q0']}）："
                    f"{len(verdict)} 个扰动方向中 {len(ok)} 个可延续、{len(bad)} 个被排除；"
                    f"允许子空间 {res['allowed_perturbation_dimension']} 维，"
                    f"齐次部分 {res['homogeneous_dimension']} 维，"
                    f"联合核 {res['kernel_dimension']} 维。")
            if ok:
                text += " 可延续方向：" + "、".join(ok[:6]) + "。"
            if bad:
                text += " 被排除方向：" + "、".join(bad[:6]) + \
                        ("…" if len(bad) > 6 else "") + "。"
            claims.append({"text": text, "status": "classification",
                           "evidence": [entry.step],
                           "scope": f"{res['q1_generators']} 维 Q1 基、"
                                    f"{len(verdict)} 维扰动基、{ptext}、一阶"})

        elif tool == "numeric_check":
            drift = res.get("rel_drift")
            if drift is None:
                continue
            carter = res.get("simulator_carter_rel_drift")
            text = (f"数值交叉检验（{res['background']}，{ptext}，场景 {res['scene']}）："
                    f"沿模拟器积分出的 {res['samples']} 个采样点，用符号层的同一个 "
                    f"K^{{μν}} 计算 {res['label']}，相对漂移 {drift:.2e}")
            if carter is not None:
                text += f"（模拟器自身 Carter 常数漂移 {carter:.2e} 作对照）"
            text += "。"
            if drift < 1e-6:
                text += " 数值上守恒，与符号恒等式一致。"
                status = "verified"
            else:
                text += " 数值上不守恒。"
                status = "nonexistence_in_span"
            claims.append({"text": text, "status": status,
                           "evidence": [entry.step],
                           "scope": f"{res['samples']} 个采样点、{res['scene']}、"
                                    f"{ptext}；数值结果不构成存在性证明"})

    summary = (f"共 {len(ledger.entries)} 次工具调用，"
               f"累计 {ledger.total_seconds:.1f}s。")
    if claims:
        summary += f" 得到 {len(claims)} 条有证书的结论。"
    else:
        summary += " 尚未得到可下结论的结果。"
    return {"summary": summary, "claims": claims}


def pick_background(question: str) -> str:
    """从问题里猜出该用哪个背景（规则模式用）。"""
    q = (question or "").lower()
    if "kerr" in q or "克尔" in q:
        return "kerr"
    if "schwarz" in q or "史瓦西" in q:
        return "schwarzschild"
    if "minkowski" in q or "平直" in q:
        return "minkowski"
    return "conformal2d"


def suggest(ledger, budget_seconds=900, question="") -> list:
    """无 LLM 时的规则式下一步建议。"""
    if not ledger.entries:
        return [{"action": "list_backgrounds", "args": {},
                 "why": "先看清有哪些背景度规"}]
    last = ledger.entries[-1]
    diag = last.diagnosis or {}
    tool = last.tool
    kind = diag.get("kind")
    spent = sum(e.seconds for e in ledger.entries)

    if tool == "finish":
        return []
    if spent > budget_seconds:
        return [{"action": "finish", "args": {},
                 "why": f"已用 {spent:.0f}s，超出预算"}]

    if kind == "artifact":
        return [{"action": last.tool, "args": last.args,
                 "why": "网格不足，加密后重算"}]

    if tool == "list_backgrounds":
        return [{"action": "inspect_background",
                 "args": {"background": pick_background(question)},
                 "why": "先看问题涉及的那个背景"}]

    if tool == "inspect_background":
        bg = last.args["background"]
        if bg == "conformal2d":
            return [{"action": "killing_search",
                     "args": {"background": bg, "degree": 2,
                              "parameters": {"epsilon": "0"}},
                     "why": "先确认 epsilon=0 时能搜到 H 与 Q0"}]
        return [{"action": "killing_search",
                 "args": {"background": bg, "degree": 2}, "why": "先做一次基准搜索"}]

    if tool == "killing_search":
        bg = last.args["background"]
        if bg == "conformal2d":
            eps = (last.args.get("parameters") or {}).get("epsilon")
            if eps in (None, "0"):
                return [{"action": "scan_parameter",
                         "args": {"background": bg, "parameter": "epsilon",
                                  "values": ["0", "1", "-1/4"], "degree": 2},
                         "why": "看 epsilon 变化时解空间维数怎么变"}]
            return [{"action": "finish", "args": {}, "why": "二维部分已可下结论"}]
        if kind == "existence" and (last.result or {}).get("gap", 0) > 0:
            return [{"action": "verify_quantity",
                     "args": {"background": bg,
                              "Q": last.result["solutions"][0]["Q"]},
                     "why": "独立回代第一个新解"}]
        perturbation_words = ("扰动", "延续", "一阶", "perturb", "continue")
        if any(word in question for word in perturbation_words):
            return [{"action": "first_order_classification",
                     "args": {"background": bg, "q0": "Q_C",
                              "family": "eq54+components"},
                     "why": "问题涉及扰动延续，进入一阶分类"}]
        return [{"action": "finish", "args": {}, "why": "当前背景下没有额外解"}]

    if tool == "scan_parameter":
        return [{"action": "finish", "args": {},
                 "why": "参数扫描已完成，可以下结论"}]

    if tool == "first_order_classification":
        return [{"action": "finish", "args": {}, "why": "已得到分类结论"}]

    return [{"action": "finish", "args": {}, "why": "没有更多规则可用"}]
