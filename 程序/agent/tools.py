"""工具层：Agent 能调用的全部动作（JSON schema + 派发）。

设计原则
--------
* 每个工具都是纯函数式的确定性计算，不依赖 LLM。
* 返回值不是"成功/失败"，而是**诊断签名**：既给数字，也给证书和适用范围。
* 工具可以脱离 Agent 单独调用（见 cli.py 的 --tool 模式）。
"""

from __future__ import annotations

import json
import traceback
from time import perf_counter

import sympy as sp

import backgrounds as bgmod
import families
import solvers
from engine import cancel_only, format_poly, parse_number

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_backgrounds",
            "description": "列出可用的背景度规（2 维模型、Minkowski、Schwarzschild、Kerr）"
                           "及其参数、已知二次不变量。第一步通常调用它。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_background",
            "description": "查看某个背景在给定参数下的度规矩阵、逆度规、Hamiltonian 和已知不变量。",
            "parameters": {
                "type": "object",
                "properties": {
                    "background": {"type": "string",
                                   "description": "背景 key，例如 conformal2d / kerr"},
                    "parameters": {"type": "object",
                                   "description": '参数覆盖，例如 {"M": "1", "a": "1/2"}'},
                },
                "required": ["background"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "killing_search",
            "description": "在指定的有限候选空间内精确搜索二次守恒量（Killing 张量）。"
                           "返回秩、零空间维数、已知不变量维数、gap，以及每个解的显式表达式和回代验证结果。"
                           "gap = nullity - known_dim > 0 才可能有新东西。",
            "parameters": {
                "type": "object",
                "properties": {
                    "background": {"type": "string"},
                    "parameters": {"type": "object"},
                    "degree": {"type": "integer",
                               "description": "系数多项式的最高次数（默认 2）"},
                    "momentum_degree": {"type": "integer",
                                        "description": "动量次数，二次守恒量填 2（默认）"},
                },
                "required": ["background"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_quantity",
            "description": "精确验证一个给定的 Q 是否与 H 泊松对易。返回 poisson_zero 与首个非零系数。",
            "parameters": {
                "type": "object",
                "properties": {
                    "background": {"type": "string"},
                    "parameters": {"type": "object"},
                    "Q": {"type": "string",
                          "description": "Q 的表达式，用坐标和动量符号书写，例如 "
                                         "\"(2*y^2*p_x^2-(1+x^2)*p_y^2)/(1+x^2+2*y^2)\""},
                },
                "required": ["background", "Q"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scan_parameter",
            "description": "扫描一个参数，给出每个取值下的 rank / nullity / gap，并标出 nullity 跳变处（分岔点）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "background": {"type": "string"},
                    "parameter": {"type": "string", "description": "参数名，如 epsilon / a"},
                    "values": {"type": "array", "items": {"type": "string"},
                               "description": '取值列表，如 ["0","1","-1/4"]'},
                    "base_parameters": {"type": "object"},
                    "degree": {"type": "integer"},
                },
                "required": ["background", "parameter", "values"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "first_order_classification",
            "description": "解一阶延续方程 {Q1,H0} + {Q0,H1} = 0，并分类：哪些度量扰动方向允许延续、"
                           "哪些在声明的有限基内被排除。会返回约束矩阵、逐方向判定和网格自证历史。",
            "parameters": {
                "type": "object",
                "properties": {
                    "background": {"type": "string"},
                    "parameters": {"type": "object"},
                    "q0": {"type": "string",
                           "description": "背景已知不变量的标签，如 Q_C / L^2 / Q0"},
                    "family": {"type": "string",
                               "enum": ["eq54", "components", "eq54+components"],
                               "description": "eq54 = 保持式(54)配合关系；components = 单分量（破坏配合关系）"},
                    "profiles": {"type": "array", "items": {"type": "string"},
                                 "description": '径向剖面，如 ["1","r^-1","r^-2","r^-3"]'},
                    "components": {"type": "array", "items": {"type": "string"},
                                   "description": '要扰动的度规分量编码，如 ["00","11","22","33","03","12"]'},
                    "q1_exponents": {"type": "object",
                                     "description": 'Q1 的坐标幂次，如 {"r":[-3,-2,-1,0,1,2],"u":[0,2]}'},
                    "augment": {"type": "array", "items": {"type": "string"},
                                "description": "增广权重，如 [\"a^2*u^2\"]，保证已知正例落在候选空间内"},
                    "grid": {"type": "array", "items": {"type": "integer"},
                             "description": "初始网格 [nr, nu]，默认 [4,4]"},
                },
                "required": ["background", "q0", "family"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "numeric_check",
            "description": "把符号层的候选 Q 交给黑洞模拟器做数值交叉检验：积分一条测地线，"
                           "用**同一个 K^{μν}** 沿轨道计算 Q，返回相对漂移。"
                           "漂移接近机器精度说明数值上守恒；显著增长说明候选有误或积分精度不足。"
                           "这是与符号恒等式独立的第二条证据。",
            "parameters": {
                "type": "object",
                "properties": {
                    "background": {"type": "string",
                                   "enum": ["kerr", "schwarzschild"]},
                    "parameters": {"type": "object"},
                    "label": {"type": "string",
                              "description": "要检验的不变量标签，如 Q_C / H / L^2"},
                    "preset": {"type": "string",
                               "description": "模拟器预设（可选，优先于自定义场景）"},
                    "r": {"type": "number", "description": "初始 BL 半径（默认 9 或 10）"},
                    "theta_deg": {"type": "number", "description": "初始极角（默认 62）"},
                    "duration": {"type": "number", "description": "仿射参数时长（默认 600）"},
                },
                "required": ["background", "label"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "结束研究并提交结论。每条结论必须附上支撑它的工具调用序号（evidence）。"
                           "策略层会检查证据是否覆盖结论。",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "整体结论，中文"},
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "status": {"type": "string",
                                           "enum": ["verified", "existence_in_span",
                                                    "nonexistence_in_span",
                                                    "classification", "inconclusive"]},
                                "evidence": {"type": "array", "items": {"type": "integer"},
                                             "description": "支撑该结论的工具调用序号"},
                                "scope": {"type": "string",
                                          "description": "适用范围，如 'M=1,a=1/2；Q1 基 138 维'"},
                            },
                            "required": ["text", "status", "evidence"],
                        },
                    },
                },
                "required": ["summary", "claims"],
            },
        },
    },
]


# ------------------------------------------------------------------ 解析 ----
def _params(bg, raw):
    """把 JSON 里的字符串参数转成 sympy 值，并与背景默认值合并。

    返回的一定是**完整**参数字典：后续 families/solvers 会直接按键取值，
    只给覆盖项会漏掉默认值（例如 eq54 族需要 M 与 a）。
    """
    out = dict(bg.params)
    for key, value in (raw or {}).items():
        sym = next((s for s in bg.params if sp.sstr(s) == key), None)
        if sym is None:
            raise KeyError(f"{bg.key} 没有参数 {key!r}；"
                           f"可选 {[sp.sstr(s) for s in bg.params]}")
        out[sym] = parse_number(value)
    return out


def _axes(bg, spec):
    if not spec:
        return None
    return tuple(sp.Symbol(name, real=True) for name in spec)


# ------------------------------------------------------------------ 派发 ----
def dispatch(name, args, ctx):
    start = perf_counter()
    try:
        result = _DISPATCH[name](args, ctx)
    except solvers.BudgetExceeded as exc:
        result = {"error": "budget_exceeded", "message": str(exc),
                  "hint": "降低 degree 或缩小 Q1 幂次范围"}
    except Exception as exc:                              # noqa: BLE001
        result = {"error": type(exc).__name__, "message": str(exc),
                  "traceback": traceback.format_exc(limit=3)}
    result = _jsonable(result)
    result.setdefault("_tool", name)
    result["_seconds"] = round(perf_counter() - start, 2)
    return result


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (bool, int, float)) or obj is None:
        return obj
    if isinstance(obj, str):
        return obj
    try:
        return sp.sstr(obj)
    except Exception:                                     # noqa: BLE001
        return str(obj)


# ------------------------------------------------------------------ 工具 ----
def _list_backgrounds(args, ctx):
    out = []
    for key, bg in bgmod.BACKGROUNDS.items():
        p = dict(bg.params)
        out.append({
            "key": key,
            "name": bg.name,
            "dim": bg.dim,
            "coords": [sp.sstr(c) for c in bg.coords],
            "momenta": [sp.sstr(q) for q in bg.momenta],
            "parameters": {sp.sstr(k): sp.sstr(v) for k, v in p.items()},
            "ansatz_axes": [sp.sstr(c) for c in bg.ansatz_axes],
            "known_invariants": [lab for lab, _, _ in bg.known(p)],
            "notes": bg.notes,
        })
    return {"backgrounds": out,
            "hint": "先用 inspect_background 看清度规与不变量，再用 killing_search / "
                    "first_order_classification 做具体实验"}


def _inspect_background(args, ctx):
    bg = bgmod.get(args["background"])
    values = _params(bg, args.get("parameters"))
    p = {**bg.params, **values}
    gi = bg.g_inv(values)
    H = bg.H(values)
    return {
        "background": bg.key,
        "name": bg.name,
        "parameters": {sp.sstr(k): sp.sstr(v) for k, v in p.items()},
        "coords": [sp.sstr(c) for c in bg.coords],
        "momenta": [sp.sstr(q) for q in bg.momenta],
        "metric": [[sp.sstr(bg.metric(p)[i, j]) for j in range(bg.dim)]
                   for i in range(bg.dim)],
        "inverse_metric": [[sp.sstr(gi[i, j]) for j in range(bg.dim)]
                           for i in range(bg.dim)],
        "H": format_poly(H, bg.coords, bg.momenta, 8),
        "known_invariants": [
            {"label": lab, "note": note,
             "Q": format_poly(poly, bg.coords, bg.momenta, 6)}
            for lab, poly, note in bg.known(values)],
        "ansatz_denominator": sp.sstr(cancel_only(bg.ansatz_denominator(p))),
        "domain_note": bg.notes,
    }


def _killing_search(args, ctx):
    bg = bgmod.get(args["background"])
    values = _params(bg, args.get("parameters"))
    out, _sols, _gens, _labels = solvers.killing_search(
        bg, values,
        degree=int(args.get("degree", 2)),
        max_generators=int(args.get("max_generators", 260)))
    return out


def _verify_quantity(args, ctx):
    bg = bgmod.get(args["background"])
    values = _params(bg, args.get("parameters"))
    return solvers.verify_quantity(bg, values, args["Q"])


def _scan_parameter(args, ctx):
    bg = bgmod.get(args["background"])
    values = _params(bg, args.get("base_parameters"))
    return solvers.scan_parameter(
        bg, args["parameter"], args["values"], values,
        degree=int(args.get("degree", 2)))


def _first_order_classification(args, ctx):
    bg = bgmod.get(args["background"])
    values = _params(bg, args.get("parameters"))
    kind = args.get("family", "components")
    profiles = args.get("profiles")
    local = {sp.sstr(s): s for s in bg.coords + bg.momenta}
    if kind in ("eq54", "eq54+components") and bg.key != "kerr":
        raise ValueError(
            "eq54 族只对 Kerr 有定义（它依赖 M 与 a）。"
            f"{bg.key} 请用 family=\"components\"。")
    specs = []
    if kind in ("eq54", "eq54+components"):
        specs += families.eq54_family(values, profiles or ("1", "r^-1", "r^-2", "r^-3"),
                                      local=local)
    if kind in ("components", "eq54+components"):
        default_components = (("00", "11", "01") if bg.dim == 2
                              else ("00", "11", "22", "33", "03", "12"))
        specs += families.component_family(
            tuple(args.get("components") or default_components),
            profiles or ("r^-2", "r^-3"),
            dim=bg.dim, symbol=bg.ansatz_axes[0], local=local)
    exponents = args.get("q1_exponents")
    if exponents is None:
        if bg.dim == 4:
            exponents = {"r": [-3, -2, -1, 0, 1, 2], "u": [0, 2]}
        else:
            exponents = {"x": [0, 1, 2], "y": [0, 1, 2]}
    grid = args.get("grid") or [4, 4]
    result = solvers.first_order_classification(
        bg, values, args["q0"], specs,
        q1_degree=int(args.get("q1_degree", 2)),
        q1_exponents=exponents,
        augment_weights=tuple(args.get("augment") or
                              (("a^2*u^2",) if bg.key == "kerr"
                               and kind != "components" else ())),
        nr=int(grid[0]), nu=int(grid[1]))
    result["pairing_relations"] = (families.pairing_report(values)
                                   if bg.key == "kerr" else [])
    result["h_spec_summary"] = [label for label, _ in specs]
    return result


def _numeric_check(args, ctx):
    import simulator as sim

    if not sim.available():
        raise RuntimeError(
            f"找不到模拟器命令行：{sim.CLI}；"
            "请先构建 Black Hole Simulator，或用 BH_SIM_ROOT 指定安装目录")
    bg = bgmod.get(args["background"])
    if bg.dim != 4 or bg.key not in ("kerr", "schwarzschild"):
        raise ValueError("数值检验只支持 4 维 Kerr / Schwarzschild 背景")
    values = _params(bg, args.get("parameters"))
    label = args.get("label", "Q_C")
    known = {lab: poly for lab, poly, _ in bg.known(values)}
    if label not in known:
        raise KeyError(f"{bg.key} 没有不变量 {label!r}；可选 {sorted(known)}")

    preset = args.get("preset")
    if preset:
        data = sim.trace(preset=preset)
        scene_desc = f"preset:{preset}"
    else:
        M = float(values.get(bgmod.M, 1))
        a = float(values.get(bgmod.a, 0))
        r0 = float(args.get("r", 9.0 if bg.key == "kerr" else 10.0))
        scene_obj = sim.scene(M=M, a=a, r=r0,
                              theta_deg=float(args.get("theta_deg", 62.0)),
                              duration=float(args.get("duration", 600.0)))
        data = sim.trace(scene_obj=scene_obj)
        scene_desc = f"M={M:g}, a={a:g}, r0={r0:g}"

    body = (data.get("bodies") or [{}])[0]
    samples = body.get("samples") or []
    if len(samples) < 3:
        raise RuntimeError(f"轨迹采样不足（{len(samples)}），无法做数值检验")

    M = float(data.get("M", 1.0))
    a = float(data.get("a", 0.0))
    mass = float(body.get("mass", 1.0))
    res = sim.evaluate_quantity(known[label], bg, values, samples, label,
                                M=M, a=a, mass=mass)
    if not res.get("ok"):
        raise RuntimeError(res.get("reason", "数值评估失败"))

    carter = sim.carter_from_samples(samples)
    ham = sim.hamiltonian_check(samples)
    step = max(1, len(res["lam"]) // 200)
    return {
        "background": bg.key,
        "parameters": {sp.sstr(k): sp.sstr(v) for k, v in values.items()},
        "label": label,
        "scene": scene_desc,
        "samples": len(samples),
        "stop": body.get("stop"),
        "rel_drift": res["rel_drift"],
        "max_abs_drift": res["max_abs_drift"],
        "v0": res["v0"],
        "v_last": res["v_last"],
        "simulator_carter_rel_drift": carter.get("rel_drift"),
        "hamiltonian_spread": ham.get("spread"),
        "series": {"lam": res["lam"][::step], "drift": res["drift"][::step]},
        "certificate": {
            "type": "numerical_cross_check",
            "method": "模拟器积分测地线 -> 同一 K^{μν} 沿轨道求 Q",
            "rel_drift": res["rel_drift"],
            "note": "数值上近似不变不能单独证明存在 Killing 张量；"
                    "它只用来与符号恒等式互证、并定位候选量的错误。",
        },
    }


def _finish(args, ctx):
    return {"accepted": True,
            "summary": args.get("summary", ""),
            "claims": args.get("claims", [])}


_DISPATCH = {
    "list_backgrounds": _list_backgrounds,
    "inspect_background": _inspect_background,
    "killing_search": _killing_search,
    "verify_quantity": _verify_quantity,
    "scan_parameter": _scan_parameter,
    "first_order_classification": _first_order_classification,
    "numeric_check": _numeric_check,
    "finish": _finish,
}


def schema_for(name):
    for item in TOOL_SCHEMAS:
        if item["function"]["name"] == name:
            return item
    raise KeyError(name)


def tool_names():
    return [item["function"]["name"] for item in TOOL_SCHEMAS]


# ------------------------------------------------------------ 前端精简 ----
def brief(result):
    """给界面的精简结果：只留能渲染的关键字段。"""
    tool = result.get("_tool")
    if result.get("error"):
        return {"error": result.get("error"), "message": result.get("message")}

    if tool == "killing_search":
        return {
            "background": result.get("background"),
            "parameters": result.get("parameters"),
            "unknowns": result.get("unknowns"),
            "equations": result.get("equations"),
            "rank": result.get("rank"),
            "nullity": result.get("nullity"),
            "known_dim": result.get("known_dim"),
            "gap": result.get("gap"),
            "solutions": [{"Q": s.get("Q"), "verified": s.get("verified")}
                          for s in (result.get("solutions") or [])[:6]],
            "certificate": result.get("certificate"),
        }
    if tool == "verify_quantity":
        return {k: result.get(k) for k in
                ("background", "parameters", "poisson_zero", "first_nonzero",
                 "Q_parsed_terms", "certificate")}
    if tool == "scan_parameter":
        return {k: result.get(k) for k in
                ("background", "parameter", "table", "transition_indices")}
    if tool == "first_order_classification":
        out = {k: result.get(k) for k in
               ("background", "parameters", "q0", "h_basis", "q1_generators",
                "kernel_dimension", "homogeneous_dimension",
                "allowed_perturbation_dimension", "verdict",
                "allowed_directions", "grid_history", "certificate")}
        out["constraint_rows"] = len(result.get("constraints") or [])
        return out
    if tool == "inspect_background":
        return {k: result.get(k) for k in
                ("background", "name", "parameters", "H", "known_invariants",
                 "metric", "inverse_metric")}
    if tool == "numeric_check":
        return {k: result.get(k) for k in
                ("background", "parameters", "label", "scene", "samples",
                 "stop", "rel_drift", "max_abs_drift", "v0", "v_last",
                 "simulator_carter_rel_drift", "hamiltonian_spread",
                 "certificate")}
    if tool == "list_backgrounds":
        return {"backgrounds": result.get("backgrounds")}
    return {k: v for k, v in result.items() if not k.startswith("_")}
