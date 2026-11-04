"""扰动族模板：把"要检验哪些度量扰动"写成可复用的规格。

两类模板
--------
* eq54_family —— Vigeland-Yunes-Stein 式 (54) 的**单函数切片**（Kerr）。
  注意式 (54) 本身不含 f：它用四个任意函数 (deltaV, deltaq, deltalambda,
  deltagamma) 给出最一般的稳态轴对称扰动。这里把它们收成一个径向剖面 f，
  即 deltaV=-B*f/S^2, deltaq=-a*w*B*f/S^2, deltalambda=deltagamma=f/J，
  等价于把径向函数作形变 F=r^2+eps*f(r) 后取一阶项。
  收成一个函数后，各度规分量之间自动出现**配合比例**：
      h_tphi = -a*w*h_tt,  h_uu = (D/w)*h_rr,  h_ru = 0。
* component_family —— 单分量扰动：只改一个度规分量，取纯径向剖面。
  它至少破坏一条配合比例，用来检验"配合关系被改变后会怎样"。
"""

from __future__ import annotations

import sympy as sp

from engine import clean

t, r, u, phi = sp.symbols("t r u phi", real=True)
M, a = sp.symbols("M a", real=True)
W = 1 - u ** 2
S = r ** 2 + a ** 2 * u ** 2
DELTA = r ** 2 - 2 * M * r + a ** 2
B = S - 2 * M * r
J = (r - M) ** 2 - (M ** 2 - a ** 2) * u ** 2


def eq54_h(values, f):
    """式 (54) 的单函数切片：四个输入由径向剖面 f 决定，已变换到 u = cos(theta)。"""
    m, aa = values[M], values[a]
    dv = -B * f / S ** 2
    dq = -aa * W * B * f / S ** 2
    dl = f / J
    dg = f / J
    h = sp.zeros(4)
    h[0, 0] = -dv
    h[0, 3] = h[3, 0] = dq
    h[1, 1] = dl * u ** 2 + dg * (r - m) ** 2 * W / DELTA
    h[1, 2] = h[2, 1] = -(r - m) * u * (dg - dl)
    h[2, 2] = (dl * W * (r - m) ** 2 + dg * u ** 2 * DELTA) / W
    h[3, 3] = W / B * (4 * aa * m * r * dq
                       - ((r ** 2 + aa ** 2) ** 2 - aa ** 2 * W * DELTA) * dv)
    # B, S, DELTA, J, W 是模块级符号表达式，必须把 M, a 换成数值，
    # 否则 H1 会带着符号参数进入后续线性代数（又慢又可能出错）。
    h = h.subs({M: m, a: aa})
    return h.applyfunc(clean)


def parse_profiles(specs, symbol=r, local=None):
    """把剖面字符串解析成表达式。

    local 必须带上背景的全部坐标符号（例如 2 维模型的 x, y），
    否则 "x^2*y^2" 里的 y 会变成一个新的、不等于坐标 y 的符号，
    后续 Poly 会把它当系数，导致域转换失败。
    """
    env = {"r": symbol}
    if local:
        env.update(local)
    out = []
    for spec in specs:
        if isinstance(spec, sp.Expr):
            out.append((sp.sstr(spec), spec))
        else:
            expr = sp.sympify(str(spec).replace("^", "**"), locals=env)
            out.append((str(spec), expr))
    return out


def eq54_family(values, profiles=("1", "r^-1", "r^-2", "r^-3"), local=None):
    """A 类：落在式 (54) 单函数切片内的扰动方向（自动满足分量配合比例）。"""
    out = []
    for label, expr in parse_profiles(profiles, r, local):
        out.append((f"eq54[f={label}]", eq54_h(values, expr)))
    return out


def component_family(entries=("00", "11", "22", "33", "03", "12"),
                     profiles=("r^-2", "r^-3"), dim=4, symbol=r, local=None):
    """B 类：单分量扰动，破坏配合关系。

    entries 是度规分量编码 "ij"（i<=j）。dim 决定矩阵阶数，
    symbol 是剖面所在的径向坐标（4 维是 r，2 维是 x）。
    """
    names = {"00": "h_tt", "11": "h_rr", "22": "h_uu", "33": "h_phiphi",
             "03": "h_tphi", "12": "h_ru", "01": "h_xy"}
    if dim == 2:
        names = {"00": "h_xx", "11": "h_yy", "01": "h_xy"}
    out = []
    for code in entries:
        i, j = int(code[0]), int(code[1])
        if max(i, j) >= dim:
            raise ValueError(f"分量 {code} 超出 {dim} 维；可用 "
                             f"{[c for c in names if max(int(c[0]), int(c[1])) < dim]}")
        for label, expr in parse_profiles(profiles, symbol, local):
            h = sp.zeros(dim)
            h[i, j] = expr
            h[j, i] = expr
            out.append((f"{names.get(code, f'h_{i}{j}')}={label}", h.applyfunc(clean)))
    return out


def pairing_report(values):
    """式 (54) 的四条配合关系（供报告引用）。"""
    return [
        "h_tphi = -a*w*h_tt",
        "h_uu = (D/w)*h_rr",
        "h_ru = 0",
        "h_phiphi = (w*f/S^2)*[(r^2+a^2)^2 - a^2*w*D - 4*a^2*M*r*w]",
    ]
