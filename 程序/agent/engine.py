"""稀疏动量多项式引擎。

相空间 (x^i, p_i)。二次守恒量 Q = K^{ij}(x) p_i p_j 用一个 dict 表示：
    { (n_0, ..., n_{d-1}) : coefficient(x) }
key 是动量指数向量，value 是关于坐标的 sympy 表达式。

泊松括号 {f,g} = sum_i (df/dx^i * dg/dp_i - df/dp_i * dg/dx^i)。
{f,g} 是三次齐次多项式，所以结果仍然是同一个 dict 结构。
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence

import sympy as sp


# ---------------------------------------------------------------- 化简 ----
def clean(expr):
    """精确有理化简；把未定义函数及其导数当作代数独立的 jet 处理。

    这样 f(r)、f'(r) 之类不会触发 SymPy 的函数域启发式，
    任意光滑剖面都能参与恒等式判定。
    """
    expr = sp.sympify(expr)
    if expr == 0:
        return sp.Integer(0)
    from sympy.core.function import AppliedUndef
    jets = sorted(expr.atoms(AppliedUndef, sp.Derivative), key=sp.srepr)
    if jets:
        repl = {jet: sp.Symbol(f"_jet_{i}") for i, jet in enumerate(jets)}
        back = {v: k for k, v in repl.items()}
        return sp.factor(sp.cancel(expr.xreplace(repl))).xreplace(back)
    return sp.factor(sp.cancel(expr))


def cancel_only(expr):
    """只要通分，不做因式分解。热路径用它——factor 在 Kerr 的有理函数上
    会慢一到两个数量级，而判定是否为零只需要 cancel。"""
    expr = sp.sympify(expr)
    return sp.Integer(0) if expr == 0 else sp.cancel(expr)


def parse_number(text):
    """把外部传来的参数（字符串）转成 sympy 数，小数优先转成精确有理数。

    "0.9" 若变成 Float，后面的秩与零空间就退化成浮点运算，
    所以这里优先走 Rational。
    """
    s = str(text).strip().replace("^", "**")
    try:
        return sp.Rational(s)
    except (TypeError, ValueError):
        return sp.sympify(s)


def tidy(expr):
    """合并成单个分式，不做多项式 GCD。

    实测：对 Kerr 一阶扰动的中间表达式，sp.cancel 需要 120s，
    sp.together 只要 1.2s。而 num==0 与 expr==0 等价，
    所以"是否为零"与"取分子"这两类操作都只需要 together。
    """
    expr = sp.sympify(expr)
    return sp.Integer(0) if expr == 0 else sp.together(expr)


def numerator(expr):
    return sp.fraction(tidy(expr))[0]


def is_zero_expr(expr) -> bool:
    """精确零判定。

    together 只合并分式，不保证分子在结构上就是 0：
    x*(y+1) - x*y - x 这类写法必须 expand 才收得掉。
    expand 比 cancel 便宜得多（后者要做多项式 GCD）。
    """
    expr = sp.sympify(expr)
    if expr == 0:
        return True
    num = sp.fraction(tidy(expr))[0]
    if num == 0:
        return True
    return sp.expand(num) == 0


def require_zero(expr, label=""):
    if not is_zero_expr(expr):
        raise AssertionError(f"{label}: nonzero residual = {clean(expr)}")
    return True


# ------------------------------------------------------------ 多项式运算 ----
def to_expr(poly: dict, momenta: Sequence[sp.Symbol]):
    """dict -> sympy 表达式。"""
    terms = []
    for powers, coef in sorted(poly.items()):
        if coef == 0:
            continue
        mono = sp.Integer(1)
        for p, n in zip(momenta, powers):
            if n:
                mono *= p ** n
        terms.append(coef * mono)
    return sp.Add(*terms) if terms else sp.Integer(0)


def from_expr(expr, momenta: Sequence[sp.Symbol]) -> dict:
    """sympy 表达式 -> dict。要求关于动量是多项式。"""
    expr = sp.expand(sp.sympify(expr))
    if expr == 0:
        return {}
    poly = sp.Poly(expr, *momenta)
    return {mon: coef for mon, coef in poly.terms() if coef != 0}


def poisson(f: dict, g: dict, coords: Sequence[sp.Symbol],
            momenta: Sequence[sp.Symbol], cancel: bool = True) -> dict:
    """{f,g}，逐系数计算，不调用全局 simplify。"""
    out: dict = {}
    for i in range(len(coords)):
        df = {pf: sp.diff(cf, coords[i]) for pf, cf in f.items()}
        dg = {pg: sp.diff(cg, coords[i]) for pg, cg in g.items()}
        for pf, cf in f.items():
            left = df[pf]
            for pg, cg in g.items():
                term = pg[i] * left * cg - pf[i] * cf * dg[pg]
                if term == 0:
                    continue
                key = list(map(sum, zip(pf, pg)))
                key[i] -= 1
                key = tuple(key)
                out[key] = out.get(key, 0) + term
    if cancel:
        out = {k: tidy(v) for k, v in out.items()}
    return {k: v for k, v in out.items() if v != 0}


def scale(poly: dict, scalar) -> dict:
    return {k: tidy(scalar * v) for k, v in poly.items() if v != 0}


def add(*polys: dict) -> dict:
    keys = set().union(*(p.keys() for p in polys)) if polys else set()
    out = {}
    for k in keys:
        value = tidy(sum(p.get(k, 0) for p in polys))
        if value != 0:
            out[k] = value
    return out


def is_zero(poly: dict) -> bool:
    return all(is_zero_expr(v) for v in poly.values())


def momentum_monomials(n: int, degree: int = 2):
    """全部 n 个动量、总次数为 degree 的单项式指数。"""
    return tuple(p for p in itertools.product(range(degree + 1), repeat=n)
                 if sum(p) == degree)


def coord_monomials(coords: Sequence[sp.Symbol], degree: int):
    """总次数 <= degree 的坐标单项式。"""
    n = len(coords)
    out = []
    for total in range(degree + 1):
        for powers in itertools.product(range(total + 1), repeat=n):
            if sum(powers) != total:
                continue
            mono = sp.Integer(1)
            for c, k in zip(coords, powers):
                if k:
                    mono *= c ** k
            out.append(mono)
    return out


# ------------------------------------------------------------ 方程提取 ----
def coordinate_rows(expr, coords: Sequence[sp.Symbol]):
    """把 C(x)=0 拆成关于坐标的多项式的全部单项式系数。

    先通分，取分子。分母在所研究的坐标域上不为零，
    因此 num=0 <=> expr=0；本函数返回 (monomial, coefficient) 列表。
    """
    expr = tidy(expr)
    if expr == 0:
        return []
    num, _den = sp.fraction(expr)
    if num == 0:
        return []
    if not num.free_symbols & set(coords):
        return [((0,) * len(coords), num)]
    poly = sp.Poly(num, *coords)
    return list(poly.terms())


def hamiltonian(g_inv, momenta: Sequence[sp.Symbol]) -> dict:
    """H = (1/2) g^{ij} p_i p_j。"""
    n = len(momenta)
    out = {}
    for i in range(n):
        for j in range(i, n):
            value = tidy(g_inv[i, j] if i == j else 2 * g_inv[i, j])
            if value == 0:
                continue
            powers = [0] * n
            powers[i] += 1
            powers[j] += 1
            key = tuple(powers)
            out[key] = tidy(out.get(key, 0) + value / 2)
    return {k: v for k, v in out.items() if v != 0}


def quadratic_from_matrix(a, momenta: Sequence[sp.Symbol]) -> dict:
    """对称矩阵 a -> Q = (1/2) p^T a p。"""
    return hamiltonian(a, momenta)


def format_poly(poly: dict, coords, momenta, max_terms: int = 12) -> str:
    """给 LLM 看的紧凑字符串。"""
    items = sorted(poly.items())
    shown = items[:max_terms]
    parts = []
    for powers, coef in shown:
        mono = sp.Integer(1)
        for p, n in zip(momenta, powers):
            if n:
                mono *= p ** n
        parts.append(f"({sp.sstr(coef)})*{sp.sstr(mono)}")
    text = " + ".join(parts)
    if len(items) > max_terms:
        text += f" + ... ({len(items) - max_terms} more terms)"
    return text or "0"
