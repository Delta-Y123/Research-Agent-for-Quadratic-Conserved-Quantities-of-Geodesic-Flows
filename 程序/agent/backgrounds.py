"""背景度规注册表。

每个背景给出：坐标、动量、度规矩阵、逆度规、已知二次不变量、候选空间分母。

约定
----
* 号差 (-,+,+,+)，G=c=1。
* 4 维背景坐标顺序固定为 (t, r, u, phi)，其中 u = cos(theta)，
  因此 p_theta = -sin(theta) * p_u，d theta = -du / sin(theta)。
* 2 维模型坐标 (x, y)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import sympy as sp

from engine import add, cancel_only, clean, hamiltonian, poisson, scale

# ------------------------------------------------------------------ 符号 ----
t, r, u, phi = sp.symbols("t r u phi", real=True)
x, y = sp.symbols("x y", real=True)
pt, pr, pu, pp = sp.symbols("p_t p_r p_u p_phi", real=True)
px, py = sp.symbols("p_x p_y", real=True)
M, a, eps = sp.symbols("M a epsilon", real=True)
X2 = sp.Symbol("x", real=True)
Y2 = sp.Symbol("y", real=True)

W = 1 - u ** 2
S = r ** 2 + a ** 2 * u ** 2
DELTA = r ** 2 - 2 * M * r + a ** 2
A_KERR = r ** 2 + a ** 2
B_KERR = S - 2 * M * r


@dataclass
class Background:
    key: str
    name: str
    coords: tuple
    momenta: tuple
    params: dict                      # symbol -> default value
    metric: Callable                  # params dict -> Matrix
    ansatz_denominator: Callable      # params dict -> expression
    invariants: Callable              # params dict -> list[(label, dict, note)]
    ansatz_axes: tuple = ()           # 系数函数允许依赖的坐标
    notes: str = ""
    dim: int = 4

    def g(self, values: dict):
        p = {**self.params, **values}
        return self.metric(p).applyfunc(cancel_only)

    def g_inv(self, values: dict):
        p = {**self.params, **values}
        return inverse(self.metric(p), self.dim).applyfunc(cancel_only)

    def H(self, values: dict) -> dict:
        return hamiltonian(self.g_inv(values), self.momenta)

    def known(self, values: dict):
        p = {**self.params, **values}
        return self.invariants(p)


def inverse(g, dim: int):
    """(t, phi) 二乘二块 + (r, u) 对角的逆；其他情形退回通用求逆。"""
    if dim == 2:
        return sp.diag(*[cancel_only(1 / g[i, i]) for i in range(2)])
    det = cancel_only(g[0, 0] * g[3, 3] - g[0, 3] ** 2)
    gi = sp.zeros(4)
    gi[0, 0], gi[3, 3] = cancel_only(g[3, 3] / det), cancel_only(g[0, 0] / det)
    gi[0, 3] = gi[3, 0] = cancel_only(-g[0, 3] / det)
    gi[1, 1], gi[2, 2] = cancel_only(1 / g[1, 1]), cancel_only(1 / g[2, 2])
    return gi


# --------------------------------------------------------------- 2 维模型 ----
def _conformal2d_metric(p):
    lam = 1 + x ** 2 + 2 * y ** 2 + p[eps] * x ** 2 * y ** 2
    return sp.diag(lam, lam)


def _conformal2d_invariants(p):
    lam = 1 + x ** 2 + 2 * y ** 2 + p[eps] * x ** 2 * y ** 2
    g_inv = inverse(_conformal2d_metric(p), 2)
    H = hamiltonian(g_inv, (px, py))
    out = [("H", H, "测地流 Hamiltonian（可约）")]
    if sp.simplify(p[eps]) == 0:
        Q0 = {(2, 0): clean(2 * y ** 2 / lam),
              (0, 2): clean(-(1 + x ** 2) / lam)}
        out.append(("Q0", Q0, "Liouville 型第二不变量（仅 epsilon=0 时守恒）"))
    return out


# ----------------------------------------------------------------- 平直 ----
def _minkowski_metric(p):
    return sp.diag(-1, 1, 1, 1)


def _minkowski_invariants(p):
    g_inv = sp.diag(-1, 1, 1, 1)
    H = hamiltonian(g_inv, (pt, pr, pu, pp))
    out = [("H", H, "平直 Hamiltonian")]
    out.append(("p_t^2", {(2, 0, 0, 0): sp.Integer(1)}, "平移不变 -> 能量平方"))
    out.append(("p_r^2+p_u^2+p_phi^2",
                {(0, 2, 0, 0): sp.Integer(1), (0, 0, 2, 0): sp.Integer(1),
                 (0, 0, 0, 2): sp.Integer(1)}, "旋转不变 -> 动量平方"))
    return out


# ------------------------------------------------------------ Schwarzschild ----
def _schwarzschild_metric(p):
    f = 1 - 2 * p[M] / r
    g = sp.zeros(4)
    g[0, 0] = -f
    g[1, 1] = clean(1 / f)
    g[2, 2] = clean(r ** 2 / W)
    g[3, 3] = r ** 2 * W
    return g


def _schwarzschild_invariants(p):
    g_inv = inverse(_schwarzschild_metric(p), 4)
    H = hamiltonian(g_inv, (pt, pr, pu, pp))
    L2 = {(0, 0, 2, 0): W, (0, 0, 0, 2): clean(1 / W)}
    return [("H", H, "Schwarzschild Hamiltonian"),
            ("L^2", L2, "总角动量平方 = p_theta^2 + p_phi^2/sin^2(theta)"),
            ("p_t^2", {(2, 0, 0, 0): sp.Integer(1)}, "能量平方（可约）"),
            ("p_phi^2", {(0, 0, 0, 2): sp.Integer(1)}, "L_z^2（可约）")]


# -------------------------------------------------------------------- Kerr ----
def _kerr_metric(p):
    m, aa = p[M], p[a]
    w = W
    s = r ** 2 + aa ** 2 * u ** 2
    d = r ** 2 - 2 * m * r + aa ** 2
    af = r ** 2 + aa ** 2
    g = sp.zeros(4)
    g[0, 0] = clean(-(d - aa ** 2 * w) / s)
    g[0, 3] = g[3, 0] = clean(-aa * w * (af - d) / s)
    g[3, 3] = clean(w * (af ** 2 - aa ** 2 * d * w) / s)
    g[1, 1] = clean(s / d)
    g[2, 2] = clean(s / w)
    return g


def _kerr_invariants(p):
    m, aa = p[M], p[a]
    g_inv = inverse(_kerr_metric(p), 4)
    H = hamiltonian(g_inv, (pt, pr, pu, pp))
    QC = add({(0, 0, 2, 0): W,
              (2, 0, 0, 0): clean(aa ** 2 * W),
              (1, 0, 0, 1): clean(2 * aa),
              (0, 0, 0, 2): clean(1 / W)},
             scale(H, clean(-2 * aa ** 2 * u ** 2)))
    return [("H", H, "Kerr Hamiltonian"),
            ("Q_C", QC, "Carter 常数（真实守恒；等价于 Carter Q + (L_z-aE)^2）"),
            ("p_t^2", {(2, 0, 0, 0): sp.Integer(1)}, "能量平方（可约）"),
            ("p_t*p_phi", {(1, 0, 0, 1): sp.Integer(1)}, "可约"),
            ("p_phi^2", {(0, 0, 0, 2): sp.Integer(1)}, "L_z^2（可约）")]


# ------------------------------------------------------------------ 注册 ----
BACKGROUNDS: dict[str, Background] = {
    "conformal2d": Background(
        key="conformal2d",
        name="二维共形平坦模型 g = lambda_eps (dx^2+dy^2)",
        coords=(x, y), momenta=(px, py), dim=2,
        params={eps: sp.Integer(0)},
        metric=_conformal2d_metric,
        ansatz_denominator=lambda p: (1 + x ** 2 + 2 * y ** 2
                                      + p[eps] * x ** 2 * y ** 2),
        invariants=_conformal2d_invariants,
        ansatz_axes=(x, y),
        notes="lambda_eps = 1 + x^2 + 2 y^2 + epsilon x^2 y^2；"
              "epsilon=0 时可分离，有第二不变量 Q0。"),
    "minkowski": Background(
        key="minkowski",
        name="四维平直时空 (t,x,y,z)",
        coords=(t, r, u, phi), momenta=(pt, pr, pu, pp),
        params={M: sp.Integer(0), a: sp.Integer(0)},
        metric=_minkowski_metric,
        ansatz_denominator=lambda p: sp.Integer(1),
        invariants=_minkowski_invariants,
        ansatz_axes=(t, r, u, phi),
        notes="坐标名沿用 (t,r,u,phi) 但度规为 diag(-1,1,1,1)。"),
    "schwarzschild": Background(
        key="schwarzschild",
        name="Schwarzschild 时空（u = cos theta）",
        coords=(t, r, u, phi), momenta=(pt, pr, pu, pp),
        params={M: sp.Integer(1), a: sp.Integer(0)},
        metric=_schwarzschild_metric,
        ansatz_denominator=lambda p: sp.Integer(1),
        invariants=_schwarzschild_invariants,
        ansatz_axes=(r, u),
        notes="f = 1 - 2M/r；工作域 r > 2M，-1 < u < 1。"),
    "kerr": Background(
        key="kerr",
        name="Kerr 时空（Boyer-Lindquist，u = cos theta）",
        coords=(t, r, u, phi), momenta=(pt, pr, pu, pp),
        params={M: sp.Integer(1), a: sp.Rational(1, 2)},
        metric=_kerr_metric,
        ansatz_denominator=lambda p: sp.Integer(1),
        invariants=_kerr_invariants,
        ansatz_axes=(r, u),
        notes="S = r^2+a^2u^2, D = r^2-2Mr+a^2, w = 1-u^2；"
              "工作域 r > r_+ = M + sqrt(M^2-a^2)，-1 < u < 1。"),
}


def get(key: str) -> Background:
    if key not in BACKGROUNDS:
        raise KeyError(f"unknown background {key!r}; "
                       f"choose from {sorted(BACKGROUNDS)}")
    return BACKGROUNDS[key]


def describe(key: str, values: dict | None = None) -> dict:
    bg = get(key)
    p = {**bg.params, **(values or {})}
    return {
        "key": bg.key,
        "name": bg.name,
        "dim": bg.dim,
        "coords": [sp.sstr(c) for c in bg.coords],
        "momenta": [sp.sstr(q) for q in bg.momenta],
        "params": {sp.sstr(k): sp.sstr(v) for k, v in p.items()},
        "metric": [[sp.sstr(clean(bg.metric(p)[i, j])) for j in range(bg.dim)]
                   for i in range(bg.dim)],
        "ansatz_denominator": sp.sstr(clean(bg.ansatz_denominator(p))),
        "known_invariants": [{"label": lab, "note": note}
                             for lab, _poly, note in bg.known(p)],
        "notes": bg.notes,
    }
