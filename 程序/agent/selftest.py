#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""内核验收测试。

    python selftest.py           # 快速（约 5 s）
    python selftest.py --full    # 含 Kerr 一阶分类（约 2-3 min）

验收标准来自手工推导与旧脚本 `程序/二维`、`程序/四维` 已经验证过的数字：
  * 所有背景的已知二次不变量必须精确满足 {Q,H}=0；
  * 二维模型 d=2 的秩必须是 16 / 17 / 17，解空间 2 / 1 / 1；
  * Kerr 一阶分类必须是 4 允许 / 12 排除，联合核 9 维且全部通过回代。
"""

from __future__ import annotations

import argparse
import sys
from time import perf_counter

import sympy as sp

import backgrounds as bgmod
import families
import solvers
from engine import is_zero, poisson


def check_invariants() -> bool:
    cases = [
        ("conformal2d", {bgmod.eps: sp.Integer(0)}),
        ("conformal2d", {bgmod.eps: sp.Integer(1)}),
        ("minkowski", {}),
        ("schwarzschild", {bgmod.M: sp.Integer(1)}),
        ("kerr", {bgmod.M: sp.Integer(1), bgmod.a: sp.Rational(1, 2)}),
        ("kerr", {bgmod.M: sp.Integer(1), bgmod.a: sp.Integer(0)}),
    ]
    ok = True
    for key, values in cases:
        bg = bgmod.get(key)
        H = bg.H(values)
        for label, poly, _note in bg.known(values):
            good = is_zero(poisson(poly, H, bg.coords, bg.momenta))
            ok &= good
            print(f"  {'PASS' if good else 'FAIL'} {{{label}, H}} = 0 "
                  f"@ {key} {values}")
    # Q0 只在 epsilon=0 出现
    bg = bgmod.get("conformal2d")
    has_q0 = any(lab == "Q0" for lab, _, _ in bg.known({bgmod.eps: sp.Integer(1)}))
    ok &= not has_q0
    print(f"  {'PASS' if not has_q0 else 'FAIL'} Q0 不在 epsilon=1 的已知不变量表中")
    return ok


def check_2d_search() -> bool:
    bg = bgmod.get("conformal2d")
    expected = [("0", sp.Integer(0), 16, 2),
                ("1", sp.Integer(1), 17, 1),
                ("-1/4", sp.Rational(-1, 4), 17, 1),
                ("2", sp.Integer(2), 17, 1)]
    ok = True
    for label, value, want_rank, want_null in expected:
        out, _sols, _gens, _labels = solvers.killing_search(
            bg, {bgmod.eps: value}, degree=2)
        good = (out["rank"] == want_rank and out["nullity"] == want_null
                and out["unknowns"] == 18
                and out["certificate"]["verified_all"])
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'} epsilon={label:>5}: "
              f"unknowns={out['unknowns']} rank={out['rank']} "
              f"nullity={out['nullity']} (want {want_rank}/{want_null})")
    return ok


def check_2d_classification() -> bool:
    """真实的 epsilon x^2y^2 扰动必须排除 Q0 的延续。"""
    import tools
    result = tools.dispatch("first_order_classification", {
        "background": "conformal2d", "q0": "Q0", "family": "components",
        "components": ["00", "11"], "profiles": ["x^2*y^2"],
        "q1_exponents": {"x": [-2, -1, 0, 1, 2], "y": [0, 1, 2]}}, {})
    verdict = result.get("verdict") or {}
    ok = (not result.get("error") and verdict
          and all(v == "EXCLUDED" for v in verdict.values())
          and result["certificate"]["verified_all_kernel_vectors"])
    print(f"  {'PASS' if ok else 'FAIL'} 2D 一阶分类: {verdict} "
          f"({result.get('seconds')}s)")
    return bool(ok)


def check_kerr_classification() -> bool:
    bg = bgmod.get("kerr")
    values = {bgmod.M: sp.Integer(1), bgmod.a: sp.Rational(1, 2)}
    specs = families.eq54_family(values) + families.component_family()
    result = solvers.first_order_classification(
        bg, values, "Q_C", specs, q1_degree=2, q1_axes=(bgmod.r, bgmod.u),
        q1_exponents={"r": [-3, -2, -1, 0, 1, 2], "u": [0, 2]},
        augment_weights=["a^2*u^2"], nr=4, nu=4)
    verdict = result["verdict"]
    allowed = [k for k, v in verdict.items() if v == "ALLOWED"]
    ok = (len(verdict) == 16 and len(allowed) == 4
          and all(k.startswith("eq54") for k in allowed)
          and result["kernel_dimension"] == 9
          and result["homogeneous_dimension"] == 5
          and result["certificate"]["verified_all_kernel_vectors"])
    print(f"  {'PASS' if ok else 'FAIL'} Kerr 一阶分类: kernel="
          f"{result['kernel_dimension']} allowed={len(allowed)}/16 "
          f"({result['seconds']}s)")
    return ok


def check_eq54_slice() -> bool:
    """eq54_h 必须等于精确族 F = r^2 + eps*f 对 eps 的一阶导数。

    式 (54) 本身**不含** f：它用四个任意函数 (deltaV, deltaq, deltalambda,
    deltagamma) 给出最一般的稳态轴对称扰动。我们把四个函数收成一个径向剖面 f，
    等价于把 Kerr 的径向函数作形变 F = r^2 + eps*f(r) 后取一阶项。

    families.eq54_h 是对这条切片的**重新实现**，所以这里把它钉在几何定义上，
    而不是钉在某个抄来的公式上。
    """
    import families as fam
    from families import DELTA, W, a, eq54_h, r, u

    M = fam.M
    f = sp.Function("f")(r)
    eps = sp.Symbol("eps")
    F = r ** 2 + eps * f
    SF, AF = F + a ** 2 * u ** 2, F + a ** 2

    g = sp.zeros(4)
    g[0, 0] = -(DELTA - a ** 2 * W) / SF
    g[0, 3] = g[3, 0] = -a * W * (AF - DELTA) / SF
    g[3, 3] = W * (AF ** 2 - a ** 2 * DELTA * W) / SF
    g[1, 1] = SF / DELTA
    g[2, 2] = SF / W

    h_exact = g.applyfunc(lambda e: sp.diff(e, eps).subs(eps, 0))
    h_ours = eq54_h({M: M, a: a}, f)

    pairs = [("tt", 0, 0), ("rr", 1, 1), ("uu", 2, 2),
             ("ru", 1, 2), ("tphi", 0, 3), ("phiphi", 3, 3)]
    ok = True
    for name, i, j in pairs:
        same = sp.simplify(sp.expand(h_exact[i, j] - h_ours[i, j])) == 0
        ok &= same
        if not same:
            print(f"  FAIL h_{name} 与精确族一阶导数不一致")

    # 单函数切片必然满足的分量配合比例
    ratios = [
        ("h_tphi = -a*w*h_tt", h_ours[0, 3] + a * W * h_ours[0, 0]),
        ("h_uu = (D/w)*h_rr", h_ours[2, 2] - DELTA / W * h_ours[1, 1]),
        ("h_ru = 0", h_ours[1, 2]),
    ]
    for label, expr in ratios:
        same = sp.simplify(expr) == 0
        ok &= same
        if not same:
            print(f"  FAIL 配合比例不成立：{label}")

    print(f"  {'PASS' if ok else 'FAIL'} eq54 单函数切片：6 个分量与精确族"
          f"一阶导数一致，3 条配合比例成立")
    return ok


def check_simulator() -> bool:
    """模拟器桥接：符号 Q_C 与模拟器内置 Carter 必须吻到机器精度。

    这是 L2 闭环的验收项——符号层与数值层用的必须是同一个 K^{μν}。
    """
    import simulator as sim

    if not sim.available():
        print(f"  SKIP 找不到模拟器：{sim.CLI}")
        return True

    bg = bgmod.get("kerr")
    vals = {bgmod.M: sp.Integer(1), bgmod.a: sp.Rational(9, 10)}
    known = {lab: poly for lab, poly, _ in bg.known(vals)}
    data = sim.trace(preset="kerr_inclined")
    body = data["bodies"][0]
    samples = body["samples"]
    M, a = float(data["M"]), float(data["a"])

    res = sim.evaluate_quantity(known["Q_C"], bg, vals, samples, "Q_C",
                                M=M, a=a, mass=body["mass"])
    carter = sim.carter_from_samples(samples)
    # 我们的 Q_C = Carter + (L_z - aE)^2，扣掉偏移后应与模拟器一致
    offsets = [(float(s["Lz"]) - a * float(s["E"])) ** 2 for s in samples]
    shifted = [v - o for v, o in zip(res["values"], offsets)]
    diff = abs(shifted[0] - carter["v0"]) / max(abs(carter["v0"]), 1e-30)

    ok = (res["ok"] and res["rel_drift"] < 1e-6 and diff < 1e-6)
    print(f"  {'PASS' if ok else 'FAIL'} 模拟器桥接: {len(samples)} 采样点，"
          f"符号 Q_C 漂移 {res['rel_drift']:.1e}，"
          f"与内置 Carter 相对差 {diff:.1e}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="包含 Kerr 一阶分类（慢）")
    args = parser.parse_args()

    print(f"SymPy {sp.__version__}")
    checks = [("背景不变量", check_invariants),
              ("二维搜索", check_2d_search),
              ("二维一阶分类", check_2d_classification),
              ("eq54 切片", check_eq54_slice),
              ("模拟器桥接", check_simulator)]
    if args.full:
        checks.append(("Kerr 一阶分类", check_kerr_classification))

    ok = True
    for name, fn in checks:
        start = perf_counter()
        print(f"\n[{name}]")
        ok &= fn()
        print(f"  ({perf_counter() - start:.1f}s)")
    print("\nSELFTEST", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
