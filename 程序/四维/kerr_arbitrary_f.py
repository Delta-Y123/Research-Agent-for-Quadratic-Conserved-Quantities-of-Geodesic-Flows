#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sympy as sp

from kerr_eq54_first_order import (
    t, r, u, phi, M, a, pt, pr, pu, pp, X, P, D, w, S, B, J,
    clean, require_zero, metric, block_inverse, eq54, quadratic,
    polynomial, scale, add, expression, poisson, CUBICS, check_cubic,
    recover_coefficient, run_case,
)

def structural_reason():
    print("=" * 78)
    print("结构原因：为什么任意 f(r) 都能延续 Carter 量")
    print("=" * 78)

    W = w * pu ** 2 + (pp + a * w * pt) ** 2 / w
    F = sp.Function("F")(r)
    sf, af = F + a ** 2 * u ** 2, F + a ** 2
    R_F = D * pr ** 2 - (af * pt + a * pp) ** 2 / D

    print("\n[1] 精确族的 Stäckel 加法分离")
    print("    S_F = F + a^2 u^2,   A_F = F + a^2,   F = F(r)")
    print("    2 S_F H_F = R_F(r) + W(u)")
    print("    R_F =", R_F)
    print("    W   =", W)
    print("    R_F 含 u ?", u in R_F.free_symbols)
    print("    W   含 r ?", r in W.free_symbols)
    print("    => 径向 R_F 与角向 W 解耦；W 完全不含径向剖面 F。")

    HF = polynomial((R_F + W) / (2 * sf))
    CF = add(polynomial(W), scale(HF, -2 * a ** 2 * u ** 2))
    print("\n[2] Carter 型不变量  C_F = W - 2 a^2 u^2 H_F")
    print("    = (F W - a^2 u^2 R_F)/(F + a^2 u^2)  （等价写法）")
    check_cubic(poisson(CF, HF), "exact family, arbitrary F(r)")
    print("    => 对任意 F(r) 精确守恒；三个幂次只是 F=r^2+eps f 的实例。")

    print("\n[3] 一阶展开  C_1 = -2 a^2 u^2 H_1  （对任意 f 恒等）")
    print("    故 {C_1,H_0}+{C_0,H_1}=0 对任意 f 自动成立。")
    print("    这解释了为什么三个幂次、以及下面任何光滑 f 都会通过。")

def main():
    structural_reason()

    print("\n" + "=" * 78)
    print("任意 f(r) 的一阶符号验证")
    print("=" * 78)

    gi = block_inverse(metric(r ** 2))
    H0 = quadratic(gi)
    C0 = add(polynomial(w * pu ** 2 + (pp + a * w * pt) ** 2 / w),
             scale(H0, -2 * a ** 2 * u ** 2))
    check_cubic(poisson(C0, H0), "background {C0,H0}")

    f_abstract = sp.Function("f")(r)
    run_case("arbitrary_f", f_abstract, gi, H0, C0)

    tests = [
        ("M^3/r", M ** 3 / r),
        ("M^4/r^2", M ** 4 / r ** 2),
        ("M^5/r^3", M ** 5 / r ** 3),
        ("M^2 r/(r^2+M^2)", M ** 2 * r / (r ** 2 + M ** 2)),
        ("M^2 (r-M)^2/(r^2+M^2)^2", M ** 2 * (r - M) ** 2 / (r ** 2 + M ** 2) ** 2),
        ("exp(-r/M)", sp.exp(-r / M)),
    ]
    for name, f in tests:
        try:
            run_case(name, f, gi, H0, C0)
        except Exception as exc:
            print(f"  [SKIP/FAIL] {name}: {exc}")

if __name__ == "__main__":
    main()
