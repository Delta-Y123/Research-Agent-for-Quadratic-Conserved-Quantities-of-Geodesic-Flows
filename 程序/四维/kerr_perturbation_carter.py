#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

import sympy as sp
from sympy import (symbols, Matrix, sin, cos, sin as sn, cos as cs,
                   Rational, expand, simplify, together, fraction, diff, Poly,
                   trigsimp)

t, r, theta, phi = symbols("t r theta phi", real=True)
pt, pr, ptheta, pphi = symbols("pt pr ptheta pphi")
M, a, epsilon = symbols("M a epsilon", positive=True)

coords = [t, r, theta, phi]
momenta = [pt, pr, ptheta, pphi]

s2 = symbols("s2", positive=True)
w = symbols("w", real=True)

def poisson_bracket(f, g):
    return sum(diff(f, q) * diff(g, p) - diff(f, p) * diff(g, q)
               for q, p in zip(coords, momenta))

def build_kerr(spin):
    Sigma = r ** 2 + spin ** 2 * cos(theta) ** 2
    Delta = r ** 2 - 2 * M * r + spin ** 2
    g0_inv = Matrix([
        [-((r ** 2 + spin ** 2) ** 2 - spin ** 2 * Delta * sin(theta) ** 2) / (Sigma * Delta),
         0, 0, -2 * M * spin * r / (Sigma * Delta)],
        [0, Delta / Sigma, 0, 0],
        [0, 0, 1 / Sigma, 0],
        [-2 * M * spin * r / (Sigma * Delta), 0, 0,
         (Delta - spin ** 2 * sin(theta) ** 2) / (Sigma * Delta * sin(theta) ** 2)],
    ])
    H0 = Rational(1, 2) * sum(g0_inv[i, j] * momenta[i] * momenta[j]
                              for i in range(4) for j in range(4))
    Q0 = ptheta ** 2 + pphi ** 2 / sin(theta) ** 2
    return g0_inv, H0, Q0

def carter_constant(g0_inv, spin):
    c2 = cos(theta) ** 2
    s2v = sin(theta) ** 2
    K_up = -spin ** 2 * c2 * g0_inv + sp.diag(-spin ** 2 * c2, 0, 1, c2 / s2v)
    return sum(K_up[i, j] * momenta[i] * momenta[j]
               for i in range(4) for j in range(4))

def get_h1_and_q1(h_mu_nu_matrix, Q1_ansatz, g0_inv, H0, Q0):
    g1_inv = -g0_inv * h_mu_nu_matrix * g0_inv
    H1 = Rational(1, 2) * sum(g1_inv[i, j] * momenta[i] * momenta[j]
                              for i in range(4) for j in range(4))
    EQ = expand(poisson_bracket(Q1_ansatz, H0) + poisson_bracket(Q0, H1))
    return H1, EQ, g1_inv

def pb_uw(f, g):
    return (diff(f, r) * diff(g, pr) - diff(f, pr) * diff(g, r)
            + 2 * w * (diff(f, s2) * diff(g, ptheta) - diff(f, ptheta) * diff(g, s2)))

def kerr_inv_uw(spin):
    Sigma = r ** 2 + spin ** 2 * (1 - s2)
    Delta = r ** 2 - 2 * M * r + spin ** 2
    return Matrix([
        [-((r ** 2 + spin ** 2) ** 2 - spin ** 2 * Delta * s2) / (Sigma * Delta),
         0, 0, -2 * M * spin * r / (Sigma * Delta)],
        [0, Delta / Sigma, 0, 0],
        [0, 0, 1 / Sigma, 0],
        [-2 * M * spin * r / (Sigma * Delta), 0, 0,
         (Delta - spin ** 2 * s2) / (Sigma * Delta * s2)],
    ])

def carter_uw(gi, spin):
    c2 = 1 - s2
    K = -spin ** 2 * c2 * gi + sp.diag(-spin ** 2 * c2, 0, 1, c2 / s2)
    return sum(K[i, j] * momenta[i] * momenta[j]
               for i in range(4) for j in range(4))

def main():
    print("=" * 78)
    print("Kerr 背景一阶微扰：二次守恒量延续性 —— 工具 + 诊断 + 正例")
    print("=" * 78)

    print("\n[A] 诊断：Q0 = ptheta^2 + pphi^2/sin^2theta 对 Kerr 是否守恒？")
    gi_kerr, H0_kerr, Q0 = build_kerr(a)
    pb_diag = sp.expand(poisson_bracket(Q0, H0_kerr))
    P = sp.Poly(pb_diag, *momenta)
    nz = [(m, c) for m, c in zip(P.monoms(), P.coeffs()) if c != 0]
    print(f"    {Q0} vs H0(kerr):  {len(nz)} 个非零动量单项式")
    p_coef = P.coeff_monomial(pt ** 2 * ptheta)
    print(f"    其中 p_t^2 p_theta 系数 = {sp.simplify(p_coef)}")
    print("    => 对 a != 0，{{Q0,H0}} != 0，Q0 不是 Kerr 的守恒量。")

    print("\n" + "-" * 78)
    print("[B] 正例 A：Schwarzschild 极限 a=0，h_tt=c/r, h_rr=d/r，Q1=0")
    print("-" * 78)

    c, d = symbols("c d")
    gi_s, H0_s, _ = build_kerr(0)

    h_s = sp.zeros(4)
    h_s[0, 0] = c / r
    h_s[1, 1] = d / r
    Q1_zero = sp.Integer(0)

    H1_s, EQ_s, g1_s = get_h1_and_q1(h_s, Q1_zero, gi_s, H0_s, Q0)

    print("    H1 =", sp.simplify(H1_s))
    print("    g1_inv 非零分量:", [((i, j), sp.simplify(g1_s[i, j]))
                                 for i in range(4) for j in range(4)
                                 if sp.simplify(g1_s[i, j]) != 0])
    print("    Q1 = 0")
    print("    EQ = {Q1,H0} + {Q0,H1} =", sp.simplify(EQ_s) if EQ_s != 0 else 0)
    ok_s = sp.simplify(EQ_s) == 0
    print(f"    验证 {{Q1,H0}} + {{Q0,H1}} = 0 ？ {'通过' if ok_s else '失败'}")
    print("    说明：h_tt,h_rr 只依赖 r，扰动仍球对称，Q0=L^2-L_z^2 保持守恒，")
    print("          故 Q1=0 即为一阶守恒解（这是题目许可的 Schwarzschild 平凡正例）。")

    print("\n    （附加）Schwarzschild + 非球对称 h_{t phi}=c/r * sin^2theta，搜索非零 Q1：")
    h_s2 = sp.zeros(4)
    h_s2[0, 3] = c / r * sin(theta) ** 2
    h_s2[3, 0] = h_s2[0, 3]
    q = symbols("q0:9")

    Q1_ansatz = sin(theta) ** 2 * sum(
        r ** (-k) * (q[3 * (k - 1) + 0] * pt * pphi
                     + q[3 * (k - 1) + 1] * pphi ** 2
                     + q[3 * (k - 1) + 2] * pt ** 2)
        for k in (1, 2, 3))
    H1_s2, EQ_s2, _ = get_h1_and_q1(h_s2, Q1_ansatz, gi_s, H0_s, Q0)
    print(f"    EQ 对动量 {len(sp.Poly(EQ_s2, *momenta).monoms())} 个单项式非平凡；")
    print("    （完整的系数提取/求解见正例 B 的流程；这里主要展示 H1 与 EQ 的构造。）")

    print("\n" + "-" * 78)
    print("[C] 正例 B：Kerr + 真 Carter 常量 Q_C，h_{t phi} 扰动，搜索 Q1")
    print("-" * 78)

    gi_uw = kerr_inv_uw(a)
    QC = carter_uw(gi_uw, a)
    H0_uw = Rational(1, 2) * sum(gi_uw[i, j] * momenta[i] * momenta[j]
                                 for i in range(4) for j in range(4))

    h_val = (sp.symbols("c") * M ** 5 / r ** 4 * s2
             * (1 + 12 * M / (7 * r) + 27 * M ** 2 / (10 * r ** 2)))
    h_uw = sp.zeros(4)
    h_uw[0, 3] = h_val
    h_uw[3, 0] = h_val
    g1_uw = -gi_uw * h_uw * gi_uw
    H1_uw = Rational(1, 2) * sum(g1_uw[i, j] * momenta[i] * momenta[j]
                                 for i in range(4) for j in range(4))

    qq = symbols("qq0:6")
    KMAX = 2
    Q1_uw = s2 * sum(
        r ** (-k) * (qq[3 * (k - 1) + 0] * pt * pphi
                     + qq[3 * (k - 1) + 1] * pphi ** 2
                     + qq[3 * (k - 1) + 2] * pt ** 2)
        for k in range(1, KMAX + 1))

    EQ_uw = sp.expand(pb_uw(Q1_uw, H0_uw) + pb_uw(QC, H1_uw))

    def clear_denoms(expr):
        num, den = sp.fraction(sp.together(sp.expand(expr)))
        return sp.expand(num)

    unknowns = list(qq)
    equations = []
    for part in (sp.expand(EQ_uw.subs(w, 0)), sp.expand(sp.diff(EQ_uw, w))):
        if part == 0:
            continue
        num = clear_denoms(part)
        if num == 0:
            continue
        Pq = sp.Poly(num, r, s2)
        for (rm, sm), cval in zip(Pq.monoms(), Pq.coeffs()):
            cval = sp.expand(cval)
            if cval == 0:
                continue
            cp = sp.Poly(cval, *unknowns)
            mvec = [cp.coeff_monomial(x) for x in unknowns]
            rest = sp.expand(cval - sum(mvec[i] * unknowns[i]
                                        for i in range(len(unknowns))))
            equations.append((mvec, rest))

    print(f"    Q1 未知系数 {unknowns}；方程数 = {len(equations)}")
    Mm = Matrix([e[0] for e in equations])
    bb = Matrix([-e[1] for e in equations])
    rankM = Mm.rank()
    print(f"    M 形状 {Mm.shape}, rank(M) = {rankM}")

    Y = Mm.T.nullspace()
    bad = 0
    for y in Y:
        if sp.expand((y.T * bb)[0]) != 0:
            bad += 1
    if bad == 0:
        print("    => 一致，存在 Q1 解。")
        sol = sp.linsolve((Mm, bb))
        print("    解:", sol)
    else:
        print(f"    => 无解（{bad} 个左零向量给出非零 y^T b）。")

if __name__ == "__main__":
    main()
