#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from itertools import combinations_with_replacement, permutations

import sympy as sp
from sympy import (symbols, diff, expand, simplify, trigsimp, together,
                   Matrix, Poly, sin, cos, fraction)

from killing_tensor_verify_4d import poisson_cubic_coeffs


def killing_conditions_uv(K_up, g_inv, u, r):
    n = 4
    dKr = [[diff(K_up[a, b], r) for b in range(n)] for a in range(n)]
    dGr = [[diff(g_inv[a, b], r) for b in range(n)] for a in range(n)]
    dKu = [[diff(K_up[a, b], u) for b in range(n)] for a in range(n)]
    dGu = [[diff(g_inv[a, b], u) for b in range(n)] for a in range(n)]

    out = []
    for (a, b, c) in combinations_with_replacement(range(n), 3):
        P = 0
        Q = 0
        for (al, be, mu) in set(permutations((a, b, c))):
            # d/dr 项
            P += dKr[al][be] * g_inv[1, mu] - K_up[1, mu] * dGr[al][be]
            # d/dtheta = 2 sin cos * d/du
            Q += 2 * (dKu[al][be] * g_inv[2, mu] - K_up[2, mu] * dGu[al][be])
        out.append(((a, b, c), P, Q))
    return out


def clear_to_equations(expr, u, r, params):
    frac = together(expr)
    num, den = fraction(frac)
    num = expand(num)
    if num == 0:
        return []
    P = Poly(num, u, r, *params)
    eqs = []
    for monom, coeff in zip(P.monoms(), P.coeffs()):
        coeff = expand(coeff)
        if coeff != 0:
            eqs.append(coeff)
    return eqs


def search_killing_tensor(basis, g_inv_u, u, r, params, label=""):
    t0 = time.time()
    n = len(basis)
    c = symbols(f"c0:{n}")

    K_up = sum((ci * Ti for ci, (nm, Ti) in zip(c, basis)), sp.zeros(4))
    K_up = (K_up + K_up.T) / 2

    conds = killing_conditions_uv(K_up, g_inv_u, u, r)

    equations = []
    for (abc, P, Q) in conds:
        equations.extend(clear_to_equations(P, u, r, params))
        equations.extend(clear_to_equations(Q, u, r, params))

    eqs = list(dict.fromkeys(equations))

    M = sp.zeros(len(eqs), n)
    for i, eq in enumerate(eqs):
        p = Poly(expand(eq), c)
        for j, cj in enumerate(c):
            M[i, j] = p.coeff_monomial(cj)

    rank = M.rank()
    nullity = M.cols - rank
    ns = M.nullspace()
    elapsed = time.time() - t0

    print("=" * 72)
    print(f"搜索结果 [{label}]")
    print("=" * 72)
    print(f"  基张量个数        = {n}")
    print(f"  独立方程个数      = {len(eqs)}")
    print(f"  系数矩阵形状      = {M.shape}")
    print(f"  秩 rank           = {rank}")
    print(f"  解空间维数 nullity = {nullity}  (耗时 {elapsed:.2f} s)")
    print(f"  未知系数 $c_i$      = {list(c)}")
    return ns, rank, nullity, M, len(eqs)


def verify_solution(vec, basis, g_inv_th, u, th, coords):
    K_up_u = sum((vi * Ti for vi, (nm, Ti) in zip(vec, basis)), sp.zeros(4))
    K_up_th = K_up_u.subs(u, sin(th) ** 2)
    K_up_th = (K_up_th + K_up_th.T) / 2
    coeffs = poisson_cubic_coeffs(K_up_th, g_inv_th, coords)
    nonzero = 0
    for (abc, cval) in coeffs:
        cval = expand(cval)
        if cval != 0 and not (simplify(cval) == 0 or simplify(trigsimp(cval)) == 0):
            nonzero += 1
    return nonzero


def report_solutions(ns, basis, g_inv_th, u, th, coords, momenta):
    names = [nm for nm, _ in basis]
    for k, v in enumerate(ns):
        vec = list(v)
        lead = next((x for x in vec if x != 0), 1)
        vec = [sp.simplify(x / lead) for x in vec]

        K_up_u = sum((vi * Ti for vi, (nm, Ti) in zip(vec, basis)), sp.zeros(4))
        K_up_th = K_up_u.subs(u, sin(th) ** 2)
        K_up_th = (K_up_th + K_up_th.T) / 2
        Q = sum(K_up_th[i, j] * momenta[i] * momenta[j]
                for i in range(4) for j in range(4))
        nz = verify_solution(vec, basis, g_inv_th, u, th, coords)

        print(f"\n  解 {k}  ($c_i$, 归一化):")
        for nm, vi in zip(names, vec):
            if vi != 0:
                print(f"      {nm:32s} $\\to$ {sp.latex(vi)}")
        print(f"      $Q = {sp.latex(sp.simplify(Q))}$")
        status = "通过" if nz == 0 else f"失败({nz} 非零项)"
        print(f"      独立复核 $\\{{Q,H\\}}=0$ : {status}")


def search_kerr_carter():
    print("\n" + "#" * 72)
    print("# 搜索 (1) Kerr：Carter 张量 Ansatz")
    print("#" * 72)
    t, r, th, ph = symbols("t r theta phi", real=True)
    pt, pr, pth, pph = symbols("pt pr ptheta pphi")
    coords = [t, r, th, ph]
    momenta = [pt, pr, pth, pph]

    M, a = symbols("M a", positive=True)
    u = symbols("u", real=True)

    Sigma_u = r ** 2 + a ** 2 * (1 - u)
    Delta = r ** 2 - 2 * M * r + a ** 2

    g_inv_u = Matrix([
        [-((r ** 2 + a ** 2) ** 2 - a ** 2 * Delta * u) / (Sigma_u * Delta),
         0, 0, -2 * M * a * r / (Sigma_u * Delta)],
        [0, Delta / Sigma_u, 0, 0],
        [0, 0, 1 / Sigma_u, 0],
        [-2 * M * a * r / (Sigma_u * Delta), 0, 0,
         (Delta - a ** 2 * u) / (Sigma_u * Delta * u)],
    ])

    Sigma = r ** 2 + a ** 2 * cos(th) ** 2
    g_inv_th = Matrix([
        [-((r ** 2 + a ** 2) ** 2 - a ** 2 * Delta * sin(th) ** 2) / (Sigma * Delta),
         0, 0, -2 * M * a * r / (Sigma * Delta)],
        [0, Delta / Sigma, 0, 0],
        [0, 0, 1 / Sigma, 0],
        [-2 * M * a * r / (Sigma * Delta), 0, 0,
         (Delta - a ** 2 * sin(th) ** 2) / (Sigma * Delta * sin(th) ** 2)],
    ])

    Ett = Matrix.diag(1, 0, 0, 0)
    Ethth = Matrix.diag(0, 0, 1, 0)
    Ephiphi = Matrix.diag(0, 0, 0, 1)

    basis = [
        (r"$g^{\mu\nu}$ (度规, $Q=2H$)", g_inv_u),
        (r"$a^2\cos^2\theta\, g^{\mu\nu}$", a ** 2 * (1 - u) * g_inv_u),
        (r"$a^2\cos^2\theta\, e_t\otimes e_t$", a ** 2 * (1 - u) * Ett),
        (r"$e_\theta\otimes e_\theta$", Ethth),
        (r"$\cot^2\theta\, e_\phi\otimes e_\phi$", (1 - u) / u * Ephiphi),
    ]

    ns, rank, nullity, M, neq = search_killing_tensor(
        basis, g_inv_u, u, r, (M, a), label="Kerr Carter Ansatz")

    report_solutions(ns, basis, g_inv_th, u, th, coords, momenta)


def search_schwarzschild():
    print("\n" + "#" * 72)
    print("# 搜索 (2) Schwarzschild：坐标基张量 Ansatz")
    print("#" * 72)
    t, r, th, ph = symbols("t r theta phi", real=True)
    pt, pr, pth, pph = symbols("pt pr ptheta pphi")
    coords = [t, r, th, ph]
    momenta = [pt, pr, pth, pph]

    M = symbols("M", positive=True)
    u = symbols("u", real=True)
    f = 1 - 2 * M / r

    g_inv_u = Matrix.diag(-1 / f, f, 1 / r ** 2, 1 / (r ** 2 * u))
    g_inv_th = Matrix.diag(-1 / f, f, 1 / r ** 2, 1 / (r ** 2 * sin(th) ** 2))

    Ett = Matrix.diag(1, 0, 0, 0)
    Err = Matrix.diag(0, 1, 0, 0)
    Ethth = Matrix.diag(0, 0, 1, 0)
    Ephiphi = Matrix.diag(0, 0, 0, 1)

    basis = [
        (r"$g^{\mu\nu}$ (度规)", g_inv_u),
        (r"$e_t\otimes e_t$", Ett),
        (r"$e_r\otimes e_r$", Err),
        (r"$e_\theta\otimes e_\theta$", Ethth),
        (r"$e_\phi\otimes e_\phi$", Ephiphi),
    ]

    ns, rank, nullity, M, neq = search_killing_tensor(
        basis, g_inv_u, u, r, (M,), label="Schwarzschild 坐标基 Ansatz")

    report_solutions(ns, basis, g_inv_th, u, th, coords, momenta)


def main():
    search_kerr_carter()
    search_schwarzschild()
    print("\n" + "#" * 72)
    print("全部搜索结束。")


if __name__ == "__main__":
    main()