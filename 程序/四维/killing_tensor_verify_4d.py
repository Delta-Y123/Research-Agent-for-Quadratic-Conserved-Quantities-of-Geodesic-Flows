#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from itertools import combinations_with_replacement, permutations

import sympy as sp
from sympy import (symbols, diff, expand, simplify, trigsimp,
                   Matrix, Rational, sin, cos)


def poisson_bracket(f, g, coords, momenta):
    return sum(diff(f, q) * diff(g, pp) - diff(f, pp) * diff(g, q)
               for q, pp in zip(coords, momenta))


def hamiltonian(g_inv, momenta):
    n = len(momenta)
    return Rational(1, 2) * sum(
        g_inv[i, j] * momenta[i] * momenta[j]
        for i in range(n) for j in range(n))


def raise_indices(K_cov, g_inv):
    return g_inv * K_cov * g_inv


def poisson_cubic_coeffs(K_up, g_inv, coords):
    n = len(coords)
    dK = [[[diff(K_up[a, b], q) for b in range(n)] for a in range(n)]
          for q in coords]
    dG = [[[diff(g_inv[a, b], q) for b in range(n)] for a in range(n)]
          for q in coords]

    out = []
    for (a, b, c) in combinations_with_replacement(range(n), 3):
        coeff = 0
        for (al, be, mu) in set(permutations((a, b, c))):
            for l in range(n):
                coeff += dK[l][al][be] * g_inv[l, mu] - K_up[l, mu] * dG[l][al][be]
        out.append(((a, b, c), coeff))
    return out


def check_zero(expr):
    for e in (simplify(expr), simplify(trigsimp(expand(expr)))):
        if e == 0:
            return True
    return False


def verify_killing_tensor(g, K, coords, momenta, covariant=True, g_inv=None,
                          label="", verbose=False):
    t0 = time.time()
    n = len(coords)

    if g_inv is None:
        g_inv = g.inv()

    K_up = K if not covariant else raise_indices(K, g_inv)
    K_up = (K_up + K_up.T) / 2

    H = hamiltonian(g_inv, momenta)
    Q = sum(K_up[i, j] * momenta[i] * momenta[j]
            for i in range(n) for j in range(n))

    coeffs = poisson_cubic_coeffs(K_up, g_inv, coords)
    nonzero = []
    for (abc, cval) in coeffs:
        cval = expand(cval)
        if cval == 0:
            continue
        if not check_zero(cval):
            nonzero.append((abc, simplify(cval)))

    ok = (len(nonzero) == 0)
    elapsed = time.time() - t0

    print("=" * 72)
    print(f"验证结果 [{label}]")
    print("=" * 72)
    if verbose:
        print("逆度规 $g^{-1} = " + sp.latex(g_inv) + "$")
        print("逆变张量 $K^{\\mu\\nu} = " + sp.latex(K_up) + "$")
        print("$Q = " + sp.latex(Q) + "$")
        print("$H = " + sp.latex(H) + "$")
    if ok:
        print(f"  通过：$\\{{Q,H\\}}=0$   (耗时 {elapsed:.2f} s)")
    else:
        print(f"  失败：存在 {len(nonzero)} 个非零系数 (耗时 {elapsed:.2f} s)")
        for abc, cval in nonzero:
            idx = "p".join(str(i) for i in abc)
            print(f"    $p_{{{idx}}}$ 系数 $\\to$ " + sp.latex(cval))
    print()
    return ok, nonzero, Q


def make_phase_space():
    t, r, th, ph = symbols("t r theta phi", real=True)
    pt, pr, pth, pph = symbols("pt pr ptheta pphi")
    return [t, r, th, ph], [pt, pr, pth, pph]


def test_minkowski():
    print("\n" + "#" * 72)
    print("# 测试 (a) Minkowski 度规")
    print("#" * 72)
    coords, momenta = make_phase_space()
    t, r, th, ph = coords

    g = Matrix.diag(-1, 1, r ** 2, r ** 2 * sin(th) ** 2)

    K_eta = Matrix.diag(-1, 1, r ** 2, r ** 2 * sin(th) ** 2)
    verify_killing_tensor(g, K_eta, coords, momenta,
                          label=r"Minkowski: $K=\eta$ (度规本身)")

    K_L2 = Matrix.diag(0, 0, r ** 4, r ** 4 * sin(th) ** 2)
    verify_killing_tensor(g, K_L2, coords, momenta,
                          label=r"Minkowski: $K=r^4(d\theta^2+\sin^2\theta\, d\phi^2)$ 即 $L^2$ 张量")


def test_schwarzschild():
    print("\n" + "#" * 72)
    print("# 测试 (b) Schwarzschild 度规")
    print("#" * 72)
    coords, momenta = make_phase_space()
    t, r, th, ph = coords

    M = symbols("M", positive=True)
    f = 1 - 2 * M / r
    g = Matrix.diag(-f, 1 / f, r ** 2, r ** 2 * sin(th) ** 2)

    K_cov = Matrix.diag(f ** 2, 0, 0, (r ** 2 * sin(th) ** 2) ** 2)
    verify_killing_tensor(g, K_cov, coords, momenta,
                          label=r"Schwarzschild: $K=d/dt\otimes d/dt+d/d\phi\otimes d/d\phi$ ($Q=p_t^2+p_\phi^2$)")

    K_L2_cov = Matrix.diag(0, 0, r ** 4, r ** 4 * sin(th) ** 2)
    verify_killing_tensor(g, K_L2_cov, coords, momenta,
                          label=r"Schwarzschild: $L^2=p_\theta^2+p_\phi^2/\sin^2\theta$")


def test_kerr():
    print("\n" + "#" * 72)
    print("# 测试 (c) Kerr 度规 (Boyer-Lindquist)，Carter 张量")
    print("#" * 72)
    coords, momenta = make_phase_space()
    t, r, th, ph = coords

    M, a = symbols("M a", positive=True)
    Sigma = r ** 2 + a ** 2 * cos(th) ** 2
    Delta = r ** 2 - 2 * M * r + a ** 2

    g = Matrix([
        [-(1 - 2 * M * r / Sigma), 0, 0, -2 * M * a * r * sin(th) ** 2 / Sigma],
        [0, Sigma / Delta, 0, 0],
        [0, 0, Sigma, 0],
        [-2 * M * a * r * sin(th) ** 2 / Sigma, 0, 0,
         (r ** 2 + a ** 2 + 2 * M * a ** 2 * r * sin(th) ** 2 / Sigma) * sin(th) ** 2],
    ])

    g_inv = Matrix([
        [-((r ** 2 + a ** 2) ** 2 - a ** 2 * Delta * sin(th) ** 2) / (Sigma * Delta),
         0, 0, -2 * M * a * r / (Sigma * Delta)],
        [0, Delta / Sigma, 0, 0],
        [0, 0, 1 / Sigma, 0],
        [-2 * M * a * r / (Sigma * Delta), 0, 0,
         (Delta - a ** 2 * sin(th) ** 2) / (Sigma * Delta * sin(th) ** 2)],
    ])

    K_up = (- a ** 2 * cos(th) ** 2 * g_inv
            + Matrix.diag(-a ** 2 * cos(th) ** 2, 0, 1, cos(th) ** 2 / sin(th) ** 2))

    verify_killing_tensor(g, K_up, coords, momenta, covariant=False, g_inv=g_inv,
                          label="Kerr: Carter 张量（逆变标准形式）")


def main():
    test_minkowski()
    test_schwarzschild()
    test_kerr()
    print("#" * 72)
    print("全部测试结束。")


if __name__ == "__main__":
    main()