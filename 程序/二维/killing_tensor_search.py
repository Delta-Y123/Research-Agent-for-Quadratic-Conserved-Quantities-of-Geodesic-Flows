#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sympy as sp
from sympy import symbols, diff, expand, Poly, Matrix, Rational


def poisson_bracket(f, g):
    return (diff(f, x) * diff(g, px) - diff(f, px) * diff(g, x)
            + diff(f, y) * diff(g, py) - diff(f, py) * diff(g, y))


def gen_monomials(max_deg, x, y):
    mons = []
    for deg in range(max_deg + 1):
        for i in range(deg, -1, -1):
            j = deg - i
            mons.append(x ** i * y ** j)
    return mons


def build_equations(d):
    basis = gen_monomials(d, x, y)
    n = len(basis)

    a_coeffs = symbols(f"a0:{n}")
    b_coeffs = symbols(f"b0:{n}")
    c_coeffs = symbols(f"c0:{n}")
    unknowns = list(a_coeffs) + list(b_coeffs) + list(c_coeffs)

    A = sum(ai * mi for ai, mi in zip(a_coeffs, basis))
    B = sum(bi * mi for bi, mi in zip(b_coeffs, basis))
    C = sum(ci * mi for ci, mi in zip(c_coeffs, basis))

    lambda_eps = 1 + x ** 2 + 2 * y ** 2 + epsilon * x ** 2 * y ** 2
    K = px ** 2 + py ** 2

    Q_tilde = A * px ** 2 + B * px * py + C * py ** 2

    G = (lambda_eps * poisson_bracket(Q_tilde, K)
         - K * poisson_bracket(Q_tilde, lambda_eps)
         - Q_tilde * poisson_bracket(lambda_eps, K))
    G = expand(G)

    Gpoly = Poly(G, px, py, x, y)
    equations = []
    for monom, coeff in zip(Gpoly.monoms(), Gpoly.coeffs()):
        pxp, pyp, xp, yp = monom
        assert pxp + pyp == 3, f"发现非三次动量项: {monom}"
        equations.append(coeff)

    return (equations, unknowns, basis, A, B, C, Q_tilde, lambda_eps, G)


def coefficient_matrix(equations, unknowns, eps_value):
    eqs = [expand(eq.subs(epsilon, eps_value)) for eq in equations]
    eqs = [eq for eq in eqs if eq != 0]
    M = sp.zeros(len(eqs), len(unknowns))
    for i, eq in enumerate(eqs):
        p = Poly(eq, unknowns)
        for j, u in enumerate(unknowns):
            M[i, j] = p.coeff_monomial(u)
    return M, eqs, len(eqs)


def reconstruct(unknowns, basis, vec):
    # 未知量按 a0.., b0.., c0.. 排列
    n = len(basis)
    A = sum(vi * mi for vi, mi in zip(vec[:n], basis))
    B = sum(vi * mi for vi, mi in zip(vec[n:2 * n], basis))
    C = sum(vi * mi for vi, mi in zip(vec[2 * n:], basis))
    return A, B, C


def main():
    global x, y, px, py, epsilon
    x, y, px, py, epsilon = symbols("x y px py epsilon")

    print("=" * 72)
    print("二次守恒量（Killing 张量）搜索")
    print("度规: g_eps = (1 + x^2 + 2*y^2 + eps*x^2*y^2)(dx^2 + dy^2)")
    print("=" * 72)

    d_input = input("请输入多项式最高次数 d (默认 2): ").strip()
    d = int(d_input) if d_input else 2

    eps_input = input("请输入要测试的 epsilon 值，用逗号分隔 (默认 0, 1, -0.25): ").strip()
    if eps_input:
        eps_values = []
        for s in eps_input.split(','):
            s = s.strip()
            if '/' in s:
                eps_values.append(Rational(s))
            elif '.' in s:
                eps_values.append(sp.Float(s))
            else:
                eps_values.append(int(s))
    else:
        eps_values = [0, 1, Rational(-1, 4)]

    print(f"\n当前配置: d = {d}, eps_values = {eps_values}")

    (equations, unknowns, basis,
     A, B, C, Q_tilde, lambda_eps, G) = build_equations(d)

    print(f"\n[步骤 4] 单项式基（{len(basis)} 项）: {basis}")
    print(f"[步骤 4] 未知系数总数 = 3 * {len(basis)} = {len(unknowns)}")

    print(f"\n[步骤 6~7] 泊松括号多项式 G 的 (px,py,x,y) 单项式个数 = "
          f"{len(Poly(G, px, py, x, y).monoms())}")
    print(f"[步骤 7] 提取出的线性方程个数（含 epsilon 符号） = {len(equations)}")

    for ev in eps_values:
        M, eqs, neq = coefficient_matrix(equations, unknowns, ev)
        rank = M.rank()
        nullity = M.cols - rank

        lambda_ev = lambda_eps.subs(epsilon, ev)
        H_ev = (px ** 2 + py ** 2) / (2 * lambda_ev)

        print("\n" + "-" * 72)
        print(f"[步骤 8] epsilon = {ev}")
        print(f"    非零方程数（去除零行后） = {neq}")
        print(f"    系数矩阵形状             = {M.shape}  (行=方程, 列=未知数)")
        print(f"    矩阵秩 rank              = {rank}")
        print(f"    解空间维数 nullity       = {nullity}")

        print(f"\n    [新增] 当前 epsilon 下的具体表达式:")
        print(f"      lambda_eps = {lambda_ev}")
        print(f"      H = {H_ev}")

        if nullity > 0:
            ns = M.nullspace()
            print(f"\n    解空间的一组基（{len(ns)} 个，按 a0..,b0..,c0.. 排列）:")
            for k, v in enumerate(ns):
                vec = list(v)
                A_s, B_s, C_s = reconstruct(unknowns, basis, vec)
                Qt_s = A_s * px ** 2 + B_s * px * py + C_s * py ** 2
                Q_s = Qt_s / lambda_ev

                print(f"      - 基向量 {k}:")
                print(f"          A = {A_s}")
                print(f"          B = {B_s}")
                print(f"          C = {C_s}")
                print(f"          Q_tilde = {Qt_s}")
                print(f"          Q = {Q_s}")

            # eps = 0 的解回代 {Q, H}，必须精确为零
            if ev == 0:
                print(f"\n    [步骤 9] 对 eps=0 的解进行符号验证: 计算 {{Q, H}}")
                for k, v in enumerate(ns):
                    vec = list(v)
                    A_s, B_s, C_s = reconstruct(unknowns, basis, vec)
                    Qt_s = A_s * px ** 2 + B_s * px * py + C_s * py ** 2
                    Q_s = Qt_s / lambda_ev
                    pb = sp.simplify(poisson_bracket(Q_s, H_ev))
                    print(f"      基向量 {k}: {{Q, H}} = {pb}")
                    assert pb == 0, f"验证失败：基向量 {k} 的泊松括号不为零！"
                print("      -> 所有基向量验证通过：{Q, H} = 0 成立。")

    print("\n全部计算完成。")


if __name__ == "__main__":
    main()
