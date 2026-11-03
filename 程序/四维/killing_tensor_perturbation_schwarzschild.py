import sympy as sp
from sympy import symbols, Rational, Matrix, Poly, expand, together, fraction, diff

def clear_denominator(expr, r, u):
    expr = sp.together(sp.expand(expr))
    num, den = sp.fraction(expr)
    return sp.expand(num)

def extract_rows(expr, momenta, unknowns, r, u, tag):
    P = sp.Poly(sp.expand(expr), *momenta)
    rows = []
    for monom, coef in zip(P.monoms(), P.coeffs()):
        if sum(monom) != 3:
            continue
        num = clear_denominator(coef, r, u)
        if num == 0:
            continue
        Qp = sp.Poly(num, r, u)
        for (rm, um), cval in zip(Qp.monoms(), Qp.coeffs()):
            cval = sp.expand(cval)
            if cval == 0:
                continue
            cp = sp.Poly(cval, *unknowns)
            mvec = [cp.coeff_monomial(q) for q in unknowns]
            rest = sp.expand(cval - sum(mvec[i] * unknowns[i]
                                        for i in range(len(unknowns))))
            rows.append((tag, monom, (rm, um), mvec, rest))
    return rows

def normalize_condition(coeffs):
    c0 = coeffs[0][1]
    return tuple((p_, sp.nsimplify(sp.cancel(c_ / c0))) for p_, c_ in coeffs)

def main():

    r = symbols("r", real=True)
    u = symbols("u", real=True)
    w = symbols("w", real=True)
    pt, pr, pth, pph = symbols("pt pr ptheta pphi")
    momenta = [pt, pr, pth, pph]

    A = symbols("A1:4")
    B = symbols("B1:4")
    C = symbols("C1:4")
    params = list(A) + list(B) + list(C)

    f = 1 - 2 / r
    P2 = (3 * u ** 2 - 1) / 2

    a_r = sum(A[j - 1] * r ** (-j) for j in range(1, 4))
    b_r = sum(B[j - 1] * r ** (-j) for j in range(1, 4))
    c_r = sum(C[j - 1] * r ** (-j) for j in range(1, 4))

    g0 = sp.diag(-f, 1 / f, r ** 2, r ** 2 * (1 - u ** 2))
    g0_inv = g0.inv()

    g1 = sp.diag(-f * a_r * P2,
                 (1 / f) * b_r * P2,
                 r ** 2 * c_r * P2,
                 r ** 2 * (1 - u ** 2) * c_r * P2)

    g1_up = -g0_inv * g1 * g0_inv

    H0 = Rational(1, 2) * sum(g0_inv[i, j] * momenta[i] * momenta[j]
                              for i in range(4) for j in range(4))
    H1 = Rational(1, 2) * sum(g1_up[i, j] * momenta[i] * momenta[j]
                              for i in range(4) for j in range(4))

    Q0 = pth ** 2 + pph ** 2 / (1 - u ** 2)

    q = symbols("q0:9")
    Q1 = P2 * sum(
        r ** (-k) * (q[3 * (k - 1) + 0] * pr ** 2
                     + q[3 * (k - 1) + 1] * pth ** 2
                     + q[3 * (k - 1) + 2] * pph ** 2)
        for k in (1, 2, 3))

    def pr_(e):
        return sp.diff(e, r)

    def pth_(e):
        return -w * sp.diff(e, u)

    def pb(f1, g1_):
        return (pr_(f1) * sp.diff(g1_, pr) - sp.diff(f1, pr) * pr_(g1_)
                + pth_(f1) * sp.diff(g1_, pth) - sp.diff(f1, pth) * pth_(g1_))

    E = expand(pb(Q1, H0) + pb(Q0, H1))

    deg_w = sp.Poly(E, w).degree() if E != 0 else 0
    assert deg_w <= 1, f"E 对 w 的次数 = {deg_w}，应为 <= 1"

    P = expand(E.subs(w, 0))
    Q = expand(sp.diff(E, w))

    rows = extract_rows(P, momenta, q, r, u, "even")
    rows += extract_rows(Q, momenta, q, r, u, "odd")

    print("=" * 76)
    print("Schwarzschild 一阶微扰：二次守恒量 Q0 + ε Q1 的延续性搜索")
    print("=" * 76)
    print(f"  Q1 未知系数      : {list(q)}")
    print(f"  扰动参数          : {list(params)}")
    print(f"  提取出的方程数    : {len(rows)}")

    M = Matrix([row[3] for row in rows])
    bvec = Matrix([-row[4] for row in rows])

    print(f"  系数矩阵 M 形状   : {M.shape}")
    rankM = M.rank()
    print(f"  rank(M)           : {rankM}")

    Y = M.T.nullspace()
    print(f"  M 的左零空间维数  : {len(Y)}")

    active_params = []
    for p_ in params:
        v = bvec.diff(p_)
        if any(e != 0 for e in v):
            active_params.append(p_)
    print(f"  b 中出现的扰动参数: {active_params}")

    obstructions = []
    for yi, y in enumerate(Y):
        expr = sp.expand((y.T * bvec)[0])
        if expr == 0:
            continue
        coeffs = [(p_, sp.expand(sp.diff(expr, p_))) for p_ in active_params]
        coeffs = [(p_, c_) for p_, c_ in coeffs if c_ != 0]
        obstructions.append((yi, coeffs))

    print()
    if not obstructions:
        print("  【结论】b 落在 col(M) 中：方程组一致，存在解。")
        sol = sp.linsolve((M, bvec))
        print("  Q1 解（含参数）:")
        print(sol)
        return

    norm_conds = []
    seen = set()
    for yi, coeffs in obstructions:
        norm = normalize_condition(coeffs)
        if norm in seen:
            continue
        seen.add(norm)
        norm_conds.append((yi, norm))

    print("  【结论】方程无解：对一般扰动，M q = b 不相容。")
    print()
    print("  证明：取 M 的左零向量 y（yᵀM = 0），则对任意 q 有")
    print("        0 = yᵀ(M q) = yᵀb.")
    print("  而 yᵀb 是扰动参数的**非零**线性组合，故参数必须满足下列约束：")
    for k, (yi, norm) in enumerate(norm_conds):
        parts = []
        for p_, c_ in norm:
            if c_ == 1:
                parts.append(f"{p_}")
            elif c_ == -1:
                parts.append(f"-{p_}")
            else:
                parts.append(f"{c_}*{p_}")
        print(f"    L{k}: " + " + ".join(parts).replace("+ -", "- ") + " = 0")

    Pmat = sp.zeros(len(norm_conds), len(active_params))
    for i, (yi, norm) in enumerate(norm_conds):
        for p_, c_ in norm:
            Pmat[i, active_params.index(p_)] = c_
    pns = Pmat.nullspace()
    print()
    print(f"  这些参数约束的解空间维数 = {len(pns)}")
    if len(pns) == 0:
        print("  ⟹ 唯一满足约束的是 (A1,A2,A3,B1,B2,B3,C1,C2,C3) = 0，")
        print("     即平凡扰动。任何非平凡 l=2 静态轴对称扰动都不能延续 Q0。")
    else:
        print("  ⟹ 存在非平凡参数子空间可能延续 Q0：")
        for v in pns:
            print("     ", v.T)

    print()
    print("  数值校验（代入具体扰动，看 rank[M|b] 是否超过 rank(M)）:")
    tests = [
        ("A1=1  (仅 g_tt 扰动)", {A[0]: 1}),
        ("B1=1  (仅 g_rr 扰动)", {B[0]: 1}),
        ("B3=1  (g_rr 的 r^-3 项)", {B[2]: 1}),
        ("C1=1  (仅 g_θθ,g_φφ 扰动)", {C[0]: 1}),
    ]
    for label, subs in tests:
        allsubs = {p_: 0 for p_ in params}
        allsubs.update(subs)
        binst = bvec.subs(allsubs)
        raug = M.row_join(binst).rank()
        verdict = "无解" if raug > rankM else "有解"
        print(f"    {label:24s}: rank[M|b] = {raug:2d} (rank M = {rankM}) -> {verdict}")

if __name__ == "__main__":
    main()
