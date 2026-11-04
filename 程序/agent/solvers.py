"""求解器：Killing 张量搜索、精确验证、参数扫描、一阶分类。

所有结论都带"证书"：
* 搜索 -> 精确有理数线性代数（秩、零空间），无抽样；
* 验证 -> 泊松括号逐系数精确消去；
* 一阶分类 -> 点值给出上界 + 每个核向量全域回代给出下界，
  两者相等即证书；网格不足时自动加密直到自证。
"""

from __future__ import annotations

import itertools
from time import perf_counter

import sympy as sp

import backgrounds as bgmod
from engine import (add, cancel_only, clean, coord_monomials, coordinate_rows, format_poly,
                    tidy, is_zero_expr,
                    from_expr, hamiltonian, is_zero, momentum_monomials,
                    poisson, quadratic_from_matrix, scale, to_expr)


class BudgetExceeded(RuntimeError):
    pass


VERBOSE = False


def _log(msg):
    if VERBOSE:
        print(msg, flush=True)


# ------------------------------------------------------------ 候选空间 ----
def ansatz_generators(bg, values, degree, axes=None, denominator=None,
                      momentum_degree=2, exponents=None):
    """Q = sum_j c_j * (monomial(x) / denominator) * p^power。

    exponents: {坐标名: [允许的幂次]}，给出时忽略 degree。
               允许负幂次（Laurent），用于 r^-3 ... r^2 这类剖面。
    """
    p = {**bg.params, **values}
    axes = tuple(axes) if axes else tuple(bg.ansatz_axes)
    den = denominator if denominator is not None else bg.ansatz_denominator(p)
    den = clean(den)
    quad = momentum_monomials(bg.dim, momentum_degree)
    if exponents:
        per_axis = [list(exponents.get(sp.sstr(c), range(degree + 1)))
                    for c in axes]
        mons = []
        for combo in itertools.product(*per_axis):
            mono = sp.Integer(1)
            for c, k in zip(axes, combo):
                if k:
                    mono *= c ** k
            mons.append(mono)
    else:
        mons = coord_monomials(axes, degree)
    gens, labels = [], []
    for power in quad:
        for mono in mons:
            gens.append({power: clean(mono / den)})
            labels.append(f"({sp.sstr(mono)})*p^{power}"
                          + ("" if den == 1 else f" / ({sp.sstr(den)})"))
    return gens, labels, axes


def express_in_basis(poly, gens, coords, momenta):
    """把 poly 写成 sum c_j gens[j]；不可表示时返回 None。"""
    keys = set(poly.keys()) | set().union(*(g.keys() for g in gens))
    rows, rhs, index = [], [], {}
    for key in sorted(keys):
        # 每个动量 key 下，比较坐标单项式系数
        entries = [(j, g.get(key, 0)) for j, g in enumerate(gens) if g.get(key, 0)]
        target = poly.get(key, 0)
        mons = set()
        for _, e in entries:
            for m, _c in coordinate_rows(e, coords):
                mons.add(m)
        for m, _c in coordinate_rows(target, coords):
            mons.add(m)
        for mon in sorted(mons):
            row = [sp.Integer(0)] * len(gens)
            for j, e in entries:
                for mm, cc in coordinate_rows(e, coords):
                    if mm == mon:
                        row[j] += cc
            b = 0
            for mm, cc in coordinate_rows(target, coords):
                if mm == mon:
                    b += cc
            index[len(rows)] = (key, mon)
            rows.append(row)
            rhs.append(clean(b))
    if not rows:
        return [sp.Integer(0)] * len(gens)
    A = sp.Matrix(rows)
    b = sp.Matrix(rhs)
    try:
        sol = A.gauss_jordan_solve(b)[0]
    except ValueError:
        return None
    if any(clean(x) != 0 for x in (A * sol - b)):
        return None
    return [clean(v) for v in sol]


# ------------------------------------------------------------ 搜索 ----
def killing_search(bg, values, degree=2, axes=None, denominator=None,
                   max_generators=260, verify=True):
    start = perf_counter()
    coords, momenta = bg.coords, bg.momenta
    p = {**bg.params, **values}
    gens, labels, axes = ansatz_generators(bg, values, degree, axes, denominator)
    if len(gens) > max_generators:
        raise BudgetExceeded(
            f"ansatz has {len(gens)} generators (> {max_generators}); "
            f"reduce degree or axes")
    H = bg.H(values)

    # 逐生成元算 {gen,H}。注意：不同生成元的系数分母**不一定相同**
    # （例如 lambda_eps 在 epsilon=2 时可分解，together 会给出不同分母），
    # 所以必须先对每个动量 key 取分母的最小公倍式再组装，
    # 否则等于把 sum_j c_j*N_j 当成 sum_j c_j*N_j/D_j 来解，会多出伪约束。
    per_key: dict = {}
    for j, gen in enumerate(gens):
        for key, expr in poisson(gen, H, coords, momenta).items():
            num, den = sp.fraction(sp.cancel(expr))
            per_key.setdefault(key, []).append((j, num, den))
    rows: dict = {}
    for key, items in per_key.items():
        common = items[0][2]
        for _j, _n, den in items[1:]:
            common = sp.lcm(common, den)
        for j, num, den in items:
            scaled = sp.expand(num * sp.cancel(common / den))
            for mon, coef in sp.Poly(scaled, *coords).terms():
                rows.setdefault((key, mon),
                                [sp.Integer(0)] * len(gens))[j] += coef
    if not rows:
        return {"error": "empty linear system"}
    mat = sp.Matrix(list(rows.values()))
    rank = mat.rank()
    nullity = mat.cols - rank
    kernel = mat.nullspace()

    solutions = []
    for vec in kernel:
        Q = add(*(scale(gen, c) for gen, c in zip(gens, vec) if c != 0))
        solutions.append(Q)

    known = []
    for label, poly, note in bg.known(values):
        vec = express_in_basis(poly, gens, coords, momenta)
        if vec is not None and any(v != 0 for v in vec):
            known.append({"label": label, "note": note,
                          "in_ansatz": True,
                          "coefficients": [sp.sstr(v) for v in vec]})
    known_dim = len(known)

    verified = []
    if verify:
        for Q in solutions:
            residual = poisson(Q, H, coords, momenta)
            verified.append(is_zero(residual))

    out = {
        "background": bg.key,
        "parameters": {sp.sstr(k): sp.sstr(v) for k, v in p.items()},
        "ansatz": {"degree": degree,
                   "axes": [sp.sstr(c) for c in axes],
                   "momentum_degree": 2,
                   "generators": len(gens),
                   "denominator": sp.sstr(clean(
                       denominator if denominator is not None
                       else bg.ansatz_denominator(p)))},
        "equations": mat.rows,
        "unknowns": mat.cols,
        "rank": rank,
        "nullity": nullity,
        "known_dim": known_dim,
        "gap": nullity - known_dim,
        "known_invariants": known,
        "solutions": [
            {"index": i,
             "verified": verified[i] if verify else None,
             "Q": format_poly(Q, coords, momenta, 10),
             "terms": len(Q)}
            for i, Q in enumerate(solutions)],
        "certificate": {
            "type": "exact_rational_linear_algebra",
            "method": "逐 (动量 key, 坐标单项式) 提取系数，精确有理数秩与零空间",
            "verified_all": all(verified) if verify else None,
        },
        "seconds": round(perf_counter() - start, 2),
    }
    return out, solutions, gens, labels


# ------------------------------------------------------------ 验证 ----
def verify_quantity(bg, values, q_text):
    coords, momenta = bg.coords, bg.momenta
    p = {**bg.params, **values}
    local = {sp.sstr(s): s for s in coords + momenta + tuple(p.keys())}
    expr = sp.sympify(q_text.replace("^", "**"), locals=local)
    poly = from_expr(expr, momenta)
    H = bg.H(values)
    residual = poisson(poly, H, coords, momenta)
    ok = is_zero(residual)
    first = None
    if not ok:
        key, val = next(((k, v) for k, v in sorted(residual.items())
                         if clean(v) != 0), (None, None))
        first = {"momentum_power": list(key), "coefficient": sp.sstr(val)}
    return {
        "background": bg.key,
        "parameters": {sp.sstr(k): sp.sstr(v) for k, v in p.items()},
        "Q_parsed_terms": len(poly),
        "poisson_zero": ok,
        "residual_terms": len(residual),
        "first_nonzero": first,
        "certificate": {"type": "symbolic_identity",
                        "verified": ok,
                        "method": "逐三次动量系数精确消去"},
    }


# ------------------------------------------------------------ 参数扫描 ----
def scan_parameter(bg, param_name, values_list, base_values=None, degree=2,
                   axes=None, denominator=None):
    sym = None
    for s in bg.params:
        if sp.sstr(s) == param_name:
            sym = s
    if sym is None:
        raise KeyError(f"{param_name} 不是 {bg.key} 的参数；"
                       f"可选 {[sp.sstr(s) for s in bg.params]}")
    table = []
    for raw in values_list:
        value = sp.Rational(str(raw)) if isinstance(raw, (int, str)) else raw
        vals = {**(base_values or {}), sym: value}
        row, *_ = killing_search(bg, vals, degree=degree, axes=axes,
                                 denominator=denominator, verify=False)
        table.append({"value": sp.sstr(value), "rank": row["rank"],
                      "nullity": row["nullity"], "known_dim": row["known_dim"],
                      "gap": row["gap"]})
    changes = [i for i in range(1, len(table))
               if table[i]["nullity"] != table[i - 1]["nullity"]]
    return {"background": bg.key, "parameter": param_name, "table": table,
            "transition_indices": changes,
            "note": "nullity 变化处即为分岔点，需单独分析"}


# ------------------------------------------------------- 一阶分类 ----
def first_order_classification(bg, values, q0_label, h_specs, q1_degree,
                               q1_axes=None, q1_exponents=None,
                               augment_weights=(), include_background=True,
                               nr=4, nu=4, max_nr=10, max_nu=6,
                               max_q1=320, verify=True):
    """解 {Q1,H0} + {Q0,H1} = 0，并给出"哪些扰动方向可延续"的分类。

    h_specs: [(label, Matrix)] 扰动基，每个是 4x4（或 dxd）对称矩阵。
    augment_weights: 形如 ["a^2*u^2"] 的权重；生成元 weight*H1[h_j] 会被加入 Q1 基，
      以保证已知正例（Q1 = -2 a^2 u^2 H1）落在候选空间内。
    """
    start = perf_counter()
    coords, momenta = bg.coords, bg.momenta
    p = {**bg.params, **values}
    gi = bg.g_inv(values)
    H0 = bg.H(values)

    known = {lab: poly for lab, poly, _ in bg.known(values)}
    if q0_label not in known:
        raise KeyError(f"{q0_label} 不是 {bg.key} 的已知不变量；"
                       f"可选 {sorted(known)}")
    Q0 = known[q0_label]

    # --- 扰动基 -> H1 ---
    allowed = set(coords) | set(p.keys())
    h1s, h_labels = [], []
    for label, h in h_specs:
        extra = h.free_symbols - allowed
        if extra:
            raise ValueError(
                f"扰动 {label} 含有未知符号 {sorted(sp.sstr(s) for s in extra)}；"
                f"只允许坐标 {[sp.sstr(c) for c in coords]} 与参数 "
                f"{[sp.sstr(k) for k in p]}")
        h1s.append(quadratic_from_matrix(-(gi * h * gi).applyfunc(tidy),
                                         momenta))
        h_labels.append(label)
    nH = len(h1s)

    # --- Q1 候选空间 ---
    q1_gens, q1_labels, axes = ansatz_generators(
        bg, values, q1_degree, q1_axes, denominator=sp.Integer(1),
        exponents=q1_exponents)
    local = {sp.sstr(s): s for s in coords + momenta + tuple(p.keys())}
    for w in augment_weights:
        w_expr = sp.sympify(str(w).replace("^", "**"), locals=local).subs(p)
        for label, h1 in zip(h_labels, h1s):
            q1_gens.append(scale(h1, w_expr))
            q1_labels.append(f"({w})*H1[{label}]")
    if include_background:
        q1_gens.extend([H0, Q0])
        q1_labels.extend(["H0", q0_label])
    if len(q1_gens) + nH > max_q1:
        raise BudgetExceeded(
            f"Q1 basis {len(q1_gens)} + {nH} perturbation columns "
            f"> {max_q1}; reduce q1_degree")
    nQ = len(q1_gens)

    # --- 列 ---
    cols = [poisson(q, H0, coords, momenta) for q in q1_gens]
    cols += [poisson(Q0, h1, coords, momenta) for h1 in h1s]
    cols = [{k: tidy(v) for k, v in c.items()} for c in cols]
    _log(f"    [solvers] {len(cols)} bracket columns built "
         f"({perf_counter()-start:.1f}s)")
    cubic = momentum_monomials(bg.dim, 3)

    axes = tuple(axes)
    a_axis, b_axis = axes[0], axes[1]
    a_pool = [sp.Rational(3 + i) for i in range(max_nr)]
    b_pool = [sp.Rational(k, max_nu + 1) for k in range(1, max_nu + 1)]

    # 关键优化：每列先分解成 num/den 两个多项式，点值只需 Horner 求值。
    # 直接对有理函数逐点调用 cancel 会做多项式 GCD，慢一到两个数量级。
    prepared = []
    for col in cols:
        entry = {}
        for k in cubic:
            value = col.get(k)
            if value is None or value == 0:
                continue
            num, den = sp.fraction(value)
            entry[k] = (sp.Poly(num, a_axis, b_axis),
                        sp.Poly(den, a_axis, b_axis))
        prepared.append(entry)
    _log(f"    [solvers] columns pre-factored into num/den "
         f"({perf_counter()-start:.1f}s)")

    def build(na, nb):
        rows, labels = [], []
        for av in a_pool[:na]:
            for bv in b_pool[:nb]:
                point = (av, bv)
                for k in cubic:
                    row, nonzero = [], False
                    for entry in prepared:
                        pair = entry.get(k)
                        if pair is None:
                            row.append(sp.Integer(0))
                            continue
                        num, den = pair
                        dval = den.eval(point)
                        if dval == 0:
                            raise ZeroDivisionError(
                                f"分母在 {point} 处为零，请换网格")
                        value = num.eval(point) / dval
                        row.append(value)
                        if value:
                            nonzero = True
                    if nonzero:
                        rows.append(row)
                        labels.append([sp.sstr(av), sp.sstr(bv), list(k)])
        mat = sp.Matrix(rows)
        dom = sp.polys.matrices.DomainMatrix.from_Matrix(mat).convert_to(sp.QQ)
        return mat, dom.nullspace().to_Matrix().T, labels

    history = []
    attempts = [(nr, nu)]
    while attempts[-1] != (max_nr, max_nu):
        na, nb = attempts[-1]
        na, nb = min(na + 2, max_nr), min(nb + 1, max_nu)
        attempts.append((na, nb))
        if (na, nb) == (max_nr, max_nu):
            break

    ker = mat = labels = None
    verified, bad = [], []
    for na, nb in attempts:
        mat, ker, labels = build(na, nb)
        _log(f"    [solvers] grid {na}x{nb}: matrix {mat.shape} "
             f"kernel {ker.cols} ({perf_counter()-start:.1f}s)")
        entry = {"a_values": [sp.sstr(v) for v in a_pool[:na]],
                 "b_count": nb, "points": na * nb,
                 "matrix_shape": list(mat.shape),
                 "kernel_dimension": ker.cols}
        history.append(entry)
        verified, bad = [], []
        if verify:
            for j in range(ker.cols):
                v = ker[:, j]
                nz = next((c for c in v if c != 0), None)
                v = v / nz
                Q = add(*(scale(q, c) for q, c in zip(q1_gens, v[:nQ]) if c != 0))
                H = add(*(scale(h1, c) for h1, c in zip(h1s, v[nQ:]) if c != 0))
                residual = add(poisson(Q, H0, coords, momenta),
                               poisson(Q0, H, coords, momenta))
                if is_zero(residual):
                    verified.append((v, Q))
                else:
                    key, val = next(((k, x) for k, x in sorted(residual.items())
                                     if not is_zero_expr(x)))
                    bad.append({"vector": j + 1,
                                "momentum_power": list(key),
                                "residual": sp.sstr(val)[:200]})
        else:
            verified = [(ker[:, j], None) for j in range(ker.cols)]
        if not bad:
            break
        entry["artifacts"] = [b["vector"] for b in bad]

    projection = ker[nQ:, :]
    allowed_cols = projection.columnspace()
    U = (sp.Matrix.hstack(*allowed_cols) if allowed_cols
         else sp.zeros(nH, 0))
    constraints = U.T.nullspace()
    K = (sp.Matrix.vstack(*(v.T for v in constraints)) if constraints
         else sp.zeros(0, nH)).rref()[0]

    beta_zero = [i for i in range(ker.cols)
                 if all(ker[nQ + q, i] == 0 for q in range(nH))]
    verdict = {}
    for idx in range(nH):
        e = sp.zeros(nH, 1)
        e[idx] = 1
        verdict[h_labels[idx]] = ("ALLOWED" if (K * e) == sp.zeros(K.rows, 1)
                                  else "EXCLUDED")

    return {
        "background": bg.key,
        "parameters": {sp.sstr(k): sp.sstr(v) for k, v in p.items()},
        "q0": q0_label,
        "h_basis": h_labels,
        "q1_generators": nQ,
        "grid_history": history,
        "matrix_shape": list(mat.shape),
        "kernel_dimension": ker.cols,
        "homogeneous_dimension": len(beta_zero),
        "allowed_perturbation_dimension": U.cols,
        "constraints": [[sp.sstr(c) for c in row] for row in K.tolist()],
        "verdict": verdict,
        "allowed_directions": [
            [h_labels[i] for i in range(nH) if U[i, j] != 0]
            for j in range(U.cols)],
        "verified_vectors": len(verified),
        "artifacts": bad,
        "certificate": {
            "type": "grid_convergence + symbolic_identity",
            "verified_all_kernel_vectors": not bad,
            "note": "点值给上界，全域回代给下界，两者相等即证书；"
                    "网格加密直到自证",
        },
        "seconds": round(perf_counter() - start, 2),
    }
