from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from time import perf_counter

import sympy as sp

from kerr_eq54_first_order import (
    B, CUBICS, D, J, M, P, S, a, add, block_inverse, check_cubic, clean,
    encode, eq54, metric, poisson, polynomial, pp, pt, pu,
    quadratic, r, raw_poisson, scale, u, w,
)

SUB = {M: sp.Integer(1), a: sp.Rational(1, 2)}
A2 = SUB[a] ** 2

def configure(av=sp.Rational(1, 2)):
    global A2
    SUB[a] = sp.Rational(av)
    A2 = SUB[a] ** 2
    return SUB

QUADRATICS = tuple(p for p in itertools.product(range(3), repeat=4) if sum(p) == 2)
GENERAL_J = (-3, -2, -1, 0, 1, 2)
GENERAL_K = (0, 1)

ALPHA_PROFILES = (sp.Integer(1), r ** -1, r ** -2, r ** -3)
BETA_ENTRIES = (((0, 0), "h_tt"), ((1, 1), "h_rr"), ((2, 2), "h_uu"),
                ((3, 3), "h_phiphi"), ((0, 3), "h_tphi"), ((1, 2), "h_ru"))
BETA_PROFILES = (r ** -2, r ** -3)

R_POOL = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
U_POOL = (sp.Rational(1, 6), sp.Rational(2, 6), sp.Rational(3, 6),
          sp.Rational(4, 6), sp.Rational(5, 6))
BASE_GRID = (4, 4)

def grid_points(nr, nu):
    return [(sp.Integer(rv), uv) for rv in R_POOL[:nr] for uv in U_POOL[:nu]]

def h1_of(h, gi):
    return quadratic(-(gi * h * gi).applyfunc(clean))

def background():
    gi = block_inverse(metric(r ** 2)).subs(SUB).applyfunc(clean)
    H0 = quadratic(gi)
    C0 = add(polynomial((w * pu ** 2 + (pp + a * w * pt) ** 2 / w).subs(SUB)),
             scale(H0, -2 * A2 * u ** 2))
    return gi, H0, C0

def perturbation_modes():
    out = []
    for f in ALPHA_PROFILES:
        h = eq54(-B * f / S ** 2, -a * w * B * f / S ** 2,
                 f / J, f / J).subs(SUB).applyfunc(clean)
        out.append((f"eq54[f={f}]", h, "A"))
    for (i, j), name in BETA_ENTRIES:
        for rho in BETA_PROFILES:
            h = sp.zeros(4)
            h[i, j] = h[j, i] = rho
            out.append((f"{name}={rho}", h.applyfunc(clean), "B"))
    return out

def q1_generators(gi, H0, C0, modes):
    gens, names = [], []
    for j in GENERAL_J:
        for k in GENERAL_K:
            rho = r ** j * u ** (2 * k)
            for power in QUADRATICS:
                mon = sp.prod(p ** n for p, n in zip(P, power))
                gens.append(polynomial(rho * mon))
                names.append(f"r^{j}*u^{2*k}*p^{power}")
    for label, h, _ in modes:
        gens.append(scale(h1_of(h, gi), A2 * u ** 2))
        names.append(f"a^2*u^2*H1[{label}]")
    gens.extend([H0, C0])
    names.extend(["H0", "Q0"])
    return gens, names

def evaluation_matrix(cols, points):
    rows, labels = [], []
    for rv, uv in points:
        values = [[sp.cancel(sp.sympify(col.get(k, 0)).subs({r: rv, u: uv}))
                   for col in cols] for k in CUBICS]
        for k, row in zip(CUBICS, values):
            if any(row):
                rows.append(row)
                labels.append([str(rv), str(uv), list(k)])
    T = sp.polys.matrices.DomainMatrix.from_Matrix(sp.Matrix(rows)).convert_to(sp.QQ)
    return T, T.nullspace().to_Matrix().T, labels

def kernel_vector_polynomials(ker, gens, h1s, H0, C0, nQ):
    out = []
    for j in range(ker.cols):
        v = ker[:, j]
        nz = next(x for x in v if x != 0)
        v = v / nz
        Q = add(*(scale(q, b) for q, b in zip(gens, v[:nQ]) if b != 0))
        H = add(*(scale(h1, b) for h1, b in zip(h1s, v[nQ:]) if b != 0))
        out.append((v, Q, H))
    return out

def verify_kernel(candidates, H0, C0):
    good, bad = [], []
    for j, (v, Q, H) in enumerate(candidates):
        try:
            check_cubic(add(poisson(Q, H0), poisson(C0, H)),
                        f"joint kernel vector {j + 1}")
            good.append((v, Q))
        except AssertionError as exc:
            bad.append((j, str(exc)))
    return good, bad

def run(output_dir: Path, max_grid=(10, 6)):
    start = perf_counter()
    print(f"SymPy {sp.__version__}; exact rational arithmetic; M=1, a=1/2", flush=True)

    gi, H0, C0 = background()
    check_cubic(poisson(C0, H0), "background {Q0,H0}")
    modes = perturbation_modes()
    gens, names = q1_generators(gi, H0, C0, modes)
    nQ, nH = len(gens), len(modes)
    n_alpha = sum(1 for _, _, kind in modes if kind == "A")
    print(f"  Q1 generators: {nQ} = {nQ - nH - 2} general + {nH} augmented + H0,Q0",
          flush=True)
    print(f"  perturbation modes: {nH} = {n_alpha} pairing-preserved + "
          f"{nH - n_alpha} pairing-broken", flush=True)

    print("  Building exact bracket columns", flush=True)
    cols = [raw_poisson(q, H0) for q in gens]
    h1s = [h1_of(h, gi) for _, h, _ in modes]
    cols += [raw_poisson(C0, h1) for h1 in h1s]
    cols = [{k: sp.cancel(v) for k, v in c.items()} for c in cols]
    print(f"  {len(cols)} bracket columns built ({perf_counter()-start:.1f}s)", flush=True)

    attempts = [BASE_GRID]
    nr, nu = BASE_GRID
    while (nr, nu) != max_grid:
        nr = min(nr + 2, max_grid[0])
        nu = min(nu + 1, max_grid[1])
        attempts.append((nr, nu))
        if nr == max_grid[0] and nu == max_grid[1]:
            break
    print(f"  grid schedule: {attempts}", flush=True)

    history, verified, ker, labels, T = [], None, None, None, None
    for nr, nu in attempts:
        points = grid_points(nr, nu)
        T, ker, labels = evaluation_matrix(cols, points)
        print(f"\n  grid r in {list(R_POOL[:nr])} x u({nu}) = {len(points)} points",
              flush=True)
        print(f"    matrix {T.shape}; kernel dimension {ker.cols}", flush=True)
        history.append({"r_values": list(R_POOL[:nr]),
                        "u_count": nu, "points": len(points),
                        "matrix_shape": list(T.shape),
                        "kernel_dimension": ker.cols})
        candidates = kernel_vector_polynomials(ker, gens, h1s, H0, C0, nQ)
        good, bad = verify_kernel(candidates, H0, C0)
        if not bad:
            verified = good
            print(f"    all {ker.cols} sampled kernel vectors pass full-domain "
                  f"verification => ker(eval L) = ker(L) on the declared span",
                  flush=True)
            break
        print(f"    {len(bad)} sampled vector(s) FAIL full-domain verification "
              f"=> interpolation artifact, grid too coarse", flush=True)
        for j, msg in bad[:2]:
            short = msg.split("nonzero residual = ")[-1]
            print(f"      vector {j + 1} residual: {short[:150]}", flush=True)
        history[-1]["artifact_vectors"] = [j + 1 for j, _ in bad]
    if verified is None:
        raise AssertionError(
            "grid refinement did not certify the kernel; enlarge --max-grid")

    projection = ker[nQ:, :]
    allowed_cols = projection.columnspace()
    U = sp.Matrix.hstack(*allowed_cols) if allowed_cols else sp.zeros(nH, 0)
    constraints = U.T.nullspace()
    K = (sp.Matrix.vstack(*(v.T for v in constraints)) if constraints
         else sp.zeros(0, nH))
    K = K.rref()[0]
    dim_allowed = U.cols
    print(f"\n  allowed perturbation subspace: dimension {dim_allowed} of {nH}",
          flush=True)
    print(f"  constraint matrix: {K.rows} x {K.cols} (K beta = 0)", flush=True)

    beta_zero = [i for i in range(ker.cols)
                 if all(ker[nQ + q, i] == 0 for q in range(nH))]
    print(f"  homogeneous kernel dimension (beta=0): {len(beta_zero)}", flush=True)
    trivial = [H0, C0, polynomial(pt ** 2), polynomial(pt * pp), polynomial(pp ** 2)]
    for i, q in enumerate(trivial):
        check_cubic(poisson(q, H0), f"known homogeneous invariant {i + 1}")
    if len(beta_zero) != len(trivial):
        raise AssertionError("homogeneous kernel dimension mismatch: inspect")

    alpha_idx = [i for i, (_, _, kind) in enumerate(modes) if kind == "A"]
    verdict = {}
    for idx in range(nH):
        e = sp.zeros(nH, 1)
        e[idx] = 1
        verdict[idx] = (K * e) == sp.zeros(K.rows, 1)

    print("\n  Per-direction verdict (single unit perturbation):", flush=True)
    for idx, (label, _, kind) in enumerate(modes):
        print(f"    [{kind}] {label:34s} "
              f"{'ALLOWED' if verdict[idx] else 'EXCLUDED'}", flush=True)

    alpha_part = U[:n_alpha, :]
    beta_part = U[n_alpha:, :]
    dim_alpha_img = alpha_part.rank() if n_alpha else 0
    dim_beta_img = beta_part.rank() if nH - n_alpha else 0
    a_inside = all(verdict[i] for i in alpha_idx)
    print(f"\n  allowed A-coordinate image dimension: {dim_alpha_img} of {n_alpha}",
          flush=True)
    print(f"  allowed B-coordinate image dimension: {dim_beta_img} of {nH - n_alpha}",
          flush=True)

    print("\n  Constraint rows on the 16 perturbation parameters:", flush=True)
    for row in K.tolist():
        print("   ", [str(x) for x in row], flush=True)

    print("\n  Allowed directions (coefficients on the perturbation basis):", flush=True)
    allowed_report = []
    for j in range(U.cols):
        col = U[:, j]
        support = [(modes[i][0], str(col[i])) for i in range(nH) if col[i] != 0]
        print(f"    dir {j + 1}: " + ", ".join(f"{n}*{c}" for n, c in support),
              flush=True)
        allowed_report.append({"support": support})

    report = {
        "scope": (f"M=1, a={SUB[a]}; first-order continuation {{Q1,H0}}+{{Q0,H1}}=0 "
                  f"within the declared {nH}-dim perturbation basis and "
                  f"{nQ}-dim Q1 basis"),
        "background_spin": str(SUB[a]),
        "method": ("exact rational evaluation (upper bound) + full-domain symbolic "
                   "verification of every sampled kernel vector (completeness) + "
                   "adaptive grid refinement until self-certified"),
        "background": "Kerr, g0 from metric(F=r^2); Q0 = true Carter constant",
        "perturbation_basis": [{"label": lab, "kind": kind,
                                "entries": {f"{i},{j}": str(h[i, j])
                                            for i in range(4) for j in range(i, 4)
                                            if h[i, j] != 0}}
                               for lab, h, kind in modes],
        "q1_generator_count": nQ,
        "q1_generator_names": names,
        "grid_history": history,
        "joint_matrix_shape": list(T.shape),
        "joint_kernel_dimension": ker.cols,
        "homogeneous_kernel_dimension": len(beta_zero),
        "allowed_perturbation_dimension": dim_allowed,
        "allowed_A_image_dimension": dim_alpha_img,
        "allowed_B_image_dimension": dim_beta_img,
        "whole_A_block_allowed": a_inside,
        "constraint_matrix": [[str(x) for x in row] for row in K.tolist()],
        "verdict": {modes[i][0]: ("ALLOWED" if verdict[i] else "EXCLUDED")
                    for i in range(nH)},
        "allowed_directions": allowed_report,
        "verified_kernel_vectors": [
            {"perturbation_coefficients": [str(x) for x in v[nQ:]],
             "Q1": encode(Q)} for v, Q in verified],
        "H0": encode(H0), "Q0": encode(C0),
        "seconds": round(perf_counter() - start, 3),
        "limits": ["Finite declared bases only; no general nonexistence theorem",
                   "No gauge fixing: a pure-gauge direction inside the span would "
                   "be reported ALLOWED without being a new conserved quantity",
                   "First order only",
                   "Fixed M=1, a=1/2; no generic (M,a) classification"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = str(SUB[a]).replace("/", "_")
    (output_dir / f"first_order_classification_a{tag}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n  Summary", flush=True)
    print(f"    certified grid: r in {list(R_POOL[:len(history[-1]['r_values'])])}"
          f" x u({history[-1]['u_count']})", flush=True)
    print(f"    kernel dim {ker.cols} = {len(beta_zero)} homogeneous "
          f"+ {dim_allowed} perturbation directions", flush=True)
    print(f"    pairing-preserved directions all allowed: {a_inside}", flush=True)
    print(f"    pairing-broken coordinate image: {dim_beta_img}", flush=True)
    if dim_beta_img == 0:
        print("    => within this span, no pairing-breaking direction continues",
              flush=True)
    else:
        print("    => some pairing-breaking directions do continue; see support",
              flush=True)
    print(f"  Classification completed in {perf_counter()-start:.2f}s", flush=True)
    return report

def main():
    parser = argparse.ArgumentParser(
        description="Kerr first-order classification: enlarged Q1 basis + "
                    "pairing-breaking metric perturbations")
    parser.add_argument("--output-dir", type=Path, default=Path("kerr_results"))
    parser.add_argument("--max-grid", type=str, default="10x6",
                        help="largest refinement grid, e.g. 10x6")
    parser.add_argument("--a", type=str, default="1/2",
                        help="rational Kerr spin with |a|<1, e.g. 1/2, 1/3, 3/4")
    args = parser.parse_args()
    configure(args.a)
    nr, nu = (int(x) for x in args.max_grid.lower().split("x"))
    run(args.output_dir, (nr, nu))
    print("SUCCESS", flush=True)

if __name__ == "__main__":
    main()
