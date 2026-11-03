#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from time import perf_counter

import sympy as sp

t, r, u, phi = sp.symbols("t r u phi", real=True)
M = sp.Symbol("M", positive=True)
a = sp.Symbol("a", real=True)
pt, pr, pu, pp = sp.symbols("p_t p_r p_u p_phi", real=True)
X = (t, r, u, phi)
P = (pt, pr, pu, pp)
D = r**2 - 2*M*r + a**2
w = 1-u**2
S = r**2 + a**2*u**2
B = S-2*M*r
J = (r-M)**2 - (M**2-a**2)*u**2
CASES = {
    "inverse_r": M**3/r,
    "inverse_r2": M**4/r**2,
    "inverse_r3": M**5/r**3,
}

def clean(expr):
    expr=sp.sympify(expr)

    from sympy.core.function import AppliedUndef
    jets=sorted(expr.atoms(AppliedUndef,sp.Derivative),key=sp.srepr)
    if jets:
        replacements={jet:sp.Symbol(f"_jet_{j}") for j,jet in enumerate(jets)}
        inverse={v:k for k,v in replacements.items()}
        return sp.factor(sp.cancel(expr.xreplace(replacements))).xreplace(inverse)
    return sp.factor(sp.cancel(expr))

def require_zero(expr, label):
    residual = clean(expr)
    if residual != 0:
        raise AssertionError(f"{label}: nonzero residual = {residual}")

def metric(F):
    sf, af = F+a**2*u**2, F+a**2
    g = sp.zeros(4)
    g[0, 0] = -B/sf
    g[0, 3] = g[3, 0] = -a*w*(af-D)/sf
    g[3, 3] = w*(af**2-a**2*D*w)/sf
    g[1, 1], g[2, 2] = sf/D, sf/w
    return g

def block_inverse(g):
    det = clean(g[0, 0]*g[3, 3]-g[0, 3]**2)
    gi = sp.zeros(4)
    gi[0, 0], gi[3, 3] = clean(g[3, 3]/det), clean(g[0, 0]/det)
    gi[0, 3] = gi[3, 0] = clean(-g[0, 3]/det)
    gi[1, 1], gi[2, 2] = clean(1/g[1, 1]), clean(1/g[2, 2])
    return gi

def eq54(delta_v, delta_q, delta_lambda, delta_gamma):
    h = sp.zeros(4)
    h[0, 0] = -delta_v
    h[0, 3] = h[3, 0] = delta_q
    h[1, 1] = delta_lambda*u**2+delta_gamma*(r-M)**2*w/D
    h[1, 2] = h[2, 1] = -(r-M)*u*(delta_gamma-delta_lambda)
    h[2, 2] = (delta_lambda*w*(r-M)**2+delta_gamma*u**2*D)/w
    h[3, 3] = w/B*(4*a*M*r*delta_q
                          -((r**2+a**2)**2-a**2*w*D)*delta_v)
    return h.applyfunc(clean)

def quadratic(A):
    out = {}
    for i in range(4):
        for j in range(i, 4):
            value = clean(A[i, j]/2 if i == j else A[i, j])
            if value != 0:
                powers = [0]*4
                powers[i] += 1
                powers[j] += 1
                out[tuple(powers)] = value
    return out

def polynomial(expr):
    return {powers: clean(c) for powers, c in sp.Poly(expr, *P).terms() if c != 0}

def scale(poly, scalar):
    return {power: clean(scalar*c) for power, c in poly.items() if c != 0}

def add(*polys):
    keys = set().union(*(p.keys() for p in polys))
    result = {k: clean(sum(p.get(k, 0) for p in polys)) for k in keys}
    return {k: v for k, v in result.items() if v != 0}

def expression(poly):
    return sp.Add(*(c*sp.prod(p**n for p, n in zip(P, powers))
                    for powers, c in sorted(poly.items())))

def poisson(f, g):
    out = {}
    for i in (1, 2):
        df = {power: sp.diff(c, X[i]) for power, c in f.items()}
        dg = {power: sp.diff(c, X[i]) for power, c in g.items()}
        for pf, cf in f.items():
            for pg, cg in g.items():
                term = pg[i]*df[pf]*cg-pf[i]*cf*dg[pg]
                if term == 0:
                    continue
                powers = [pf[j]+pg[j] for j in range(4)]
                powers[i] -= 1
                key = tuple(powers)
                out[key] = out.get(key, 0)+term
    return {key: clean(value) for key, value in out.items()}

CUBICS = tuple(p for p in itertools.product(range(4), repeat=4) if sum(p) == 3)

def check_cubic(poly, label):
    if set(poly)-set(CUBICS):
        raise AssertionError(f"{label}: unexpected momentum degree")
    for power in CUBICS:
        require_zero(poly.get(power, 0), f"{label}, power={power}")
    print(f"  PASS {label}: 20/20 cubic coefficients vanish", flush=True)

def recover_coefficient(source, direction):
    candidates = []
    for power in CUBICS:
        s, d = clean(source.get(power, 0)), clean(direction.get(power, 0))
        if d == 0:
            require_zero(s, f"coefficient recovery {power}")
        else:
            candidates.append(clean(-s/d))
    if not candidates:
        raise AssertionError("Restricted recovery is degenerate; no coefficient fixed")
    c = candidates[0]
    if c.free_symbols or c.has(sp.Function):
        raise AssertionError(f"Recovery needs a constant, got {c}")
    for other in candidates[1:]:
        require_zero(other-c, "inconsistent recovered coefficients")
    return c

def encode(poly):
    return {",".join(map(str, power)): str(c) for power, c in sorted(poly.items())}

def run_case(name, f, g0i, H0, C0, show=False):
    start = perf_counter()
    print(f"\n[{name}] f(r) = {f}", flush=True)
    inputs = (-B*f/S**2, -a*w*B*f/S**2, f/J, f/J)
    h = eq54(*inputs)

    z = sp.Symbol("F", real=True)
    href = metric(z).diff(z).subs(z, r**2)*f
    for i in range(4):
        for j in range(i, 4):
            require_zero(h[i, j]-href[i, j], f"Eq54 vs metric derivative ({i},{j})")
    print("  PASS Eq.(54) agrees with derivative of exact metric family", flush=True)

    inv1 = -(g0i*h*g0i)
    inv1 = inv1.applyfunc(clean)
    H1 = quadratic(inv1)

    H1_expected = add(scale(H0, -f/S),
                     polynomial(-f*pt*((r**2+a**2)*pt+a*pp)/(S*D)))
    for power in set(H1) | set(H1_expected):
        require_zero(H1.get(power, 0)-H1_expected.get(power, 0),
                     f"inverse-metric H1 cross-check {power}")
    print("  PASS H1 = -(1/2) p^T g0^(-1) h g0^(-1) p", flush=True)

    source = poisson(C0, H1)
    qbasis = scale(H1, a**2*u**2)
    direction = poisson(qbasis, H0)
    recovered = recover_coefficient(source, direction)
    Q1 = scale(qbasis, recovered)
    print(f"  RECOVERED c = {recovered} in Q1=c*a^2*u^2*H1", flush=True)

    residual = add(poisson(Q1, H0), poisson(C0, H1))
    check_cubic(residual, "{Q1,H0}+{Q0,H1}")

    witness = next(((p, clean(v)) for p, v in source.items() if clean(v) != 0), None)
    if witness is None:
        raise AssertionError("Negative control failed: uncorrected Q0 already conserved")
    print("  PASS negative control: Q1=0 fails for generic nonzero a", flush=True)

    for value in Q1.values():
        require_zero(value.subs(a, 0), "a=0 limit of Q1")
    print("  PASS a=0 limit: Q1=0", flush=True)

    if show:
        print("  h (coordinates t,r,u,phi):")
        for i in range(4):
            for j in range(i, 4):
                if h[i, j] != 0:
                    print(f"    h[{i},{j}] = {h[i,j]}")
        print("  H1 =", expression(H1))
        print("  Q1 =", expression(Q1))

    elapsed = perf_counter()-start
    print(f"  Completed in {elapsed:.2f}s", flush=True)
    return {
        "case": name, "f": str(f), "recovered_c": str(recovered),
        "h": {f"{i},{j}": str(h[i,j]) for i in range(4) for j in range(i,4)
              if h[i,j] != 0},
        "H1": encode(H1), "Q1": encode(Q1),
        "first_order_residual": {",".join(map(str,p)): "0" for p in CUBICS},
        "negative_control_nonzero_power": list(witness[0]),
        "negative_control_nonzero_coefficient": str(witness[1]),
        "seconds": round(elapsed, 3),
    }

def generic_function_test():
    print("\n[arbitrary f(r)] symbolic M,a, f and f'", flush=True)
    gi = block_inverse(metric(r**2))
    H0 = quadratic(gi)
    C0 = add(polynomial(w*pu**2+(pp+a*w*pt)**2/w), scale(H0,-2*a**2*u**2))
    result = run_case("arbitrary_function", sp.Function("f")(r), gi, H0, C0)

    F = sp.Function("F")(r)
    sf, af = F+a**2*u**2, F+a**2
    HF = polynomial((D*pr**2-(af*pt+a*pp)**2/D
                     +w*pu**2+(pp+a*w*pt)**2/w)/(2*sf))
    CF = add(polynomial(w*pu**2+(pp+a*w*pt)**2/w),scale(HF,-2*a**2*u**2))
    check_cubic(poisson(CF,HF), "exact structured family, arbitrary F(r)")
    result["exact_family_verified"] = True
    return result

def raw_poisson(f, g):
    out = {}
    for i in (1,2):
        for pf,cf in f.items():
            for pg,cg in g.items():
                term=pg[i]*sp.diff(cf,X[i])*cg-pf[i]*cf*sp.diff(cg,X[i])
                if term != 0:
                    power=list(map(sum,zip(pf,pg))); power[i]-=1
                    key=tuple(power)
                    out[key]=out.get(key,0)+term
    return out

def classification(output_dir, grid_size=4):
    start=perf_counter()
    print("\n[classification] fixed exact M=1, a=1/2", flush=True)
    sub={M:sp.Integer(1),a:sp.Rational(1,2)}
    gi=block_inverse(metric(r**2)).subs(sub).applyfunc(clean)
    g=metric(r**2).subs(sub).applyfunc(clean)
    H0=quadratic(gi)
    C0=add(polynomial((w*pu**2+(pp+a*w*pt)**2/w).subs(sub)),
           scale(H0,-2*sub[a]**2*u**2))
    hfull=eq54(-B*M**3/r/S**2,-a*w*B*M**3/r/S**2,
               M**3/r/J,M**3/r/J).subs(sub).applyfunc(clean)
    hs=[]
    for entries in [[(1,1)],[(2,2)],[(0,0),(0,3),(3,0),(3,3)]]:
        h=sp.zeros(4)
        for i,j in entries: h[i,j]=hfull[i,j]
        hs.append(h)
    Hs=[quadratic(-(gi*h*gi).applyfunc(clean)) for h in hs]
    full=add(*Hs)
    positive=scale(full,-2*sub[a]**2*u**2)

    profiles=[sp.Integer(1)]
    for value in positive.values():
        if not any(clean(value-v)==0 for v in profiles): profiles.append(value)
    quadratics=tuple(p for p in itertools.product(range(3),repeat=4) if sum(p)==2)
    candidates=[{power:v} for v in profiles for power in quadratics]+[H0,C0]
    names=[f"profile_{j} * p^{power}" for j in range(len(profiles))
           for power in quadratics]+["H0","Q0"]
    n=len(candidates)
    print(f"  Q1 basis: {len(profiles)} profiles x 10 monomials + H0,Q0 = {n} generators",flush=True)
    print("  Building exact bracket columns",flush=True)
    cols=[raw_poisson(q,H0) for q in candidates]+[raw_poisson(C0,h) for h in Hs]

    cols=[{k:sp.cancel(v) for k,v in col.items()} for col in cols]
    print("  Bracket columns built; constructing rational evaluation matrix",flush=True)

    points=[(sp.Rational(3+i),sp.Rational(j+1,grid_size+2))
            for i in range(grid_size) for j in range(grid_size)]
    rows=[]; labels=[]
    for ir,(rv,uv) in enumerate(points):
        values=[[sp.cancel(col.get(k,0).subs({r:rv,u:uv}))
                 if hasattr(col.get(k,0),'subs') else sp.Integer(0)
                 for col in cols] for k in CUBICS]
        for k,row in zip(CUBICS,values):
            if any(row): rows.append(row);labels.append([str(rv),str(uv),list(k)])
        print(f"  Exact points: {ir+1}/{len(points)}",flush=True)
    T=sp.polys.matrices.DomainMatrix.from_Matrix(sp.Matrix(rows)).convert_to(sp.QQ)

    ker=T.nullspace().to_Matrix().T
    print(f"  Joint matrix {T.shape}; kernel dimension {ker.cols}",flush=True)
    verified=[]
    for j in range(ker.cols):
        v=ker[:,j]
        nz=next(x for x in v if x!=0);v=v/nz
        Q=add(*(scale(q,b) for q,b in zip(candidates,v[:n]) if b!=0))
        H=add(*(scale(h,c) for h,c in zip(Hs,v[n:]) if c!=0))
        check_cubic(add(poisson(Q,H0),poisson(C0,H)),f"full-domain joint kernel vector {j+1}")
        verified.append((v,Q))

    projection=ker[n:,:]
    allowed=projection.columnspace()
    U=sp.Matrix.hstack(*allowed) if allowed else sp.zeros(3,0)
    constraints=U.T.nullspace()
    constraints_matrix=sp.Matrix.vstack(*(v.T for v in constraints)) if constraints else sp.zeros(0,3)
    constraints_matrix=constraints_matrix.rref()[0]
    print("  COMPLETE finite-basis parameter constraints:",constraints_matrix.tolist(),flush=True)

    bg_kernel_dim=ker.cols-U.cols
    trivial=[H0,C0,polynomial(pt**2),polynomial(pt*pp),polynomial(pp**2)]
    for i,q in enumerate(trivial): check_cubic(poisson(q,H0),f"known homogeneous invariant {i+1}")

    inv_eval=sp.Matrix([[q.get(k,sp.Integer(0)).subs({r:3,u:sp.Rational(1,3)})
                         for q in trivial] for k in quadratics])
    if inv_eval.rank()!=5:
        raise AssertionError("Known invariant independence check failed")
    if bg_kernel_dim!=5:
        raise AssertionError("Additional homogeneous modes or redundant basis: inspect before summarizing")
    print("  Homogeneous space: exactly H0,Q0,p_t^2,p_t*p_phi,p_phi^2",flush=True)

    output_dir.mkdir(parents=True,exist_ok=True)
    matrix=sp.Matrix(rows)
    report={
      "scope":"M=1,a=1/2; stationary axisymmetric quadratic Q1 in the explicit finite span only",
      "method":"exact rational evaluation upper bound + full symbolic verification of every kernel generator",
      "f":"1/r", "perturbation_parameters":["c_rr","c_uu","c_tphi_block"],
      "h_basis":[{f"{i},{j}":str(h[i,j]) for i in range(4) for j in range(i,4) if h[i,j]!=0} for h in hs],
      "profiles":[str(v) for v in profiles],"candidate_names":names,
      "candidate_basis":[encode(q) for q in candidates],
      "basis_generator_count":n,"joint_matrix_shape":list(matrix.shape),
      "joint_kernel_dimension":ker.cols,"homogeneous_coefficient_kernel_dimension":bg_kernel_dim,
      "allowed_parameter_dimension":U.cols,
      "constraints_on_c":[[str(x) for x in row] for row in constraints_matrix.tolist()],
      "allowed_parameter_basis":[[str(x) for x in row] for row in U.tolist()],
      "verified_joint_kernel":[{"coefficients":[str(x) for x in v],"Q1":encode(q)} for v,q in verified],
      "evaluation_rows":labels,"evaluation_matrix":[[str(x) for x in row] for row in rows],
      "H0":encode(H0),"Q0":encode(C0),"H1_basis":[encode(h) for h in Hs],
      "positive_Q1":encode(positive),"seconds":round(perf_counter()-start,3),
      "limits":["No unrestricted smooth-function nonexistence theorem",
                "No generic M,a classification inferred from fixed rational background",
                "No second-order calculation"]}
    (output_dir/"classification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")

    for j in range(3):
        e=sp.eye(3)[:,j]
        print(f"  Single-sector perturbation {j}: "+("EXCLUDED in declared span" if constraints_matrix*e!=sp.zeros(constraints_matrix.rows,1) else "ALLOWED"),flush=True)
    print(f"  Classification completed in {perf_counter()-start:.2f}s",flush=True)
    return report

def extension_main():
    parser=argparse.ArgumentParser(description="Kerr Eq54: arbitrary-function verification and finite-basis classification")
    parser.add_argument("--mode",choices=["all","generic","classify","regression"],default="all")
    parser.add_argument("--output-dir",type=Path,default=Path("kerr_results"))
    parser.add_argument("--grid-size",type=int,default=4,
                        help="exact grid size; insufficient grid fails symbolic kernel verification")
    parser.add_argument("--case",choices=["all",*CASES],help="legacy positive-example selector (implies regression)")
    parser.add_argument("--show",action="store_true",help="show legacy h,H1,Q1 (implies regression)")
    parser.add_argument("--json",type=Path,help="legacy regression JSON output")
    args=parser.parse_args()
    if args.case or args.show or args.json: args.mode="regression"
    print(f"SymPy {sp.__version__}; exact arithmetic throughout",flush=True)
    if args.grid_size<2: parser.error("--grid-size must be >=2")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.mode in ("all","generic"):
        result=generic_function_test()
        (args.output_dir/"arbitrary_function.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    if args.mode in ("all","classify"): classification(args.output_dir,args.grid_size)
    if args.mode=="regression":
        gi=block_inverse(metric(r**2));H0=quadratic(gi)
        C0=add(polynomial(w*pu**2+(pp+a*w*pt)**2/w),scale(H0,-2*a**2*u**2))
        check_cubic(poisson(C0,H0),"background")
        chosen=CASES if args.case in (None,"all") else {args.case:CASES[args.case]}
        results=[run_case(name,f,gi,H0,C0,args.show) for name,f in chosen.items()]
        if args.json:
            args.json.parent.mkdir(parents=True,exist_ok=True)
            args.json.write_text(json.dumps({"H0":encode(H0),"Q0":encode(C0),"results":results},ensure_ascii=False,indent=2),encoding="utf-8")
    print("SUCCESS",flush=True)

if __name__ == "__main__":
    extension_main()
