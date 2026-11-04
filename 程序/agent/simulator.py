"""与 Black Hole Simulator 的桥接层。

模拟器（C++/Qt）负责：给定度规参数与初始条件，积分测地线，吐出
Boyer–Lindquist 坐标与协变动量；渲染黑洞图像。

本模块负责：调用它、解析结果、并用**符号** $K^{\\mu\\nu}$ 沿轨道计算候选
$Q=K^{\\mu\\nu}p_\\mu p_\\nu$ 的漂移。

为什么这样分工
--------------
模拟器只知道它内置的 Carter 常数。候选守恒量 $Q$ 的形式由符号层决定，
所以不在 C++ 里重写一遍，而是让模拟器把 $(x^\\mu, p_\\mu)$ 吐出来，
由 Python 用同一个符号表达式求值。这样"符号证明"和"数值漂移"用的是
**同一个 $K^{\\mu\\nu}$**，两条证据才是真的独立互证。
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

import sympy as sp

from engine import to_expr

SIM_ROOT = Path(os.environ.get("BH_SIM_ROOT", r"D:\For Spacetime Simulator"))
CLI = SIM_ROOT / "dist" / "bh_cli.exe"

PRESET_IDS = ("schw_circular", "schw_precess", "schw_escape", "schw_capture",
              "kerr_inclined", "light_bend", "light_beam", "shadow")


class SimulatorError(RuntimeError):
    pass


def available() -> bool:
    return CLI.exists()


def _run(args, timeout=300):
    if not available():
        raise SimulatorError(f"找不到模拟器命令行：{CLI}")
    try:
        proc = subprocess.run([str(CLI), *args], capture_output=True,
                              text=True, timeout=timeout,
                              cwd=str(Path(tempfile.gettempdir())))
    except subprocess.TimeoutExpired as exc:
        raise SimulatorError(f"模拟器超时（{timeout}s）：{' '.join(args)}") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise SimulatorError(f"模拟器返回 {proc.returncode}：{detail}")
    return proc.stdout


def list_presets() -> list[dict]:
    """返回 [{'id': ..., 'title': ...}]。"""
    out = _run(["--list-presets"])
    presets = []
    for line in out.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0] in PRESET_IDS:
            presets.append({"id": parts[0], "title": parts[1].strip()})
    return presets


# ------------------------------------------------------------------ 场景 ----
def ks_from_bl(r: float, theta: float, phi_ks: float, a: float) -> tuple:
    """Boyer–Lindquist (r, θ, φ_KS) -> Kerr-Schild 笛卡尔。

    与模拟器 `ksFromBL` 完全一致；注意 φ_KS 已经是 Kerr-Schild 方位角
    （= ϕ_BL − arctan(a/r)），不是 BL 的 ϕ。
    """
    st, ct = math.sin(theta), math.cos(theta)
    cp, sp = math.cos(phi_ks), math.sin(phi_ks)
    return ((r * cp - a * sp) * st, (r * sp + a * cp) * st, r * ct)


def circular_speed(r: float, a: float, M: float = 1.0) -> float:
    """赤道顺行圆轨道的 ZAMO 局部速度（与模拟器一致）。"""
    r2 = r * r
    delta = r2 - 2.0 * M * r + a * a
    if delta <= 0.0:
        return 0.0
    A = (r2 + a * a) ** 2 - a * a * delta
    omega = 2.0 * M * a * r / A
    Om = 1.0 / (r ** 1.5 + a)
    return abs((Om - omega) * A / (r2 * math.sqrt(delta)))


def scene(M=1.0, a=0.0, *, r=10.0, theta_deg=90.0, phi_ks=0.0,
          speed=None, radial=0.0, duration=600.0, max_r=400.0,
          kind="particle", mass=1.0, name=None, disk=True,
          disk_in=6.0, disk_out=24.0, camera=None) -> dict:
    """构造一个单天体场景（JSON 结构与模拟器 Experiment 一致）。

    speed=None 时取赤道圆轨道速度；radial 是速度相对 φ̂ 的偏角（度），
    用来做偏心/径向轨道。
    """
    theta = math.radians(theta_deg)
    pos = ks_from_bl(r, theta, phi_ks, a)
    if speed is None:
        speed = circular_speed(r, a, M)
    rad = math.radians(radial)
    vel = (speed * math.sin(rad), 0.0, speed * math.cos(rad))

    cam = camera or {"x": -24.0, "y": 0.0, "z": 6.0,
                     "yaw": 0.0, "pitch": 0.0, "fov": 55.0}
    return {
        "version": 1,
        "name": name or f"M={M:g}, a={a:g}, r0={r:g}",
        "M": M, "a": a, "Q": 0.0,
        "duration": duration, "maxR": max_r,
        "camera": cam,
        "disk": {"enabled": disk, "rIn": disk_in, "rOut": disk_out, "temp": 1.0},
        "bodies": [{
            "name": "Probe", "kind": kind,
            "x": pos[0], "y": pos[1], "z": pos[2],
            "vx": vel[0], "vy": vel[1], "vz": vel[2],
            "mass": mass, "radius": 0.25, "color": "#ffb347",
        }],
    }


def _scene_arg(preset, scene_obj, tmpdir):
    if scene_obj is not None:
        path = Path(tmpdir) / "scene.json"
        path.write_text(json.dumps(scene_obj, ensure_ascii=False), encoding="utf-8")
        return ["--scene", str(path)]
    if preset is None:
        raise SimulatorError("需要 preset 或 scene 之一")
    return ["--preset", str(preset)]


# ------------------------------------------------------------------ 调用 ----
def trace(preset=None, scene_obj=None, timeout=300) -> dict:
    """积分并返回轨迹 JSON（含 BL 坐标与协变动量）。"""
    with tempfile.TemporaryDirectory(prefix="bh_trace_") as tmp:
        out = Path(tmp) / "trace.json"
        args = _scene_arg(preset, scene_obj, tmp)
        _run([*args, "--no-render", "--trace", str(out)], timeout=timeout)
        if not out.exists():
            raise SimulatorError("模拟器没有生成轨迹文件")
        return json.loads(out.read_text(encoding="utf-8"))


def render(preset=None, scene_obj=None, width=960, height=540, steps=6000,
           out_path=None, timeout=600) -> Path:
    """渲染一张图，返回 PNG 路径。"""
    if out_path is None:
        raise SimulatorError("需要 out_path")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bh_render_") as tmp:
        args = _scene_arg(preset, scene_obj, tmp)
        _run([*args, "--width", str(width), "--height", str(height),
              "--steps", str(steps), "--out", str(out_path)], timeout=timeout)
    if not out_path.exists():
        raise SimulatorError("模拟器没有生成图像")
    return out_path


# -------------------------------------------------------- 候选 Q 的数值监测 ----
def _compile(poly: dict, bg, params: dict):
    expr = to_expr(poly, bg.momenta)
    if params:
        expr = expr.subs(params)
    return sp.lambdify([*bg.coords, *bg.momenta], expr, modules="numpy")


def _bl_inverse(r, theta, M, a):
    """Boyer–Lindquist 逆度规 (t, r, θ, φ) 的五个独立分量。"""
    s2 = math.sin(theta) ** 2
    S = r * r + a * a * math.cos(theta) ** 2
    D = r * r - 2.0 * M * r + a * a
    A = (r * r + a * a) ** 2 - a * a * D * s2
    return {
        "tt": -A / (S * D),
        "tp": -2.0 * M * a * r / (S * D),
        "rr": D / S,
        "thth": 1.0 / S,
        "pp": (D - a * a * s2) / (S * D * s2),
    }


def momenta_series(samples, M, a, mass=1.0, twoH=None) -> list:
    """把模拟器采样转成 (t, r, u, phi; p_t, p_r, p_u, p_φ)。

    $p_t, p_\\theta, p_\\phi$ 直接来自模拟器（它们已被 Carter 常数的高精度守恒
    验证过）；$p_r$ 由**质量壳**定出，符号取 $\\dot r$ 的方向：

        g^{rr} p_r^2 = 2H - g^{tt}p_t^2 - 2g^{t\\phi}p_tp_\\phi
                            - g^{\\theta\\theta}p_\\theta^2 - g^{\\phi\\phi}p_\\phi^2

    不用 Kerr-Schild→BL 的 Jacobian，是因为 $t_{KS} \\ne t_{BL}$，
    $\\partial t_{KS}/\\partial r$ 的约定很容易出错；质量壳给出的是同一个
    $p_r$，而且与模拟器自身的 2H 检查自洽。
    """
    if twoH is None:
        twoH = -(mass ** 2)
    out = []
    n = len(samples)
    for i, s in enumerate(samples):
        theta = float(s["theta"])
        sin_t = math.sin(theta)
        if abs(sin_t) < 1e-9:
            out.append(None)
            continue
        r = float(s["r"])
        pt = float(s["pt"])
        pth = float(s["pth"])
        pph = float(s["pph"])
        g = _bl_inverse(r, theta, M, a)
        if g["rr"] == 0.0:
            out.append(None)
            continue
        rhs = (twoH - g["tt"] * pt * pt - 2.0 * g["tp"] * pt * pph
               - g["thth"] * pth * pth - g["pp"] * pph * pph)
        pr2 = rhs / g["rr"]
        pr = math.sqrt(pr2) if pr2 > 0.0 else 0.0

        # 符号：dr/dλ = g^{rr} p_r，g^{rr} > 0（视界外）
        if n > 1:
            if i == 0:
                dr = float(samples[1]["r"]) - r
            elif i == n - 1:
                dr = r - float(samples[-2]["r"])
            else:
                dr = float(samples[i + 1]["r"]) - float(samples[i - 1]["r"])
            if dr < 0.0:
                pr = -pr

        out.append({
            "t": float(s["t"]),
            "r": r,
            "u": math.cos(theta),
            "phi": float(s["phi"]),
            "pt": pt,
            "pr": pr,
            "pu": -pth / sin_t,
            "pph": pph,
        })
    return out


def evaluate_quantity(poly: dict, bg, params: dict, samples: list,
                      label: str = "Q", M=1.0, a=0.0, mass=1.0) -> dict:
    """沿一条轨迹计算候选 Q，返回漂移统计。

    注意这里用的是**符号层给出的同一个 K^{μν}**，
    所以数值曲线和符号证明检验的是同一个对象。
    """
    fn = _compile(poly, bg, params)
    series = momenta_series(samples, M, a, mass=mass)
    lam, values, skipped = [], [], 0
    for s, m in zip(samples, series):
        if m is None:
            skipped += 1
            continue
        try:
            val = float(fn(m["t"], m["r"], m["u"], m["phi"],
                           m["pt"], m["pr"], m["pu"], m["pph"]))
        except Exception:                                  # noqa: BLE001
            skipped += 1
            continue
        if not math.isfinite(val):
            skipped += 1
            continue
        lam.append(float(s["lam"]))
        values.append(val)

    if len(values) < 2:
        return {"label": label, "ok": False, "reason": "有效采样点不足",
                "skipped": skipped}

    v0 = values[0]
    scale = max(abs(v0), 1e-30)
    drift = [(v - v0) / scale for v in values]
    return {
        "label": label,
        "ok": True,
        "lam": lam,
        "values": values,
        "drift": drift,
        "v0": v0,
        "v_last": values[-1],
        "rel_drift": abs(values[-1] - v0) / scale,
        "max_abs_drift": max(abs(d) for d in drift),
        "n": len(values),
        "skipped": skipped,
    }


def carter_from_samples(samples: list) -> dict:
    """模拟器自带的 Carter 常数漂移，用作对照。"""
    vals = [float(s["Qcarter"]) for s in samples]
    if len(vals) < 2:
        return {"ok": False}
    v0 = vals[0]
    scale = max(abs(v0), 1e-30)
    return {"ok": True, "v0": v0,
            "rel_drift": abs(vals[-1] - v0) / scale,
            "lam": [float(s["lam"]) for s in samples], "values": vals}


def hamiltonian_check(samples: list) -> dict:
    """2H 应当恒为 -m²（有质量）或 0（光子）。"""
    vals = [float(s["H2"]) for s in samples]
    if not vals:
        return {"ok": False}
    return {"ok": True, "min": min(vals), "max": max(vals),
            "spread": max(vals) - min(vals)}
