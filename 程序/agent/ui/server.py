#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地服务：把 Agent 与模拟器接到浏览器上。

只用标准库（http.server + SSE），不引入任何框架。

    python ui/server.py            # http://127.0.0.1:8765
    python ui/server.py --port 9000 --open

接口
----
GET  /                          单页界面
GET  /static/...                前端静态资源（含本地 KaTeX）
GET  /media/<run>/<file>        运行产物（渲染图）
GET  /api/info                  环境信息（模拟器可用性、预设、背景、模型）
POST /api/run                   启动一次 Agent 运行 -> {run_id}
GET  /api/run/<id>/stream       SSE 事件流
POST /api/run/<id>/cancel       取消
GET  /api/run/<id>/report       生成的 markdown 报告
GET  /api/runs                  历史运行列表
POST /api/tool                  单步调用一个工具
POST /api/sim/trace             积分一条轨迹，返回 BL 状态序列
POST /api/sim/monitor           沿轨迹监测候选 Q 的漂移
POST /api/sim/render            渲染一张黑洞图像
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import queue
import sys
import threading
import traceback
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

UI_DIR = Path(__file__).resolve().parent
AGENT_DIR = UI_DIR.parent
RUNS_DIR = AGENT_DIR / "runs"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

try:
    import sympy as sp  # noqa: E402

    import backgrounds as bgmod  # noqa: E402
    import simulator as sim  # noqa: E402
    import tools  # noqa: E402
    from agent import ResearchAgent  # noqa: E402
    from engine import parse_number, to_expr  # noqa: E402
    from llm import DEFAULT_MODEL, DEFAULT_URL, LLMClient  # noqa: E402
except ImportError as exc:                              # pragma: no cover
    print(f"缺少依赖：{exc.name}")
    print("请先安装： python -m pip install sympy")
    print(f"（当前 Python：{sys.executable}）")
    raise SystemExit(1)

MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".woff2": "font/woff2",
    ".woff": "font/woff",
    ".ttf": "font/ttf",
    ".md": "text/markdown; charset=utf-8",
}

# --------------------------------------------------------------- 运行管理 ----
RUNS: dict[str, dict] = {}
RUNS_LOCK = threading.Lock()


def _new_run_dir(tag: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RUNS_DIR / f"{stamp}-{tag}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (bool, int, float)) or obj is None:
        return obj
    if isinstance(obj, str):
        return obj
    try:
        return sp.sstr(obj)
    except Exception:                                    # noqa: BLE001
        return str(obj)


def start_run(payload: dict) -> dict:
    question = (payload.get("question") or "").strip()
    if not question:
        raise ValueError("缺少 question")
    mode = payload.get("mode", "llm")
    max_steps = int(payload.get("max_steps", 10))
    budget = float(payload.get("budget", 1800))
    model = payload.get("model") or DEFAULT_MODEL
    effort = payload.get("reasoning_effort") or "high"

    run_dir = _new_run_dir(mode)
    run_id = run_dir.name
    q: queue.Queue = queue.Queue()
    holder: dict = {"agent": None, "cancelled": False}

    def on_event(payload_event: dict):
        q.put(_jsonable(payload_event))

    client = None
    if mode == "llm":
        key = os.environ.get("MIMO_API_KEY", "")
        if not key:
            raise ValueError("服务进程没有 MIMO_API_KEY；请用 --mode rules 或先设置环境变量")
        client = LLMClient(key, url=DEFAULT_URL, model=model,
                           reasoning_effort=effort, verbose=False)

    def worker():
        try:
            agent = ResearchAgent(question, run_dir, client=client,
                                  max_steps=max_steps, budget_seconds=budget,
                                  verbose=False, on_event=on_event)
            holder["agent"] = agent
            final = agent.run_llm() if mode == "llm" else agent.run_rules()
            agent.finish()
            q.put({"type": "close", "final": _jsonable(final)})
        except Exception as exc:                          # noqa: BLE001
            q.put({"type": "error", "message": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc(limit=4)})
            q.put({"type": "close"})
        finally:
            with RUNS_LOCK:
                RUNS[run_id]["status"] = "done"

    with RUNS_LOCK:
        RUNS[run_id] = {"id": run_id, "dir": run_dir, "queue": q,
                        "status": "running", "question": question,
                        "mode": mode, "holder": holder, "started": datetime.now()}
    threading.Thread(target=worker, daemon=True).start()
    return {"run_id": run_id, "dir": str(run_dir)}


# ------------------------------------------------------------- 模拟器封装 ----
def _bg_and_params(background: str, parameters: dict):
    bg = bgmod.get(background)
    vals = dict(bg.params)
    for key, value in (parameters or {}).items():
        sym = next((s for s in bg.params if sp.sstr(s) == key), None)
        if sym is None:
            raise ValueError(f"{bg.key} 没有参数 {key}")
        vals[sym] = parse_number(value)
    return bg, vals


def sim_trace(payload: dict) -> dict:
    preset = payload.get("preset")
    scene_obj = payload.get("scene")
    if scene_obj is None and payload.get("custom"):
        c = payload["custom"]
        scene_obj = sim.scene(
            M=float(c.get("M", 1.0)), a=float(c.get("a", 0.0)),
            r=float(c.get("r", 10.0)), theta_deg=float(c.get("theta_deg", 90.0)),
            phi_ks=float(c.get("phi_ks", 0.0)),
            speed=c.get("speed"), radial=float(c.get("radial", 0.0)),
            duration=float(c.get("duration", 600.0)),
            kind=c.get("kind", "particle"))
    data = sim.trace(preset=preset, scene_obj=scene_obj)
    for body in data.get("bodies", []):
        body["samples"] = body.get("samples", [])[::max(1, len(body.get("samples", [])) // 400)]
    return data


def sim_monitor(payload: dict) -> dict:
    preset = payload.get("preset")
    scene_obj = payload.get("scene")
    if scene_obj is None and payload.get("custom"):
        c = payload["custom"]
        scene_obj = sim.scene(
            M=float(c.get("M", 1.0)), a=float(c.get("a", 0.0)),
            r=float(c.get("r", 10.0)), theta_deg=float(c.get("theta_deg", 90.0)),
            phi_ks=float(c.get("phi_ks", 0.0)),
            speed=c.get("speed"), radial=float(c.get("radial", 0.0)),
            duration=float(c.get("duration", 600.0)),
            kind=c.get("kind", "particle"))

    background = payload.get("background", "kerr")
    bg, vals = _bg_and_params(background, payload.get("parameters"))
    labels = payload.get("labels") or ["Q_C", "H"]
    perturb = payload.get("perturb") or None

    data = sim.trace(preset=preset, scene_obj=scene_obj)
    body = (data.get("bodies") or [{}])[0]
    samples = body.get("samples", [])
    M = float(data.get("M", 1.0))
    a = float(data.get("a", 0.0))
    mass = float(body.get("mass", 1.0))

    known = {lab: poly for lab, poly, _ in bg.known(vals)}
    series = []
    for label in labels:
        poly = known.get(label)
        if poly is None:
            continue
        res = sim.evaluate_quantity(poly, bg, vals, samples, label,
                                    M=M, a=a, mass=mass)
        if res.get("ok"):
            series.append(_thin(res))

    if perturb:
        base_label = perturb.get("label", "Q_C")
        delta = float(perturb.get("delta", 0.05))
        key = tuple(perturb.get("key") or (0, 0, 0, 2))
        poly = dict(known.get(base_label) or {})
        if key in poly:
            poly[key] = poly[key] * (1.0 + delta)
            res = sim.evaluate_quantity(poly, bg, vals, samples,
                                        f"{base_label} 系数×(1+{delta:g})",
                                        M=M, a=a, mass=mass)
            if res.get("ok"):
                series.append(_thin(res))

    carter = sim.carter_from_samples(samples)
    ham = sim.hamiltonian_check(samples)
    return {
        "M": M, "a": a, "mass": mass,
        "body": {"name": body.get("name"), "kind": body.get("kind"),
                 "stop": body.get("stop"), "message": body.get("message"),
                 "samples": len(samples)},
        "series": series,
        "carter": {"rel_drift": carter.get("rel_drift")} if carter.get("ok") else None,
        "hamiltonian": ham,
    }


def _thin(res: dict, n: int = 300) -> dict:
    lam, vals, drift = res["lam"], res["values"], res["drift"]
    step = max(1, len(lam) // n)
    return {
        "label": res["label"],
        "lam": lam[::step],
        "values": vals[::step],
        "drift": drift[::step],
        "v0": res["v0"],
        "rel_drift": res["rel_drift"],
        "max_abs_drift": res["max_abs_drift"],
        "n": res["n"],
    }


def sim_render(payload: dict) -> dict:
    preset = payload.get("preset")
    scene_obj = payload.get("scene")
    if scene_obj is None and payload.get("custom"):
        c = payload["custom"]
        scene_obj = sim.scene(
            M=float(c.get("M", 1.0)), a=float(c.get("a", 0.0)),
            r=float(c.get("r", 10.0)), theta_deg=float(c.get("theta_deg", 90.0)),
            phi_ks=float(c.get("phi_ks", 0.0)),
            speed=c.get("speed"), radial=float(c.get("radial", 0.0)),
            duration=float(c.get("duration", 600.0)),
            kind=c.get("kind", "particle"))
    run_dir = _new_run_dir("render")
    out = run_dir / "render.png"
    sim.render(preset=preset, scene_obj=scene_obj,
               width=int(payload.get("width", 960)),
               height=int(payload.get("height", 540)),
               steps=int(payload.get("steps", 6000)),
               out_path=out)
    return {"url": f"/media/{run_dir.name}/render.png"}


def tool_call(payload: dict) -> dict:
    name = payload.get("name")
    args = payload.get("args") or {}
    result = tools.dispatch(name, args, {})
    return {"brief": tools.brief(result),
            "raw": _jsonable(result)}


# ------------------------------------------------------------------ 服务 ----
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "BHResearchUI/1.0"

    def log_message(self, fmt, *args):                   # noqa: A003
        if os.environ.get("BH_UI_VERBOSE"):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # ------------------------------------------------------------ 工具 ----
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False,
                                    default=str).encode("utf-8"), MIME[".json"])

    def _error(self, code: int, message: str, detail: str = ""):
        self._json({"error": message, "detail": detail}, code)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:                                # noqa: BLE001
            return {}

    def _file(self, path: Path):
        if not path.is_file():
            self._error(404, "not found", str(path))
            return
        ctype = MIME.get(path.suffix.lower()) or mimetypes.guess_type(str(path))[0] \
            or "application/octet-stream"
        self._send(200, path.read_bytes(), ctype)

    # ------------------------------------------------------------ 路由 ----
    def do_GET(self):                                    # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path in ("/", "/index.html"):
            self._file(UI_DIR / "index.html")
            return
        if path.startswith("/static/"):
            self._file((UI_DIR / path[len("/static/"):]).resolve())
            return
        if path.startswith("/media/"):
            self._file((RUNS_DIR / path[len("/media/"):]).resolve())
            return

        if path == "/api/info":
            self._json({
                "simulator": sim.available(),
                "sim_path": str(sim.CLI),
                "presets": (sim.list_presets() if sim.available() else []),
                "backgrounds": [
                    {"key": k, "name": b.name, "dim": b.dim,
                     "params": {sp.sstr(p): sp.sstr(v) for p, v in b.params.items()},
                     "known": [lab for lab, _, _ in b.known(dict(b.params))]}
                    for k, b in bgmod.BACKGROUNDS.items()],
                "tools": tools.tool_names(),
                "model": DEFAULT_MODEL,
                "has_key": bool(os.environ.get("MIMO_API_KEY")),
            })
            return

        if path == "/api/runs":
            items = []
            if RUNS_DIR.exists():
                for d in sorted(RUNS_DIR.iterdir(), reverse=True):
                    if d.is_dir():
                        items.append({"id": d.name,
                                      "has_report": (d / "report.md").exists()})
            self._json({"runs": items[:40]})
            return

        if path.startswith("/api/run/") and path.endswith("/report"):
            run_id = path[len("/api/run/"):-len("/report")]
            f = RUNS_DIR / run_id / "report.md"
            if not f.exists():
                self._error(404, "no report", run_id)
                return
            self._send(200, f.read_bytes(), MIME[".md"])
            return

        if path.startswith("/api/run/") and path.endswith("/stream"):
            self._stream(path[len("/api/run/"):-len("/stream")])
            return

        self._error(404, "unknown path", path)

    def do_POST(self):                                   # noqa: N802
        path = unquote(urlparse(self.path).path)
        try:
            payload = self._body()
            if path == "/api/run":
                self._json(start_run(payload))
            elif path == "/api/tool":
                self._json(tool_call(payload))
            elif path == "/api/sim/trace":
                self._json(sim_trace(payload))
            elif path == "/api/sim/monitor":
                self._json(sim_monitor(payload))
            elif path == "/api/sim/render":
                self._json(sim_render(payload))
            elif path.startswith("/api/run/") and path.endswith("/cancel"):
                run_id = path[len("/api/run/"):-len("/cancel")]
                with RUNS_LOCK:
                    state = RUNS.get(run_id)
                if not state:
                    self._error(404, "unknown run", run_id)
                    return
                agent = state["holder"].get("agent")
                if agent:
                    agent.cancel()
                self._json({"cancelled": True})
            else:
                self._error(404, "unknown path", path)
        except sim.SimulatorError as exc:
            self._error(503, "模拟器不可用", str(exc))
        except Exception as exc:                         # noqa: BLE001
            self._error(500, f"{type(exc).__name__}: {exc}",
                        traceback.format_exc(limit=5))

    # ------------------------------------------------------------ SSE ----
    def _stream(self, run_id: str):
        with RUNS_LOCK:
            state = RUNS.get(run_id)
        if not state:
            self._error(404, "unknown run", run_id)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def chunk(data: bytes):
            self.wfile.write(f"{len(data):X}\r\n".encode("ascii"))
            self.wfile.write(data)
            self.wfile.write(b"\r\n")
            self.wfile.flush()

        q: queue.Queue = state["queue"]
        try:
            chunk(b": connected\n\n")
            while True:
                try:
                    event = q.get(timeout=20)
                except queue.Empty:
                    chunk(b": keepalive\n\n")
                    continue
                payload = json.dumps(event, ensure_ascii=False, default=str)
                chunk(f"data: {payload}\n\n".encode("utf-8"))
                if event.get("type") == "close":
                    break
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except Exception:                            # noqa: BLE001
                pass


def main(argv=None):
    ap = argparse.ArgumentParser(description="研究 Agent 本地界面")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--open", action="store_true", help="启动后打开浏览器")
    args = ap.parse_args(argv)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    url = f"http://{args.host}:{args.port}/"
    print(f"界面已启动：{url}")
    print(f"模拟器：{sim.CLI}  ({'可用' if sim.available() else '不可用'})")
    print(f"模型：{DEFAULT_MODEL}  密钥：{'已设置' if os.environ.get('MIMO_API_KEY') else '未设置'}")
    print("Ctrl+C 停止")
    if args.open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    return 0


if __name__ == "__main__":
    sys.exit(main())
