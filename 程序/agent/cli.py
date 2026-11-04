#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""命令行入口。

    python cli.py list                       # 看有哪些背景
    python cli.py tool <name> '<json>'       # 单步调用一个工具（不需要 LLM）
    python cli.py ask "研究问题"             # 完整 Agent（需要 LLM）
    python cli.py ask "研究问题" --mode rules # 规则模式（离线可跑）

环境变量：
    MIMO_API_KEY     必填（LLM 模式）
    MIMO_BASE_URL    默认 https://agi.tongji.edu.cn/v1/chat/completions
    MIMO_MODEL       默认 MiMo-V2.6-Pro
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import tools  # noqa: E402

DEFAULT_BASE = os.environ.get("MIMO_BASE_URL",
                              "https://agi.tongji.edu.cn/v1/chat/completions")
DEFAULT_MODEL = os.environ.get("MIMO_MODEL", "MiMo-V2.6-Pro")


def _run_dir(base: Path, tag: str = "") -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}-{tag}" if tag else stamp
    path = base / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def cmd_list(args):
    result = tools.dispatch("list_backgrounds", {}, {})
    for bg in result["backgrounds"]:
        print(f"\n[{bg['key']}] {bg['name']}")
        print(f"  维数 {bg['dim']}  坐标 {bg['coords']}  动量 {bg['momenta']}")
        print(f"  参数 {bg['parameters']}")
        print(f"  已知不变量 {bg['known_invariants']}")
        print(f"  备注 {bg['notes']}")


def cmd_tool(args):
    payload = json.loads(args.json_args) if args.json_args else {}
    result = tools.dispatch(args.name, payload, {})
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text if len(text) < 12000 else text[:12000] + "\n...(truncated)")
    if args.save:
        Path(args.save).write_text(text, encoding="utf-8")


def cmd_ask(args):
    from agent import ResearchAgent
    from llm import LLMClient

    run_dir = _run_dir(Path(args.out), args.mode)
    client = None
    if args.mode == "llm":
        key = os.environ.get("MIMO_API_KEY", "")
        if not key:
            print("缺少 MIMO_API_KEY；可改用 --mode rules 离线运行", file=sys.stderr)
            return 2
        client = LLMClient(key, url=args.base_url, model=args.model,
                           temperature=args.temperature,
                           max_tokens=args.max_tokens,
                           reasoning_effort=args.reasoning_effort,
                           verbose=args.verbose)

    bot = ResearchAgent(args.question, run_dir, client=client,
                        max_steps=args.max_steps,
                        budget_seconds=args.budget,
                        verbose=not args.quiet)
    print(f"研究问题：{args.question}")
    print(f"运行目录：{run_dir}\n")
    final = bot.run_llm() if args.mode == "llm" else bot.run_rules()
    report = bot.finish()

    print("\n" + "=" * 72)
    print(report)
    if client is not None:
        print(f"\nLLM 调用 {client.calls} 次，累计 {client.total_tokens} tokens")
    print(f"\n产物：{run_dir}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="测地流二次守恒量研究 Agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="列出背景度规")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("tool", help="单步调用工具")
    p.add_argument("name")
    p.add_argument("json_args", nargs="?", default="{}")
    p.add_argument("--save", help="把结果存成 JSON 文件")
    p.set_defaults(func=cmd_tool)

    p = sub.add_parser("ask", help="跑完整 Agent")
    p.add_argument("question")
    p.add_argument("--mode", choices=["llm", "rules"], default="llm")
    p.add_argument("--out", default=str(HERE / "runs"))
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--budget", type=float, default=1800.0)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--base-url", default=DEFAULT_BASE)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--reasoning-effort", default="high")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_ask)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
