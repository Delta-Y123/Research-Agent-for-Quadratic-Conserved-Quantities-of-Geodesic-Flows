"""Agent 主循环：LLM 规划 + 确定性工具 + 策略守门 + 台账。

三种运行方式
------------
* `run_llm`   —— LLM 决定下一步（默认）
* `run_rules` —— 规则表决定下一步（离线可跑，用于对照实验）
* 单步        —— 直接调用某个工具（见 cli.py --tool）

所有过程事件通过 `on_event(payload)` 推给界面；payload 是可 JSON 化的 dict。
"""

from __future__ import annotations

import json
from pathlib import Path

import policy
import tools
from ledger import Ledger
from llm import LLMClient
from prompts import CRITIQUE_TEMPLATE, SYSTEM_PROMPT

HEAVY_KEYS = ("constraints", "grid_history", "solutions", "known_invariants",
              "verified_kernel_vectors", "h_basis", "pairing_relations")


def compact(obj, limit: int = 3500) -> str:
    text = json.dumps(obj, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    if isinstance(obj, dict):
        slim = dict(obj)
        for key in HEAVY_KEYS:
            value = slim.get(key)
            if isinstance(value, list) and len(value) > 3:
                slim[key] = value[:3] + [f"... ({len(value) - 3} more)"]
        text = json.dumps(slim, ensure_ascii=False, default=str)
    if len(text) > limit:
        text = text[:limit] + " ...(truncated)"
    return text


class ResearchAgent:
    def __init__(self, question: str, run_dir: Path, client: LLMClient | None = None,
                 max_steps: int = 10, budget_seconds: float = 1800,
                 verbose: bool = True, on_event=None):
        self.question = question
        self.run_dir = Path(run_dir)
        self.client = client
        self.max_steps = max_steps
        self.budget_seconds = budget_seconds
        self.verbose = verbose
        self.on_event = on_event
        self.cancelled = False
        self.ledger = Ledger(self.run_dir, question)
        self.ctx = {"run_dir": str(self.run_dir)}

    # ------------------------------------------------------------- 内部 ----
    def _say(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def _emit(self, **payload):
        if self.on_event is None:
            return
        try:
            self.on_event(payload)
        except Exception:                                   # noqa: BLE001
            pass

    def cancel(self):
        self.cancelled = True

    def _execute(self, name, args, reasoning=""):
        result = tools.dispatch(name, args, self.ctx)
        diag = policy.diagnose(result)
        entry = self.ledger.add(name, args, result, diag,
                                result.get("_seconds", 0.0), reasoning)
        self._say(f"  [{entry.step}] {name} -> {diag.get('kind')} "
                  f"({entry.seconds:.1f}s) {diag.get('reason', '')[:90]}")
        self._emit(type="step", step=entry.step, tool=name, args=args,
                   kind=diag.get("kind"), reason=diag.get("reason"),
                   advice=diag.get("advice"), seconds=entry.seconds,
                   brief=tools.brief(result),
                   totals={"steps": len(self.ledger.entries),
                           "seconds": round(self.ledger.total_seconds, 2)})
        return entry, result, diag

    # --------------------------------------------------------- LLM 模式 ----
    def run_llm(self) -> dict:
        if self.client is None:
            raise ValueError("LLM 模式需要 LLMClient")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.question},
        ]
        critiques = 0
        final = None
        self._emit(type="start", mode="llm", question=self.question,
                   max_steps=self.max_steps, budget=self.budget_seconds)
        for step in range(self.max_steps):
            if self.cancelled:
                self._emit(type="cancelled")
                break
            if self.ledger.total_seconds > self.budget_seconds:
                self.ledger.note(f"超出时间预算 {self.budget_seconds}s，提前停止")
                break
            self._say(f"\n--- step {step + 1} ---")
            self._emit(type="round", round=step + 1, max=self.max_steps)
            resp = self.client.chat(messages, tools=tools.TOOL_SCHEMAS)
            if resp.reasoning:
                self._say(f"  (thinking) {resp.reasoning[:160].strip()}...")
                self._emit(type="thinking", round=step + 1,
                           text=resp.reasoning[:4000])

            if not resp.tool_calls:
                content = (resp.content or "").strip()
                self._say(f"  (no tool call) {content[:200]}")
                self._emit(type="note", text=content[:2000] or "(空回复)")
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": "请不要只做文字说明。要么调用一个工具继续实验，"
                               "要么调用 finish 提交结论（每条结论要带 evidence 和 scope）。"})
                continue

            messages.append(resp.assistant_message())
            finished = False
            for tc in resp.tool_calls:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                entry, result, diag = self._execute(name, args, resp.reasoning)
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": compact(result)})
                if name == "finish":
                    problems = policy.check_claims(args.get("claims"), self.ledger)
                    if problems:
                        critiques += 1
                        self._say(f"  [policy] {len(problems)} 个问题，打回")
                        self._emit(type="critique", problems=problems,
                                   attempt=critiques)
                        messages.append({
                            "role": "user",
                            "content": CRITIQUE_TEMPLATE.format(
                                problems=json.dumps(problems, ensure_ascii=False,
                                                    indent=2))})
                        if critiques >= 3:
                            self.ledger.note("策略层连续 3 次打回，按最后一次提交记录")
                            final = args
                            finished = True
                    else:
                        final = args
                        finished = True
                        self._say("  [policy] 结论通过检查")
                        self._emit(type="accepted", claims=args.get("claims", []))
            if finished:
                break
        if final is None:
            final = self._fallback_final("达到步数上限或预算上限，未能提交完整结论")
        self.ledger.set_final(final)
        self._emit(type="final", summary=final.get("summary", ""),
                   claims=final.get("claims", []))
        return final

    # -------------------------------------------------------- 规则模式 ----
    def run_rules(self) -> dict:
        self._say("规则模式（无 LLM）")
        self._emit(type="start", mode="rules", question=self.question,
                   max_steps=self.max_steps, budget=self.budget_seconds)
        for _ in range(self.max_steps):
            if self.cancelled:
                self._emit(type="cancelled")
                break
            actions = policy.suggest(self.ledger, self.budget_seconds,
                                     self.question)
            if not actions:
                break
            action = actions[0]
            if action["action"] == "finish":
                break
            if self.ledger.total_seconds > self.budget_seconds:
                self.ledger.note("超出时间预算，提前停止")
                break
            self._say(f"\n--- {action['action']} （{action.get('why','')}）---")
            self._emit(type="plan", action=action["action"],
                       why=action.get("why", ""))
            entry, result, diag = self._execute(action["action"],
                                                action.get("args") or {},
                                                action.get("why", ""))
            if diag.get("kind") == "artifact":
                continue
        final = policy.compose_claims(self.ledger)
        self.ledger.set_final(final)
        self._emit(type="final", summary=final.get("summary", ""),
                   claims=final.get("claims", []))
        return final

    # ------------------------------------------------------------- 收尾 ----
    def _fallback_final(self, reason: str) -> dict:
        base = policy.compose_claims(self.ledger)
        base["summary"] = (base.get("summary", "") + f"\n\n注意：{reason}。").strip()
        return base

    def finish(self) -> str:
        report = self.ledger.write_report()
        self._emit(type="done", report=report,
                   run_dir=str(self.run_dir),
                   totals={"steps": len(self.ledger.entries),
                           "seconds": round(self.ledger.total_seconds, 2)})
        return report
