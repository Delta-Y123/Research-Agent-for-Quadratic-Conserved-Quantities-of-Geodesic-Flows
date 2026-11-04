"""实验记录：jsonl 台账 + 可读报告。

台账里的每一条都绑定：假设、动作、参数、诊断签名、证书、耗时。
报告固定三栏：已验证 / 仅在有限基内成立 / 未解决。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Entry:
    step: int
    tool: str
    args: dict
    diagnosis: dict
    seconds: float
    reasoning: str = ""
    result: dict = field(default_factory=dict)

    def brief(self) -> dict:
        return {"step": self.step, "tool": self.tool, "args": self.args,
                "kind": self.diagnosis.get("kind"),
                "reason": self.diagnosis.get("reason"),
                "seconds": self.seconds}


class Ledger:
    def __init__(self, run_dir: Path, question: str = ""):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.question = question
        self.entries: list[Entry] = []
        self.notes: list[str] = []
        self.final: dict | None = None
        self.started = datetime.now().isoformat(timespec="seconds")

    # ------------------------------------------------------------------ 写 ----
    def add(self, tool, args, result, diagnosis, seconds, reasoning="") -> Entry:
        entry = Entry(step=len(self.entries) + 1, tool=tool, args=args,
                      diagnosis=diagnosis, seconds=seconds,
                      reasoning=reasoning, result=result)
        self.entries.append(entry)
        with (self.run_dir / "ledger.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"step": entry.step, "tool": tool,
                                 "args": args, "diagnosis": diagnosis,
                                 "seconds": seconds,
                                 "reasoning": reasoning,
                                 "result": result},
                                ensure_ascii=False, default=str) + "\n")
        return entry

    def note(self, text: str):
        self.notes.append(text)

    def set_final(self, final: dict):
        self.final = final
        (self.run_dir / "final.json").write_text(
            json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ 读 ----
    @property
    def total_seconds(self) -> float:
        return sum(e.seconds for e in self.entries)

    def summary(self) -> dict:
        kinds = {}
        for e in self.entries:
            kinds[e.diagnosis.get("kind")] = kinds.get(e.diagnosis.get("kind"), 0) + 1
        return {"steps": len(self.entries), "seconds": round(self.total_seconds, 1),
                "kinds": kinds,
                "tools": [e.tool for e in self.entries]}

    # ---------------------------------------------------------------- 报告 ----
    def render_report(self) -> str:
        lines = [f"# 研究记录：{self.question}", "",
                 f"- 开始时间：{self.started}",
                 f"- 工具调用：{len(self.entries)} 次，累计 {self.total_seconds:.1f}s",
                 ""]

        lines += ["## 实验台账", "",
                  "| # | 工具 | 关键参数 | 诊断 | 判定 | 耗时 |",
                  "|---|---|---|---|---|---|"]
        for e in self.entries:
            args = json.dumps(e.args, ensure_ascii=False)
            if len(args) > 60:
                args = args[:57] + "..."
            reason = (e.diagnosis.get("reason") or "")[:48]
            lines.append(f"| {e.step} | `{e.tool}` | {args} | {reason} | "
                         f"{e.diagnosis.get('kind')} | {e.seconds:.1f}s |")
        lines.append("")

        if self.final:
            lines += ["## 结论", ""]
            verified, limited, open_items = [], [], []
            for claim in self.final.get("claims", []):
                bucket = {"verified": verified, "existence_in_span": verified,
                          "nonexistence_in_span": limited,
                          "classification": limited,
                          "inconclusive": open_items}.get(
                              claim.get("status"), open_items)
                bucket.append(claim)
            for title, bucket in (("已验证", verified),
                                  ("仅在有限候选空间内成立", limited),
                                  ("未解决 / 证据不足", open_items)):
                lines.append(f"### {title}")
                if not bucket:
                    lines.append("- （无）")
                for claim in bucket:
                    scope = claim.get("scope")
                    ev = ", ".join(f"#{e}" for e in claim.get("evidence", []))
                    lines.append(f"- {claim.get('text')}"
                                 + (f"　`适用范围：{scope}`" if scope else "")
                                 + (f"　证据：{ev}" if ev else ""))
                lines.append("")
            if self.final.get("summary"):
                lines += ["### 总结", "", self.final["summary"], ""]

        lines += ["## 证书", ""]
        for e in self.entries:
            cert = (e.result or {}).get("certificate")
            if cert:
                lines.append(f"- #{e.step} `{e.tool}`：{cert.get('type')} "
                             f"（{json.dumps({k: v for k, v in cert.items() if k != 'type'}, ensure_ascii=False)}）")
        lines.append("")

        if self.notes:
            lines += ["## 人工/策略备注", ""] + [f"- {n}" for n in self.notes] + [""]

        lines += ["## 适用边界", "",
                  "- 所有否定性结论只对声明过的有限候选空间成立，不是一般不存在性定理。",
                  "- 一阶结论只到 O(ε) ，精确可分离族含二阶及更高阶项。",
                  "- 未做规范固定：落在基内的纯规范方向会被判为 ALLOWED，但不构成新的守恒量。",
                  ""]
        return "\n".join(lines)

    def write_report(self):
        text = self.render_report()
        (self.run_dir / "report.md").write_text(text, encoding="utf-8")
        return text
