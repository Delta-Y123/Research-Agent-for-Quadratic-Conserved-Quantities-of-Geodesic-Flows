"""OpenAI 兼容的 LLM 客户端（同济 MiMo 端点）。

实测该端点：
* 支持原生 function calling（finish_reason = "tool_calls"）；
* 支持 response_format = {"type": "json_object"}；
* 思维链放在 message["reasoning"]，思考默认开启。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from transport import TransportError, post_json

DEFAULT_URL = "https://agi.tongji.edu.cn/v1/chat/completions"
DEFAULT_MODEL = "MiMo-V2.6-Pro"


@dataclass
class LLMResponse:
    content: str = ""
    reasoning: str = ""
    tool_calls: list = field(default_factory=list)
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    def assistant_message(self) -> dict:
        msg = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg

    def summary(self) -> dict:
        return {"finish_reason": self.finish_reason,
                "content": self.content,
                "reasoning": self.reasoning,
                "tool_calls": [{"name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"]}
                               for tc in self.tool_calls],
                "usage": self.usage}


class LLMClient:
    def __init__(self, api_key: str, url: str = DEFAULT_URL,
                 model: str = DEFAULT_MODEL, temperature: float = 0.3,
                 max_tokens: int = 4096, reasoning_effort: str = "high",
                 timeout: int = 600, max_retries: int = 3,
                 verbose: bool = False):
        if not api_key:
            raise ValueError("缺少 API key")
        self.api_key = api_key
        self.url = url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.max_retries = max_retries
        self.verbose = verbose
        self.calls = 0
        self.total_tokens = 0

    def chat(self, messages, tools=None, tool_choice=None,
             response_format=None, max_tokens=None) -> LLMResponse:
        payload = {"model": self.model,
                   "messages": messages,
                   "max_tokens": max_tokens or self.max_tokens,
                   "temperature": self.temperature,
                   "stream": False}
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        if response_format:
            payload["response_format"] = response_format

        last_error = None
        for attempt in range(self.max_retries):
            try:
                data = post_json(self.url, self.api_key, payload,
                                 timeout=self.timeout)
            except TransportError as exc:
                last_error = exc
                if 400 <= (exc.status or 0) < 500 and exc.status != 429:
                    raise
                time.sleep(2 ** attempt)
                continue
            self.calls += 1
            usage = data.get("usage") or {}
            self.total_tokens += usage.get("total_tokens") or 0
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            return LLMResponse(
                content=msg.get("content") or "",
                reasoning=msg.get("reasoning") or "",
                tool_calls=msg.get("tool_calls") or [],
                finish_reason=choice.get("finish_reason") or "",
                usage=usage,
                raw=data,
            )
        raise TransportError(0, f"重试 {self.max_retries} 次仍失败: {last_error}")

    # ---------------------------------------------------------------- 便捷 ----
    def ask_json(self, system: str, user: str, max_tokens: int | None = None):
        """要求返回 JSON 对象；解析失败时抛出。"""
        resp = self.chat([{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                         response_format={"type": "json_object"},
                         max_tokens=max_tokens)
        return json.loads(resp.content), resp
