"""HTTP 传输层。

本机（Windows）实测：Python 的 OpenSSL TLS 指纹会被网关 WAF 拦成
403 "当前网络环境不支持"，而 curl.exe（Schannel）可以通过。
因此默认优先用 curl.exe，失败再退回 requests。
在别的网络环境下 requests 通常也能直接工作。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile

import requests


class TransportError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:600]}")
        self.status = status
        self.body = body


def _curl_path():
    return shutil.which("curl") or shutil.which("curl.exe")


def _post_curl(url, headers, payload, timeout):
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        cmd = [_curl_path(), "-s", "-S", "-X", "POST", url,
               "--max-time", str(int(timeout)),
               "--data-binary", f"@{path}"]
        for k, v in headers.items():
            cmd += ["-H", f"{k}: {v}"]
        out = subprocess.run(cmd, capture_output=True, text=True,
                             encoding="utf-8", timeout=timeout + 30)
        body = out.stdout or ""
        if out.stderr.strip():
            raise TransportError(0, f"curl stderr: {out.stderr.strip()}")
        if not body.strip():
            raise TransportError(0, "empty response body")
        return body
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _post_requests(url, headers, payload, timeout):
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code >= 400:
        raise TransportError(resp.status_code, resp.text)
    return resp.text


def post_json(url, api_key, payload, timeout=600, prefer="curl"):
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {api_key}"}
    order = ("curl", "requests") if prefer == "curl" else ("requests", "curl")
    errors = []
    for backend in order:
        if backend == "curl" and not _curl_path():
            continue
        try:
            body = (_post_curl(url, headers, payload, timeout)
                    if backend == "curl"
                    else _post_requests(url, headers, payload, timeout))
            return json.loads(body)
        except TransportError as exc:
            errors.append(f"{backend}: {exc}")
        except json.JSONDecodeError as exc:
            errors.append(f"{backend}: bad JSON ({exc})")
    raise TransportError(0, " | ".join(errors))
