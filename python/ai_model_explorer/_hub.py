"""Minimal HTTP for Hugging Face: file list, small JSON files, and byte ranges. Standard library only."""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Optional

from . import __version__

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


class Hub:
    def __init__(self, token: Optional[str] = None, timeout: float = 60.0):
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self.timeout = timeout
        self.bytes_read = 0

    def get(self, url: str, rng: Optional[str] = None):
        """Return (status, body bytes). Never raises for HTTP errors; the caller decides."""
        url = url.replace("https://huggingface.co", ENDPOINT, 1)
        h = {"User-Agent": f"ai-model-explorer/{__version__}"}
        if rng:
            h["Range"] = rng
        if self.token and url.startswith(ENDPOINT):
            h["Authorization"] = f"Bearer {self.token}"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=self.timeout) as r:
                body = r.read()
                self.bytes_read += len(body)
                return r.status, body
        except urllib.error.HTTPError as e:
            return e.code, e.read() or b""
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            host = urllib.request.urlparse(url).netloc
            raise ConnectionError(f"cannot reach {host}: {getattr(e, 'reason', e)}") from None

    def json(self, url: str):
        st, body = self.get(url)
        if st in (401, 403):
            raise PermissionError("gated or private repository: accept its license on Hugging Face and pass a token (HF_TOKEN)")
        if st == 404:
            m = re.search(r"/api/models/([^?]+)", url)
            raise FileNotFoundError(f"repository not found: {m.group(1)}" if m else f"not found: {url}")
        if st >= 400:
            raise RuntimeError(f"HTTP {st} for {url}")
        return json.loads(body.decode("utf-8"))

    def model_info(self, repo: str) -> dict:
        return self.json(f"https://huggingface.co/api/models/{repo}?blobs=true")
