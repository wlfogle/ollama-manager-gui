import os
import time
from typing import Optional, Dict

import requests

HF_HEADERS = {"User-Agent": "ollama-manager-gui/1.0 (+https://github.com/wlfogle/ollama-manager-gui)", "Accept": "application/json"}


def hf_headers() -> Dict[str, str]:
    h = dict(HF_HEADERS)
    tok = os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACE_API_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok.strip()}"
    return h


def get_json_with_retry(url: str, params: Optional[dict] = None, headers: Optional[dict] = None, retries: int = 3, backoff: float = 1.0):
    last_err = None
    for i in range(max(1, retries)):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            if r.status_code in (429, 502, 503, 504):
                time.sleep(backoff * (2 ** i))
                last_err = Exception(f"HTTP {r.status_code} from {url}")
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(backoff * (2 ** i))
    if last_err:
        raise last_err
    return None