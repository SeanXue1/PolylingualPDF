from __future__ import annotations

import re
from typing import Optional

import requests

from .cache import TranslationCache


PARAGRAPH_PROMPT = (
    "Translate the following Japanese text into Simplified Chinese.\n"
    "CRITICAL: The input is a single paragraph. Translate it as one coherent paragraph.\n"
    "Keep English words, numbers, units, URLs, and part numbers unchanged.\n"
    "Output ONLY the translated result. No explanations.\n"
)

SYSTEM_PROMPT_JSON = """You are a translation AI. Translate Japanese text to Simplified Chinese.

INPUT: JSON object with "items" array. Each item has "id" (integer) and "text" (Japanese string).
OUTPUT: JSON object with "items" array. Each item has "id" (same as input) and "text" (Chinese translation).

CRITICAL RULES:
- Translate only the text values in the JSON input. Do not translate or echo the instruction text.
- Return EXACTLY the same number of items as the input.
- Every input ID must appear exactly once in the output.
- Do not omit, merge, or split any items.
- Translate Japanese Kanji to Chinese equivalents.
- Keep English words, numbers, units, URLs, and part numbers unchanged.
- Output ONLY valid JSON. No explanations.
"""

DEFAULT_TIMEOUT = 600


class _BaseTranslator:
    def translate_paragraph(self, text: str) -> str:
        cache = getattr(self, "cache", None)
        cached = cache.get(text) if cache else None
        if cached is not None and not _is_identity_failure(text, cached) and not _looks_like_prompt_echo(cached):
            return cached

        result = self.translate_json_batch([(0, text)]).get(0, "")
        if not result or _looks_like_prompt_echo(result):
            result = self._call_paragraph_api(text)

        if cache and not _is_identity_failure(text, result) and not _looks_like_prompt_echo(result):
            cache.put(text, result)
        return result

    def _call_paragraph_api(self, text: str) -> str:
        raise NotImplementedError

    def translate_json_batch(self, items: list[tuple[int, str]]) -> dict[int, str]:
        """Translate items using JSON structured I/O."""
        import json as _json

        if not items:
            return {}

        result: dict[int, str] = {}
        uncached: list[tuple[int, str]] = []
        src_by_id: dict[int, str] = {}
        for id_, text in items:
            src_by_id[id_] = text
            cached = self.cache.get(text) if self.cache else None
            if cached is not None and not _is_identity_failure(text, cached) and not _looks_like_prompt_echo(cached):
                result[id_] = cached
            else:
                uncached.append((id_, text))

        if not uncached:
            return result

        input_data = {
            "source": "ja",
            "target": "zh-CN",
            "items": [{"id": id_, "text": text} for id_, text in uncached],
        }
        input_str = _json.dumps(input_data, ensure_ascii=False)
        raw = self._call_json_api(input_str, len(uncached))

        try:
            output_data = _json.loads(raw)
            for item in output_data.get("items", []):
                id_ = item.get("id")
                trans = item.get("text", "").strip()
                if id_ is None or not trans:
                    continue
                if _looks_like_prompt_echo(trans):
                    continue
                result[id_] = trans
                src = src_by_id.get(id_, "")
                if self.cache and not _is_identity_failure(src, trans):
                    self.cache.put(src, trans)
        except _json.JSONDecodeError:
            pass

        return result

    def _call_json_api(self, input_json: str, n_items: int) -> str:
        raise NotImplementedError


def _has_japanese(text: str) -> bool:
    return any("\u3040" <= c <= "\u309f" or "\u30a0" <= c <= "\u30ff" for c in text)


def _looks_like_prompt_echo(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    lower = t.lower()
    markers = (
        "translate the following",
        "output only",
        "json object",
        "items array",
        "simplified chinese",
        "single paragraph",
        "do not explain",
        "不要解释",
        "翻译以下",
        "简体中文",
        "只输出",
        "单个段落",
    )
    return any(marker in lower for marker in markers)


def _is_identity_failure(text: str, translation: str) -> bool:
    """True if translation identical to input AND input appears Japanese."""
    if translation.strip() != text.strip():
        return False
    if _has_japanese(text):
        return True
    return bool(re.findall(r"[\u4e00-\u9fff]", text))


class OllamaTranslator(_BaseTranslator):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:8b",
        cache: Optional[TranslationCache] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.cache = cache
        self.timeout = timeout

    def _call_paragraph_api(self, text: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": PARAGRAPH_PROMPT},
                {"role": "user", "content": text},
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": 0.1, "num_gpu": 100},
        }
        resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    def _call_json_api(self, input_json: str, n_items: int) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_JSON},
                {
                    "role": "user",
                    "content": (
                        f"There are exactly {n_items} input records.\n\n"
                        f"Return exactly {n_items} output records.\n\n"
                        "Do not omit, merge, or split.\n\n"
                        "Every input id must appear exactly once.\n\n"
                        f"{input_json}"
                    ),
                },
            ],
            "format": "json",
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_gpu": 100},
        }
        resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["message"]["content"]


class OpenRouterTranslator(_BaseTranslator):
    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-4o-mini",
        cache: Optional[TranslationCache] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.api_key = api_key
        self.model = model
        self.cache = cache
        self.timeout = timeout

    def _call_paragraph_api(self, text: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": PARAGRAPH_PROMPT},
                {"role": "user", "content": text},
            ],
            "temperature": 0.1,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def _call_json_api(self, input_json: str, n_items: int) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_JSON},
                {
                    "role": "user",
                    "content": (
                        f"There are exactly {n_items} input records.\n\n"
                        f"Return exactly {n_items} output records.\n\n"
                        "Do not omit, merge, or split.\n\n"
                        "Every input id must appear exactly once.\n\n"
                        f"{input_json}"
                    ),
                },
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


class GeminiTranslator(_BaseTranslator):
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        cache: Optional[TranslationCache] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.api_key = api_key
        self.model = model
        self.cache = cache
        self.timeout = timeout

    def _call_paragraph_api(self, text: str) -> str:
        from google import genai

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=self.model,
            contents=text,
            config=genai.types.GenerateContentConfig(
                system_instruction=PARAGRAPH_PROMPT,
                temperature=0.1,
            ),
        )
        return response.text.strip()

    def _call_json_api(self, input_json: str, n_items: int) -> str:
        from google import genai

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=self.model,
            contents=(
                f"There are exactly {n_items} input records.\n\n"
                f"Return exactly {n_items} output records.\n\n"
                "Do not omit, merge, or split.\n\n"
                "Every input id must appear exactly once.\n\n"
                f"{input_json}"
            ),
            config=genai.types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT_JSON,
                temperature=0,
                response_mime_type="application/json",
            ),
        )
        return response.text.strip()


def create_translator(config, cache: Optional[TranslationCache] = None):
    if config.engine == "openrouter":
        return OpenRouterTranslator(
            api_key=config.openrouter_api_key,
            model=config.openrouter_model,
            cache=cache,
        )
    if config.engine == "gemini":
        return GeminiTranslator(
            api_key=config.gemini_api_key,
            model=config.gemini_model,
            cache=cache,
        )
    return OllamaTranslator(
        base_url=config.ollama_url,
        model=config.model,
        cache=cache,
    )
