from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests

from .cache import TranslationCache


SYSTEM_PROMPT = (
    """You are a translation script. Translate Japanese text into Simplified Chinese.

[CONSTRAINTS]
- You must translate Japanese Kanji into Chinese equivalents (e.g., 手紙 -> 信).
- Do not copy-paste the input text.
- Keep English words, numbers, units, URLs, and part numbers unchanged.
- Lines are separated by ' ||| '. CRITICAL: each ' ||| ' separates one line. Output the EXACT SAME number of lines as the input — count the ' ||| ' separators and match the count. Do NOT merge lines.
- Output ONLY the translated result. No explanations.
"""
)

PARAGRAPH_PROMPT = (
    "Translate the following Japanese text into Simplified Chinese.\n"
    "CRITICAL: The input is a single paragraph. Translate it as one coherent paragraph.\n"
    "Do NOT add '|||' or any other line separators to the output.\n"
    "Keep English words, numbers, units, URLs, and part numbers unchanged.\n"
    "Output ONLY the translated result. No explanations.\n"
)

SYSTEM_PROMPT_JSON = """You are a translation AI. Translate Japanese text to Simplified Chinese.

INPUT: JSON object with "items" array. Each item has "id" (integer) and "text" (Japanese string).
OUTPUT: JSON object with "items" array. Each item has "id" (same as input) and "text" (Chinese translation).

CRITICAL RULES:
- Return EXACTLY the same number of items as the input.
- Every input ID must appear exactly once in the output.
- Do not omit, merge, or split any items.
- Translate Japanese Kanji to Chinese equivalents (e.g., 手紙 -> 信).
- Keep English words, numbers, units, URLs, and part numbers unchanged.
- Output ONLY valid JSON. No explanations.
"""

DEFAULT_TIMEOUT = 600


def _count_tokens(text: str) -> int:
    approx = len(text) / 2.5
    return max(1, int(approx))


def build_chunks(paragraphs: list, min_tokens: int = 800, max_tokens: int = 1200, max_items: int = 15) -> list[list[int]]:
    chunks: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0

    for i, para in enumerate(paragraphs):
        pt = _count_tokens(para.text)
        if current and (current_tokens + pt > max_tokens or len(current) >= max_items):
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(i)
        current_tokens += pt

    if current:
        if current_tokens < min_tokens and chunks and len(chunks[-1]) + len(current) <= max_items:
            chunks[-1].extend(current)
        else:
            chunks.append(current)

    return chunks


class _BaseTranslator:
    def translate(self, text: str) -> str:
        raise NotImplementedError

    def translate_paragraph(self, text: str) -> str:
        cache = getattr(self, 'cache', None)
        cached = cache.get(text) if cache else None
        if cached is not None and not _is_identity_failure(text, cached):
            return cached
        result = self._call_paragraph_api(text)
        if cache and not _is_identity_failure(text, result):
            cache.put(text, result)
        return result

    def _call_paragraph_api(self, text: str) -> str:
        raise NotImplementedError

    def _translate_batch(self, texts: list[str]) -> list[str]:
        if not texts:
            return []

        results: dict[int, str] = {}
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, t in enumerate(texts):
            cached = self.cache.get(t) if self.cache else None
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(t)

        if not uncached_texts:
            return [results.get(i, "") for i in range(len(texts))]

        if len(uncached_texts) == 1:
            idx = uncached_indices[0]
            translated = self.translate(uncached_texts[0])
            results[idx] = translated
            return [results.get(i, "") for i in range(len(texts))]

        numbered = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(uncached_texts))
        prompt = (
            "Translate the following Japanese paragraphs to Simplified Chinese. "
            "Output each translation prefixed by [1], [2], etc. "
            "No explanations, no notes.\n\n"
            f"{numbered}"
        )

        raw = self._call_api(prompt)

        clean = re.sub(r'\*\*\[(\d+)\]\*\*', r'[\1]', raw)
        clean = re.sub(r'^\*\*|\*\*$', '', clean)

        for m in re.finditer(r'\[(\d+)\]\s*(.+?)(?=\n\[\d+\]|\Z)', clean, re.DOTALL):
            idx = int(m.group(1)) - 1
            t = m.group(2).strip().lstrip("*").strip()
            if t and not re.search(r'\[\d+\]', t) and 0 <= idx < len(uncached_texts):
                orig = uncached_indices[idx]
                results[orig] = t
                src_text = uncached_texts[idx]
                if self.cache and not _is_identity_failure(src_text, t):
                    self.cache.put(src_text, t)

        for i, src in enumerate(uncached_texts):
            orig = uncached_indices[i]
            if orig not in results or not results[orig]:
                results[orig] = ""

        return [results.get(i, "") for i in range(len(texts))]

    def translate_json_batch(self, items: list[tuple[int, str]]) -> dict[int, str]:
        """Translate items using JSON structured I/O.
        items: list of (id, text) pairs.
        Returns: dict of {id: translated_text} (may be partial on failure).
        """
        import json as _json
        if not items:
            return {}

        # Check cache
        result: dict[int, str] = {}
        uncached: list[tuple[int, str]] = []
        src_by_id: dict[int, str] = {}
        for id_, text in items:
            src_by_id[id_] = text
            cached = self.cache.get(text) if self.cache else None
            if cached is not None and not _is_identity_failure(text, cached):
                result[id_] = cached
            else:
                uncached.append((id_, text))

        if not uncached:
            return result

        # Build JSON input
        input_data = {
            "source": "ja",
            "target": "zh-CN",
            "items": [{"id": id_, "text": text} for id_, text in uncached]
        }
        input_str = _json.dumps(input_data, ensure_ascii=False)

        raw = self._call_json_api(input_str, len(uncached))

        # Parse JSON response
        try:
            output_data = _json.loads(raw)
            for item in output_data.get("items", []):
                id_ = item.get("id")
                trans = item.get("text", "").strip()
                if id_ is not None and trans:
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
    """True if text contains hiragana or katakana (unambiguously Japanese)."""
    return any('\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' for c in text)


def _is_identity_failure(text: str, translation: str) -> bool:
    """True if translation identical to input AND input appears Japanese (hiragana, katakana, or Kanji)."""
    if translation.strip() != text.strip():
        return False
    if _has_japanese(text):
        return True
    # Pure Kanji or mixed CJK — unchanged text is likely a translation failure
    return bool(re.findall(r'[\u4e00-\u9fff]', text))


def _retry_translate(translator: _BaseTranslator, text: str, max_retries: int = 2) -> str:
    import warnings
    import time
    for attempt in range(max_retries):
        try:
            t = translator.translate(text)
            if t.strip() and not _is_identity_failure(text, t):
                return t
            if not t.strip():
                warnings.warn(f"  [WARN] translation returned empty, retry {attempt + 1}/{max_retries}")
            else:
                warnings.warn(f"  [WARN] translation returned unchanged for Japanese text, retry {attempt + 1}/{max_retries}")
        except Exception as e:
            warnings.warn(f"  [WARN] translation failed (attempt {attempt + 1}/{max_retries}): {e}")
        if attempt < max_retries - 1:
            time.sleep(2)
    return text


def translate_batches_concurrent(
    translator: _BaseTranslator,
    all_paragraphs: list,
    chunks: list[list[int]],
    max_workers: int = 2,
) -> None:
    import warnings

    retry_cache: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        def _translate_chunk(indices: list[int]) -> None:
            texts = [all_paragraphs[i].text for i in indices]
            try:
                translations = translator.translate_batch(texts)
                for idx, trans in zip(indices, translations):
                    src = all_paragraphs[idx].text
                    if trans.strip() and not _is_identity_failure(src, trans):
                        all_paragraphs[idx].translation = trans
                    else:
                        if trans.strip() and _is_identity_failure(src, trans):
                            warnings.warn(f"  [WARN] batch returned unchanged for Japanese text, scheduling retry")
                        # Keep text as-is; the pipeline identity check will retry the full batch
                        all_paragraphs[idx].translation = src
                        retry_cache[src] = src
            except Exception as e:
                warnings.warn(f"  [WARN] batch translation failed for {len(indices)} paragraphs: {e}")
                for idx in indices:
                    src = all_paragraphs[idx].text
                    all_paragraphs[idx].translation = ""
                    retry_cache[src] = src

        list(pool.map(_translate_chunk, chunks))

    # Retry failed batches in bulk
    if retry_cache:
        srcs = list(retry_cache.keys())
        retry_pages = set()
        for p in all_paragraphs:
            if p.text in retry_cache:
                retry_pages.add(p.blocks[0].page_num if p.blocks else 0)
        warnings.warn(f"  [RETRY] retrying {len(srcs)} paragraphs across pages {sorted(retry_pages)}")
        try:
            results = translator.translate_batch(srcs)
            for src, trans in zip(srcs, results):
                if trans.strip() and not _is_identity_failure(src, trans):
                    for p in all_paragraphs:
                        if p.text == src:
                            p.translation = trans
                            break
        except Exception as e:
            warnings.warn(f"  [WARN] bulk retry also failed: {e}")


class OllamaTranslator(_BaseTranslator):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen3:8b", cache: Optional[TranslationCache] = None, timeout: int = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.cache = cache
        self.timeout = timeout

    def _call_api(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
            "stream": False,
            "think": False,
            "options": {"temperature": 0.1, "num_gpu": 100},
        }
        resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def _call_paragraph_api(self, text: str) -> str:
        payload = {
            "model": self.model,
            "prompt": f"{PARAGRAPH_PROMPT}\n\n{text}",
            "stream": False,
            "think": False,
            "options": {"temperature": 0.1, "num_gpu": 100},
        }
        resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def _call_json_api(self, input_json: str, n_items: int) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_JSON},
                {"role": "user", "content": f"There are exactly {n_items} input records.\n\nReturn exactly {n_items} output records.\n\nDo not omit, merge, or split.\n\nEvery input id must appear exactly once.\n\n{input_json}"},
            ],
            "format": "json",
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_gpu": 100},
        }
        resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def translate(self, text: str) -> str:
        cached = self.cache.get(text) if self.cache else None
        if cached is not None and not _is_identity_failure(text, cached):
            return cached

        result = self._call_api(text)

        if self.cache and not _is_identity_failure(text, result):
            self.cache.put(text, result)
        return result

    def translate_batch(self, texts: list[str]) -> list[str]:
        return self._translate_batch(texts)


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

    def _call_api(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
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
                {"role": "user", "content": f"There are exactly {n_items} input records.\n\nReturn exactly {n_items} output records.\n\nDo not omit, merge, or split.\n\nEvery input id must appear exactly once.\n\n{input_json}"},
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

    def translate(self, text: str) -> str:
        cached = self.cache.get(text) if self.cache else None
        if cached is not None and not _is_identity_failure(text, cached):
            return cached

        result = self._call_api(text)

        if self.cache and not _is_identity_failure(text, result):
            self.cache.put(text, result)
        return result

    def translate_batch(self, texts: list[str]) -> list[str]:
        return self._translate_batch(texts)


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

    def _call_api(self, prompt: str) -> str:
        from google import genai
        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,
            ),
        )
        return response.text.strip()

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
            contents=f"There are exactly {n_items} input records.\n\nReturn exactly {n_items} output records.\n\nDo not omit, merge, or split.\n\nEvery input id must appear exactly once.\n\n{input_json}",
            config=genai.types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT_JSON,
                temperature=0,
                response_mime_type="application/json",
            ),
        )
        return response.text.strip()

    def translate(self, text: str) -> str:
        cached = self.cache.get(text) if self.cache else None
        if cached is not None and not _is_identity_failure(text, cached):
            return cached

        result = self._call_api(text)

        if self.cache and not _is_identity_failure(text, result):
            self.cache.put(text, result)
        return result

    def translate_batch(self, texts: list[str]) -> list[str]:
        return self._translate_batch(texts)


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
