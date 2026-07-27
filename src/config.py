from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class Config:
    engine: str = "ollama"
    model: str = "qwen3:8b"
    openrouter_api_key: str = ""
    openrouter_model: str = "openai/gpt-4o-mini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    ollama_url: str = "http://localhost:11434"
    batch_pages: int = 10
    translate_batch_pages: int = 3
    chunk_min_tokens: int = 400
    chunk_max_tokens: int = 600
    chunk_max_items: int = 15
    cache_db: str = "translation_cache.db"
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    no_ocr: bool = False
    use_gpu: bool = True
    ocr_engine: str = "both"
    ocr_lang: str = "japan"
    ocr_confidence: float = 0.1
    text_threshold: float = 0.7
    low_text: float = 0.4
    min_size: int = 10
    canvas_size: int = 4096
    output_suffix: str = "_translated"
    ocr_only: bool = False
    stage: str = "full"
    extract_dpi: int = 300
    cluster_algorithm: str = "simple"
    line_spacing_ratio: float = 1.5
    column_gap: float = 50.0
    threads: dict = field(default_factory=lambda: {
        "ocr": 2,
        "translator": 1,
    })

    @classmethod
    def load(cls, path: str) -> Config:
        if not os.path.exists(path):
            return cls()
        if yaml is None:
            raise ImportError("PyYAML is required to load config files")
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return cls()
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def merge_cli(self, args) -> None:
        for key in ("engine", "model", "page_start", "page_end", "no_ocr", "batch_pages", "translate_batch_pages", "ocr_only", "stage"):
            val = getattr(args, key, None)
            if val is not None:
                setattr(self, key, val)
