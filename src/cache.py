from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path


class TranslationCache:
    def __init__(self, db_path: str = "translation_cache.db"):
        self.db_path = db_path
        self._local = threading.local()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.execute(
                "CREATE TABLE IF NOT EXISTS cache ("
                "  source_hash TEXT PRIMARY KEY,"
                "  source_text TEXT NOT NULL,"
                "  translated_text TEXT NOT NULL"
                ")"
            )
        return self._local.conn

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> str | None:
        h = self._hash(text)
        cur = self._conn().execute(
            "SELECT translated_text FROM cache WHERE source_hash = ?", (h,)
        )
        row = cur.fetchone()
        return row[0] if row else None

    def put(self, source: str, translated: str) -> None:
        h = self._hash(source)
        self._conn().execute(
            "INSERT OR REPLACE INTO cache (source_hash, source_text, translated_text) VALUES (?, ?, ?)",
            (h, source, translated),
        )
        self._conn().commit()

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
