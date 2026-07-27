from __future__ import annotations

import json
import sqlite3
from typing import Any

from .models import Paragraph, TextBlock

OCR_DB_PATH = "ocr_cache.db"


def _conn() -> sqlite3.Connection:
    db = sqlite3.connect(OCR_DB_PATH)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=OFF")
    _init_tables(db)
    return db


def _init_tables(db: sqlite3.Connection) -> None:
    db.execute("""
        CREATE TABLE IF NOT EXISTS ocr_pages (
            page_num INTEGER PRIMARY KEY,
            width REAL NOT NULL,
            height REAL NOT NULL,
            dpi INTEGER NOT NULL DEFAULT 300,
            status TEXT NOT NULL DEFAULT 'ocr_done'
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS ocr_paragraphs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            page_num INTEGER NOT NULL,
            para_idx INTEGER NOT NULL,
            text TEXT NOT NULL,
            translated_text TEXT NOT NULL DEFAULT '',
            line_bboxes_json TEXT NOT NULL,
            FOREIGN KEY (page_num) REFERENCES ocr_pages(page_num)
        )
    """)
    db.commit()


def save_page_ocr(page_num: int, width: float, height: float, dpi: int, paragraphs: list[Paragraph]) -> None:
    db = _conn()
    db.execute(
        "INSERT OR REPLACE INTO ocr_pages (page_num, width, height, dpi, status) VALUES (?, ?, ?, ?, 'ocr_done')",
        (page_num, width, height, dpi),
    )
    db.execute("DELETE FROM ocr_paragraphs WHERE page_num = ?", (page_num,))
    for pi, para in enumerate(paragraphs):
        line_bboxes_data = [[list(b), t, fs] for b, t, fs in para.line_bboxes]
        db.execute(
            "INSERT INTO ocr_paragraphs (page_num, para_idx, text, line_bboxes_json) VALUES (?, ?, ?, ?)",
            (page_num, pi, para.text, json.dumps(line_bboxes_data, ensure_ascii=False)),
        )
    db.commit()


def load_page_ocr(page_num: int) -> tuple[float, float, int, list[Paragraph]] | None:
    db = _conn()
    row = db.execute("SELECT width, height, dpi FROM ocr_pages WHERE page_num = ?", (page_num,)).fetchone()
    if not row:
        return None
    width, height, dpi = row
    rows = db.execute(
        "SELECT para_idx, text, translated_text, line_bboxes_json FROM ocr_paragraphs WHERE page_num = ? ORDER BY para_idx",
        (page_num,),
    ).fetchall()
    paragraphs = []
    for _, text, translated_text, bboxes_json in rows:
        line_bboxes_data = json.loads(bboxes_json)
        line_bboxes = [(tuple(b), t, fs) for b, t, fs in line_bboxes_data]
        para = Paragraph(
            blocks=[TextBlock(bbox=(0, 0, 0, 0), text=text, source="ocr", page_num=page_num)],
            line_bboxes=line_bboxes,
        )
        para.translation = translated_text
        paragraphs.append(para)
    return width, height, dpi, paragraphs


def page_exists(page_num: int) -> bool:
    db = _conn()
    return db.execute("SELECT 1 FROM ocr_pages WHERE page_num = ?", (page_num,)).fetchone() is not None


def load_translate_batch(start_page: int, end_page: int) -> list[tuple[int, float, float, int, list[Paragraph]]]:
    db = _conn()
    rows = db.execute(
        "SELECT page_num, width, height, dpi FROM ocr_pages WHERE page_num BETWEEN ? AND ? AND status IN ('ocr_done', 'translated') ORDER BY page_num",
        (start_page, end_page),
    ).fetchall()
    result = []
    for page_num, width, height, dpi in rows:
        _, _, _, paragraphs = load_page_ocr(page_num)
        result.append((page_num, width, height, dpi, paragraphs))
    return result


def save_translations(page_num: int, paragraphs: list[Paragraph]) -> None:
    db = _conn()
    for pi, para in enumerate(paragraphs):
        db.execute(
            "UPDATE ocr_paragraphs SET translated_text = ? WHERE page_num = ? AND para_idx = ?",
            (para.translation, page_num, pi),
        )
    db.commit()


def mark_page_translated(page_num: int) -> None:
    db = _conn()
    db.execute("UPDATE ocr_pages SET status = 'translated' WHERE page_num = ?", (page_num,))
    db.commit()


def get_max_page() -> int:
    db = _conn()
    row = db.execute("SELECT MAX(page_num) FROM ocr_pages").fetchone()
    return row[0] if row and row[0] else 0


def get_untranslated_pages() -> list[int]:
    db = _conn()
    rows = db.execute("SELECT page_num FROM ocr_pages WHERE status = 'ocr_done' ORDER BY page_num").fetchall()
    return [r[0] for r in rows]
