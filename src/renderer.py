from __future__ import annotations

import os
from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageFont

from .inpaint import inpaint_page
from .models import PageResult, TextCluster

CJK_FONTNAME = "CJK"

_CJK_FONT_PATH: str | None = None
_CJK_FONT_BYTES: bytes | None = None

DEBUG_DRAW_BOXES = True  # Set True to draw colored OCR box boundaries


def _find_cjk_font() -> str | None:
    global _CJK_FONT_PATH
    if _CJK_FONT_PATH:
        return _CJK_FONT_PATH
    candidates = [
        "C:/Windows/Fonts/SimHei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/msgothic.ttc",
        "C:/Windows/Fonts/msmincho.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            _CJK_FONT_PATH = p
            return p
    return None


def _font_size_from_line_height(line_h: float) -> int:
    """Return the best font size that fits within a line of height *line_h* (PDF pts).

    Font size is driven purely by line height – this is the correct physical
    constraint for body text where translated text may be longer than the
    original line box but must remain the same visual size.
    A small margin (15 %) is kept so ascenders/descenders don't clip.
    """
    if line_h < 2:
        return 0
    font_path = _find_cjk_font()
    if not font_path:
        return max(1, int(line_h * 0.72))
    # Binary-search for the largest size whose glyph height fits line_h.
    low, high = 1, int(line_h * 1.2)  # upper bound slightly above line_h
    best = 1
    # Use a representative single CJK character for height measurement.
    _SAMPLE = "国"
    while low <= high:
        mid = (low + high) // 2
        try:
            font = ImageFont.truetype(font_path, mid)
            bbox = font.getbbox(_SAMPLE)
            th = bbox[3] - bbox[1]
        except Exception:
            return max(1, int(line_h * 0.72))
        if th <= line_h * 0.88:  # 12 % margin for descenders
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best





def _optimal_font_size(text: str, box_w: float, box_h: float) -> int:
    """Return font size fitting *text* as a single-line string in the box.

    This is kept for heading / single-element sizing where the text must
    literally fit within the box width (e.g. a short title).
    """
    font_path = _find_cjk_font()
    if not font_path:
        return 0
    if not text.strip() or box_w < 2 or box_h < 2:
        return 0
    low, high = 1, 200
    best = 1
    margin_w = box_w * 0.05
    margin_h = box_h * 0.05
    while low <= high:
        mid = (low + high) // 2
        try:
            font = ImageFont.truetype(font_path, mid)
            bbox = font.getbbox(text)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            return 12
        if tw <= box_w - margin_w and th <= box_h - margin_h:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best





def _get_cjk_font() -> tuple[str, bytes]:
    global _CJK_FONT_BYTES, _CJK_FONT_PATH
    font_path = _find_cjk_font()
    if font_path:
        try:
            font = fitz.Font(fontfile=font_path)
            _CJK_FONT_BYTES = font.buffer
            return os.path.basename(font_path), _CJK_FONT_BYTES
        except Exception:
            pass
    font = fitz.Font("cjk")
    return CJK_FONTNAME, font.buffer


def _pdf_point(coord: float, dpi: int) -> float:
    return coord * 72.0 / dpi


def _compute_usable_rect(bbox: tuple[float, float, float, float], dpi: int, padding_pt: float = 3) -> tuple[float, float, float, float]:
    left = _pdf_point(bbox[0], dpi) + padding_pt
    top = _pdf_point(bbox[1], dpi) + padding_pt
    right = _pdf_point(bbox[2], dpi) - padding_pt
    bottom = _pdf_point(bbox[3], dpi) - padding_pt
    return (left, top, right, bottom)


def _estimate_median_font_size(
    line_bboxes: list[tuple[tuple[float, float, float, float], str, float]],
    dpi: int,
) -> int:
    sizes = []
    for bbox, _, _ in line_bboxes:
        _, ly0, _, ly1 = bbox
        line_h_pt = _pdf_point(ly1 - ly0, dpi)
        if line_h_pt < 2:
            continue
        fs = _font_size_from_line_height(line_h_pt)
        if fs > 0:
            sizes.append(fs)
    if not sizes:
        return 10
    sizes.sort()
    return sizes[len(sizes) // 2]


def _wrap_text_to_width(text: str, width_pt: float, fs: int) -> list[str]:
    if not text.strip() or width_pt < 2:
        return [text] if text.strip() else []

    font_path = _find_cjk_font()
    font = None
    if font_path:
        try:
            font = ImageFont.truetype(font_path, fs)
        except Exception:
            pass

    def measure(t: str) -> float:
        if not t:
            return 0.0
        if font:
            try:
                bbox = font.getbbox(t)
                return bbox[2] - bbox[0]
            except Exception:
                pass
        w = 0.0
        for c in t:
            if '\u4e00' <= c <= '\u9fff' or '\u3000' <= c <= '\u30ff' or '\u3040' <= c <= '\u309f':
                w += fs
            else:
                w += fs * 0.5
        return w

    lines = []
    remaining = text.strip()
    limit = width_pt * 0.95

    while remaining:
        idx = 0
        while idx < len(remaining):
            if measure(remaining[:idx + 1]) <= limit:
                idx += 1
            else:
                break
        if idx == 0:
            idx = 1
        lines.append(remaining[:idx].strip())
        remaining = remaining[idx:].strip()

    return lines


def _fit_text_to_rect(
    text: str,
    usable_rect: tuple[float, float, float, float],
    initial_fs: int,
    line_spacing: float = 1.25,
    min_fs: int = 4,
) -> tuple[int, list[str]]:
    left, top, right, bottom = usable_rect
    width_pt = right - left
    height_pt = bottom - top
    if width_pt < 2 or height_pt < 2:
        return initial_fs, [text] if text.strip() else []

    fs = initial_fs
    while fs >= min_fs:
        lines = _wrap_text_to_width(text, width_pt, fs)
        if not lines:
            return fs, [text] if text.strip() else []
        total_height = len(lines) * fs * line_spacing
        if total_height <= height_pt:
            return fs, lines
        fs -= 1

    fs = min_fs
    lines = _wrap_text_to_width(text, width_pt, fs)
    return fs, lines


def _render_cluster_text(
    page: fitz.Page,
    cluster: TextCluster,
    fontname: str,
    dpi: int,
    inpainted: bool = False,
) -> None:
    if not cluster.translation or not cluster.translation.strip():
        return

    text = cluster.translation.strip()
    usable_rect = _compute_usable_rect(cluster.bbox, dpi)
    left, top, right, bottom = usable_rect
    if right - left < 4 or bottom - top < 4:
        return

    initial_fs = _estimate_median_font_size(cluster.all_line_bboxes, dpi)
    line_spacing = 1.25
    best_fs, lines = _fit_text_to_rect(text, usable_rect, initial_fs, line_spacing)
    if not lines:
        return

    if not inpainted:
        pad = _pdf_point(3, dpi)
        page.draw_rect(
            (left - pad, top - pad, right + pad, bottom + pad),
            color=(1, 1, 1), fill=(1, 1, 1), width=1
        )

    line_height = best_fs * line_spacing
    y_cursor = top + best_fs
    for line in lines:
        if not line.strip():
            y_cursor += line_height
            continue
        page.insert_text((left, y_cursor), line, fontname=fontname, fontsize=best_fs)
        y_cursor += line_height


def _render_translated_paragraph(
    page: fitz.Page,
    para,
    text: str,
    fontname: str,
    dpi: int,
    inpainted: bool = False,
) -> None:
    if not text.strip():
        return

    usable_rect = _compute_usable_rect(para.bbox, dpi)
    left, top, right, bottom = usable_rect
    if right - left < 4 or bottom - top < 4:
        return

    sizes = []
    for bbox, _, _ in para.line_bboxes:
        _, ly0, _, ly1 = bbox
        line_h_pt = _pdf_point(ly1 - ly0, dpi)
        if line_h_pt < 2:
            continue
        fs = _font_size_from_line_height(line_h_pt)
        if fs > 0:
            sizes.append(fs)
    initial_fs = 10
    if sizes:
        sizes.sort()
        initial_fs = sizes[len(sizes) // 2]

    line_spacing = 1.25
    best_fs, lines = _fit_text_to_rect(text, usable_rect, initial_fs, line_spacing)
    if not lines:
        return

    if not inpainted:
        pad = _pdf_point(3, dpi)
        page.draw_rect(
            (left - pad, top - pad, right + pad, bottom + pad),
            color=(1, 1, 1), fill=(1, 1, 1), width=1
        )

    line_height = best_fs * line_spacing
    y_cursor = top + best_fs
    for line in lines:
        if not line.strip():
            y_cursor += line_height
            continue
        page.insert_text((left, y_cursor), line, fontname=fontname, fontsize=best_fs)
        y_cursor += line_height


def _render_page(page: fitz.Page, pr: PageResult, fontname: str, inpainted: bool = False, debug_only: bool = False) -> None:
    dpi = pr.source_dpi

    # ================================================================
    # PASS 0: Render clusters (unified blue boxes) as whole paragraphs.
    # ================================================================
    clustered_ids = set()
    for c in pr.clusters:
        for p in c.paragraphs:
            clustered_ids.add(id(p))

    for cluster in pr.clusters:
        x0, y0, x1, y1 = cluster.bbox
        x_pt = _pdf_point(x0, dpi)
        x1_pt = _pdf_point(x1, dpi)
        y0_pt = _pdf_point(y0, dpi)
        y1_pt = _pdf_point(y1, dpi)
        box_w = x1_pt - x_pt
        erase_h = y1_pt - y0_pt

        if box_w < 4 or erase_h < 4:
            continue

        page.draw_rect((x_pt, y0_pt, x1_pt, y1_pt), color=(0, 0, 1), width=2.0)

        if debug_only:
            continue

        if not cluster.translation:
            continue

        _render_cluster_text(page, cluster, fontname, dpi, inpainted=inpainted)

    # ================================================================
    # PASS 1: Render unclustered paragraphs within their own bbox.
    # ================================================================
    for para in pr.paragraphs:
        if DEBUG_DRAW_BOXES and para.line_bboxes:
            for raw_bbox, _, _ in para.line_bboxes:
                lx0, ly0, lx1, ly1 = raw_bbox
                x_pt = _pdf_point(lx0, dpi)
                x1_pt = _pdf_point(lx1, dpi)
                y0_pt = _pdf_point(ly0, dpi)
                y1_pt = _pdf_point(ly1, dpi)
                if y1_pt - y0_pt < 2 or x1_pt - x_pt < 2:
                    continue
                page.draw_rect((x_pt, y0_pt, x1_pt, y1_pt), color=(1, 0, 0), width=0.5)

        if debug_only:
            continue

        if id(para) in clustered_ids:
            continue
        text = para.translation or ""
        if not text.strip():
            continue
        _render_translated_paragraph(page, para, text, fontname, dpi, inpainted=inpainted)


def _inpaint_and_insert(src_doc: fitz.Document, out_doc: fitz.Document, pr: PageResult) -> fitz.Page:
    src_page = src_doc[pr.page_num - 1]
    dpi = pr.source_dpi
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = src_page.get_pixmap(matrix=mat, alpha=False)

    inpainted_bytes = inpaint_page(
        pix.samples,
        pix.width,
        pix.height,
        pr,
    )

    rect = src_page.rect
    out_doc.new_page(width=rect.width, height=rect.height)
    page = out_doc[-1]
    page.insert_image(rect, stream=inpainted_bytes)
    return page


def _rasterize_and_insert(src_doc: fitz.Document, out_doc: fitz.Document, pr: PageResult) -> fitz.Page:
    src_page = src_doc[pr.page_num - 1]
    dpi = pr.source_dpi
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = src_page.get_pixmap(matrix=mat, alpha=False)
    rect = src_page.rect
    out_doc.new_page(width=rect.width, height=rect.height)
    page = out_doc[-1]
    page.insert_image(rect, stream=pix.tobytes("jpeg"))
    return page


def render_single_page_to_doc(
    src_doc: fitz.Document,
    out_doc: fitz.Document,
    pr: PageResult,
    fontname: str,
    fontbuffer: bytes,
    disable_inpaint: bool = False,
    debug_only: bool = False,
) -> None:
    has_boxes = any(len(p.line_bboxes) > 0 for p in pr.paragraphs)

    if debug_only:
        page = _rasterize_and_insert(src_doc, out_doc, pr)
        page.insert_font(fontname=fontname, fontbuffer=fontbuffer)
        _render_page(page, pr, fontname, inpainted=False, debug_only=True)
        return

    if not disable_inpaint and has_boxes:
        page = _inpaint_and_insert(src_doc, out_doc, pr)
        page.insert_font(fontname=fontname, fontbuffer=fontbuffer)
        _render_page(page, pr, fontname, inpainted=True)
    elif has_boxes:
        page = _rasterize_and_insert(src_doc, out_doc, pr)
        page.insert_font(fontname=fontname, fontbuffer=fontbuffer)
        _render_page(page, pr, fontname, inpainted=False)
    else:
        src_idx = pr.page_num - 1
        out_doc.insert_pdf(src_doc, from_page=src_idx, to_page=src_idx)
        page = out_doc[-1]
        page.insert_font(fontname=fontname, fontbuffer=fontbuffer)
        _render_page(page, pr, fontname, inpainted=True)


def render_pdf(
    input_path: str,
    output_path: str,
    pages: list[PageResult],
    disable_inpaint: bool = False,
    debug_only: bool = False,
) -> str:
    src_doc = fitz.open(input_path)
    out_doc = fitz.open()

    fontname, fontbuffer = _get_cjk_font()

    for pr in pages:
        render_single_page_to_doc(src_doc, out_doc, pr, fontname, fontbuffer, disable_inpaint, debug_only)

    out_doc.save(output_path, garbage=4, deflate=True)
    out_doc.close()
    src_doc.close()
    return output_path
