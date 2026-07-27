from __future__ import annotations

import os
from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .inpaint import inpaint_page
from .models import PageResult

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


def _text_fits_width(text: str, box_w: float, fs: int) -> bool:
    if not text.strip():
        return True
    font_path = _find_cjk_font()
    if not font_path:
        w = 0.0
        for c in text:
            if '\u4e00' <= c <= '\u9fff' or '\u3000' <= c <= '\u30ff' or '\u3040' <= c <= '\u309f':
                w += fs
            else:
                w += fs * 0.5
        return w <= box_w * 0.95
    try:
        font = ImageFont.truetype(font_path, fs)
        bbox = font.getbbox(text)
        tw = bbox[2] - bbox[0]
        return tw <= box_w * 0.95
    except Exception:
        return True


def _all_lines_fit(full_text: str, box_widths: list[float], fs: int) -> bool:
    wrapped = _wrap_text_to_boxes(full_text, box_widths, fs)
    for i, line in enumerate(wrapped):
        if not line.strip():
            continue
        bw = box_widths[i] if i < len(box_widths) else box_widths[-1]
        if not _text_fits_width(line, bw, fs):
            return False
    return True


def _find_best_fs(full_text: str, box_widths: list[float], start_fs: int, min_fs: int = 4) -> int:
    for fs in range(start_fs, min_fs - 1, -1):
        if _all_lines_fit(full_text, box_widths, fs):
            return fs
    return min_fs


def _wrap_text_to_boxes(full_text: str, box_widths: list[float], fs: int) -> list[str]:
    font_path = _find_cjk_font()
    font = None
    if font_path:
        try:
            font = ImageFont.truetype(font_path, fs)
        except Exception:
            pass

    def get_width(text: str) -> float:
        if not text:
            return 0.0
        if font:
            try:
                bbox = font.getbbox(text)
                return bbox[2] - bbox[0]
            except Exception:
                pass
        w = 0.0
        for c in text:
            if '\u4e00' <= c <= '\u9fff' or '\u3000' <= c <= '\u30ff' or '\u3040' <= c <= '\u309f':
                w += fs
            else:
                w += fs * 0.5
        return w

    lines = []
    remaining = full_text.strip()
    
    for bw in box_widths:
        if not remaining:
            lines.append("")
            continue
        
        limit = bw * 0.95
        idx = 0
        
        while idx < len(remaining):
            test_str = remaining[:idx + 1]
            tw = get_width(test_str)
            if tw <= limit:
                idx += 1
            else:
                break
        
        if 0 < idx < len(remaining):
            if remaining[idx-1].isalnum() and remaining[idx].isalnum():
                space_idx = idx - 1
                while space_idx > 0 and remaining[space_idx].isalnum():
                    space_idx -= 1
                if space_idx > 0 and not remaining[space_idx].isalnum():
                    idx = space_idx + 1
        
        if idx == 0:
            idx = 1
            
        lines.append(remaining[:idx].strip())
        remaining = remaining[idx:].strip()
        
    if remaining and lines:
        lines[-1] = (lines[-1] + " " + remaining).strip()
        
    return lines


def _join_translated_lines(lines: list[str]) -> str:
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return ""
    full = " ".join(non_empty)
    if any('\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' for c in full):
        return "".join(non_empty).strip()
    return full.strip()



def _max_font_size_for_width(text: str, box_w: float, start_fs: int) -> int:
    """Return the largest font size (<= *start_fs*) where *text* fits in *box_w*.

    Uses the PIL glyph renderer for accurate per-font metrics.
    Returns *start_fs* unchanged if *text* is empty or the font is unavailable.
    """
    if not text.strip() or box_w < 2 or start_fs <= 1:
        return start_fs
    font_path = _find_cjk_font()
    if not font_path:
        return start_fs
    # Quick check: does it already fit?
    try:
        chk = ImageFont.truetype(font_path, start_fs)
        tw = chk.getbbox(text)[2] - chk.getbbox(text)[0]
        if tw <= box_w * 0.95:
            return start_fs
    except Exception:
        return start_fs
    # Binary search downward from start_fs.
    low, high = 1, start_fs - 1
    best = 1
    while low <= high:
        mid = (low + high) // 2
        try:
            font = ImageFont.truetype(font_path, mid)
            tw = font.getbbox(text)[2] - font.getbbox(text)[0]
        except Exception:
            break
        if tw <= box_w * 0.95:
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


def _consistent_font_size_for_paragraph(
    render_lines: list[tuple[tuple[float, float, float, float], str, float]],
    dpi: int,
) -> int:
    """Return a consistent font size for all lines in a paragraph.

    Font size is derived from the *line height* of each line (the correct
    physical constraint), not from fitting the translated text string width
    into the box.  We take the median height-based size so that one
    unusually short or tall line does not skew the whole paragraph.
    """
    sizes: list[int] = []
    for raw_bbox, line_text, _ in render_lines:
        if not line_text.strip():
            continue
        lx0, ly0, lx1, ly1 = raw_bbox
        y0_pt = _pdf_point(ly0, dpi)
        y1_pt = _pdf_point(ly1, dpi)
        line_h = y1_pt - y0_pt
        if line_h < 2:
            continue
        fs = _font_size_from_line_height(line_h)
        if fs > 0:
            sizes.append(fs)
    if not sizes:
        return 0
    sizes.sort()
    # Use the median so outlier lines (very short/tall) don't dominate.
    return sizes[len(sizes) // 2]


def _compute_page_body_metrics(
    paragraphs: list,
    dpi: int,
) -> tuple[int, float]:
    """Compute the page-wide body font size and reference body line height.

    Scans all multi-line paragraphs (>= 2 rendered lines) to determine:
    - ``body_font``: median height-based font size across all body paragraph lines.
    - ``body_line_h``: 75th-percentile line height (PDF points) of body lines.

    The body_line_h is used downstream to decide whether a *single-line*
    paragraph is an orphaned body sentence (same line height as body) or a
    genuine heading (significantly taller line).

    Returns (body_font, body_line_h).  Both are 0 when no multi-line
    paragraphs with translations exist.
    """
    all_line_heights: list[float] = []

    for para in paragraphs:
        if not para.translated_lines or not para.line_bboxes:
            continue
        render_lines = [
            (bbox, tline, fs)
            for (bbox, _, fs), tline in zip(para.line_bboxes, para.translated_lines)
            if tline.strip()
        ]
        if len(render_lines) < 2:
            continue
        for bbox, _, _ in render_lines:
            lx0, ly0, lx1, ly1 = bbox
            lh = _pdf_point(ly1 - ly0, dpi)
            if lh > 0:
                all_line_heights.append(lh)

    if not all_line_heights:
        return 0, 0.0

    all_line_heights.sort()
    idx75 = int(len(all_line_heights) * 0.75)
    body_line_h = all_line_heights[min(idx75, len(all_line_heights) - 1)]

    # Derive body_font from the median body line height so that font size
    # matches the physical line slots on the page.
    idx50 = len(all_line_heights) // 2
    median_line_h = all_line_heights[idx50]
    body_font = _font_size_from_line_height(median_line_h)

    return body_font, body_line_h


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


def _build_render_lines(
    para, dpi: int,
) -> list[tuple[tuple[float, float, float, float], str, float]]:
    """Build the list of (bbox, translated_text, font_size) for a paragraph."""
    render_lines: list[tuple[tuple[float, float, float, float], str, float]] = []
    if para.translated_lines and para.line_bboxes:
        for (bbox_orig, _, font_sz), tline in zip(para.line_bboxes, para.translated_lines):
            render_lines.append((bbox_orig, tline, font_sz))
    return render_lines


def _is_body_paragraph(
    render_lines: list[tuple[tuple[float, float, float, float], str, float]],
    page_body_font: int,
    body_line_h: float,
    heading_scale: float,
    dpi: int,
) -> bool:
    """Return True if the paragraph should be treated as body text."""
    if page_body_font <= 0 or body_line_h <= 0:
        return False
    para_heights: list[float] = []
    for bbox, _, _ in render_lines:
        lx0, ly0, lx1, ly1 = bbox
        lh = _pdf_point(ly1 - ly0, dpi)
        if lh > 0:
            para_heights.append(lh)
    if para_heights:
        avg_h = sum(para_heights) / len(para_heights)
        if avg_h > body_line_h * heading_scale:
            return False
    return True


def _para_box_widths(
    render_lines: list[tuple[tuple[float, float, float, float], str, float]],
    dpi: int,
) -> list[float]:
    """Return the PDF-point widths of each line box."""
    widths: list[float] = []
    for raw_bbox, _, _ in render_lines:
        lx0, _, lx1, _ = raw_bbox
        bw = _pdf_point(lx1 - lx0, dpi)
        widths.append(max(2.0, bw))
    return widths


def _render_page(page: fitz.Page, pr: PageResult, fontname: str, inpainted: bool = False, debug_only: bool = False) -> None:
    dpi = pr.source_dpi

    page_body_font, body_line_h = _compute_page_body_metrics(pr.paragraphs, dpi)
    HEADING_SCALE = 1.8

    # ================================================================
    # PASS 1: Find the unified body font size across ALL body paragraphs.
    #         This is the largest font where every body paragraph's text
    #         fits when wrapped across its line boxes.
    # ================================================================
    body_para_data: list[tuple[str, list[float]]] = []  # (full_text, box_widths)
    for para in pr.paragraphs:
        rl = _build_render_lines(para, dpi)
        if not rl:
            continue
        if not _is_body_paragraph(rl, page_body_font, body_line_h, HEADING_SCALE, dpi):
            continue
        bw = _para_box_widths(rl, dpi)
        ft = _join_translated_lines([lt for _, lt, _ in rl])
        if ft.strip():
            body_para_data.append((ft, bw))

    if page_body_font > 0:
        unified_body_fs = page_body_font
    else:
        unified_body_fs = 10

    # ================================================================
    # PASS 0: Render clusters (unified blue boxes).
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

        # Draw blue debug border (always, to show cluster boundary)
        page.draw_rect((x_pt, y0_pt, x1_pt, y1_pt), color=(0, 0, 1), width=2.0)

        if debug_only:
            continue

        if not cluster.translation:
            continue

        # Erase original text
        page.draw_rect((x_pt, y0_pt, x1_pt, y1_pt), color=(1, 1, 1), fill=(1, 1, 1), width=1)

        # Compute average original line height for baseline font size
        all_heights = []
        for bbox, _, _ in cluster.all_line_bboxes:
            lh = _pdf_point(bbox[3] - bbox[1], dpi)
            if lh > 0:
                all_heights.append(lh)
        avg_line_h = sum(all_heights) / len(all_heights) if all_heights else erase_h * 0.1
        start_fs = max(4, _font_size_from_line_height(avg_line_h))

        # Max lines: use original line count + 30% headroom for longer translations
        # (NOT box_h / line_h, which includes inter-line gaps and inflates the count)
        n_orig = len(cluster.all_line_bboxes)
        max_lines = max(1, int(n_orig * 1.3))

        # Find largest font where wrapped text fits in max_lines
        text = cluster.translation
        best_fs = start_fs
        final_wrapped = [text]
        for fs in range(start_fs, 3, -1):
            wrapped = _wrap_text_to_boxes(text, [box_w] * max_lines, fs)
            non_empty = [l for l in wrapped if l.strip()]
            if len(non_empty) <= max_lines:
                best_fs = fs
                final_wrapped = wrapped
                break

        # Render lines vertically centered in the erase box
        total_text_h = len(final_wrapped) * best_fs * 1.2
        render_y = y0_pt + (erase_h - total_text_h) / 2 + best_fs
        for line in final_wrapped:
            if line.strip():
                page.insert_text((x_pt, render_y), line, fontname=fontname, fontsize=best_fs)
            render_y += best_fs * 1.2

    # ================================================================
    # PASS 2: Draw white boxes and render translated text.
    # ================================================================
    font_path = _find_cjk_font()

    for para in pr.paragraphs:
        # --- Debug: draw red box boundaries (ALL paragraphs, clustered or not) ---
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
        # --- Draw white boxes to erase original text ---
        if para.line_bboxes:
            pad = _pdf_point(3, dpi) if not inpainted else 0
            for raw_bbox, _, _ in para.line_bboxes:
                lx0, ly0, lx1, ly1 = raw_bbox
                x_pt = _pdf_point(lx0, dpi)
                x1_pt = _pdf_point(lx1, dpi)
                y0_pt = _pdf_point(ly0, dpi)
                y1_pt = _pdf_point(ly1, dpi)
                if y1_pt - y0_pt < 2:
                    continue
                if not inpainted:
                    page.draw_rect((x_pt - pad, y0_pt - pad, x1_pt + pad, y1_pt + pad), color=(1, 1, 1), fill=(1, 1, 1), width=1)
        elif not inpainted:
            raw_x0, raw_y0, raw_x1, raw_y1 = para.bbox
            x0 = _pdf_point(raw_x0, dpi)
            x1 = _pdf_point(raw_x1, dpi)
            y0_pt = _pdf_point(raw_y0, dpi)
            ph = _pdf_point(raw_y1, dpi) - _pdf_point(raw_y0, dpi)
            if x1 - x0 >= 2 and ph >= 2:
                page.draw_rect((x0, y0_pt, x1, y0_pt + ph), color=(1, 1, 1), fill=(1, 1, 1), width=1)

        # --- Build render_lines ---
        render_lines = _build_render_lines(para, dpi)

        if not render_lines:
            # Fallback for paragraphs with only para.translation (no line_bboxes)
            if para.translation:
                text = para.translation
                raw_x0, raw_y0, raw_x1, raw_y1 = para.bbox
                x0 = _pdf_point(raw_x0, dpi)
                x1 = _pdf_point(raw_x1, dpi)
                pw = x1 - x0
                ph = _pdf_point(raw_y1, dpi) - _pdf_point(raw_y0, dpi)
                if pw < 2 or ph < 2:
                    continue
                fontsize = max(ph * 0.25, 6)
                max_chars = max(1, int(pw / (fontsize * 0.5)))
                wrapped = []
                remaining = text
                while remaining:
                    if len(remaining) <= max_chars:
                        wrapped.append(remaining)
                        break
                    wrapped.append(remaining[:max_chars])
                    remaining = remaining[max_chars:]
                y0_pt = _pdf_point(raw_y0, dpi)
                for i, line in enumerate(wrapped):
                    ty = y0_pt + fontsize + i * (fontsize * 1.2)
                    if ty + fontsize > y0_pt + ph:
                        break
                    page.insert_text((x0, ty), line, fontname=fontname, fontsize=fontsize)
            continue

        # --- Classify body vs heading ---
        is_body = _is_body_paragraph(render_lines, page_body_font, body_line_h, HEADING_SCALE, dpi)

        box_widths = _para_box_widths(render_lines, dpi)
        full_text = _join_translated_lines([lt for _, lt, _ in render_lines])

        if is_body:
            para_fs = max(1, unified_body_fs)
        else:
            para_fs = max(1, _consistent_font_size_for_paragraph(render_lines, dpi) or 10)
        min_fs = max(4, para_fs // 3)
        best_fs = _find_best_fs(full_text, box_widths, para_fs, min_fs=min_fs)

        final_lines = _wrap_text_to_boxes(full_text, box_widths, best_fs)

        for idx, (raw_bbox, _, font_sz) in enumerate(render_lines):
            line_text = final_lines[idx] if idx < len(final_lines) else ""
            if not line_text.strip():
                continue

            lx0, ly0, lx1, ly1 = raw_bbox
            x_pt = _pdf_point(lx0, dpi)
            x1_pt = _pdf_point(lx1, dpi)
            y0_pt = _pdf_point(ly0, dpi)
            y1_pt = _pdf_point(ly1, dpi)
            line_h = y1_pt - y0_pt
            box_w = x1_pt - x_pt
            if line_h < 2 or box_w < 2:
                continue

            fs = best_fs

            if font_path:
                try:
                    pil_font = ImageFont.truetype(font_path, int(fs))
                    ascent, descent = pil_font.getmetrics()
                    glyph_h = ascent + descent
                    if glyph_h > 0 and glyph_h < line_h * 2:
                        baseline_y = y0_pt + (line_h - glyph_h) / 2 + ascent
                    else:
                        baseline_y = y0_pt + fs
                except Exception:
                    baseline_y = y0_pt + fs
            else:
                baseline_y = y0_pt + fs

            page.insert_text(
                (x_pt, baseline_y),
                line_text,
                fontname=fontname,
                fontsize=fs,
            )


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
