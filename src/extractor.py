from __future__ import annotations

from typing import Optional

import fitz  # PyMuPDF
from .models import PageResult, TextBlock


def open_pdf(path: str) -> fitz.Document:
    return fitz.open(path)


def get_page_image(page: fitz.Page, dpi: int = 300) -> bytes:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


def extract_native_blocks(page: fitz.Page) -> list[TextBlock]:
    blocks = []
    raw = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    for raw_block in raw.get("blocks", []):
        if raw_block.get("type") != 0:
            continue
        bbox = tuple(raw_block["bbox"])
        text_parts = []
        max_font_size = 0.0
        for line in raw_block.get("lines", []):
            for span in line.get("spans", []):
                text_parts.append(span.get("text", ""))
                max_font_size = max(max_font_size, span.get("size", 0))
        text = "".join(text_parts).strip()
        if text:
            blocks.append(
                TextBlock(
                    bbox=bbox,
                    text=text,
                    source="native",
                    page_num=0,
                    font_size=round(max_font_size, 1),
                )
            )
    return blocks


def _merge_horizontal_blocks(blocks: list[TextBlock], x_thresh: float = 5.0) -> list[TextBlock]:
    if not blocks:
        return []
    sorted_blocks = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
    merged = [sorted_blocks[0]]
    for b in sorted_blocks[1:]:
        prev = merged[-1]
        prev_x0, prev_y0, prev_x1, prev_y1 = prev.bbox
        x0, y0, x1, y1 = b.bbox
        same_line = abs(y0 - prev_y0) < 3.0 and abs(y1 - prev_y1) < 3.0
        x_gap = x0 - prev_x1
        if same_line and 0 < x_gap < x_thresh:
            merged[-1] = TextBlock(
                bbox=(prev_x0, min(prev_y0, y0), max(prev_x1, x1), max(prev_y1, y1)),
                text=prev.text + b.text,
                source=prev.source,
                page_num=prev.page_num,
            )
        else:
            merged.append(b)
    return merged


def needs_ocr(page: fitz.Page, native_blocks: list[TextBlock], char_threshold: int = 20) -> bool:
    if not native_blocks:
        return True
    total_chars = sum(len(b.text) for b in native_blocks)
    return total_chars < char_threshold


def extract_page(page: fitz.Page, page_num: int, dpi: int = 300) -> PageResult:
    rect = page.rect
    result = PageResult(
        page_num=page_num,
        width=rect.width,
        height=rect.height,
        source_dpi=dpi,
    )

    native_blocks = extract_native_blocks(page)
    native_blocks = _merge_horizontal_blocks(native_blocks)

    for b in native_blocks:
        b.page_num = page_num

    result.blocks = native_blocks
    result.needs_ocr = needs_ocr(page, native_blocks)

    if result.needs_ocr:
        result.image = get_page_image(page, dpi=dpi)

    # Extract image bounding boxes (in PDF point space) for image-region filtering.
    # Only keep images with meaningful size (> 50x50 points) to avoid tiny icons/decorations.
    try:
        image_infos = page.get_image_info(xrefs=True)
        for info in image_infos:
            bbox = info.get("bbox")
            if bbox:
                x0, y0, x1, y1 = bbox
                w = x1 - x0
                h = y1 - y0
                if w > 50 and h > 50:
                    result.image_bboxes.append((x0, y0, x1, y1))
    except Exception:
        pass  # Graceful degradation if image info extraction fails

    return result


def extract_pdf(
    path: str,
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
    dpi: int = 300,
) -> list[PageResult]:
    doc = open_pdf(path)
    pages = []
    start = (page_start - 1) if page_start else 0
    end = min(page_end, len(doc)) if page_end else len(doc)

    for i in range(start, end):
        page = doc[i]
        result = extract_page(page, i + 1, dpi=dpi)
        pages.append(result)

    doc.close()
    return pages
