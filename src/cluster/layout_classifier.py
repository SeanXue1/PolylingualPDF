from __future__ import annotations


def _page_diagonal(page_width: float, page_height: float) -> float:
    return (page_width ** 2 + page_height ** 2) ** 0.5


def classify_paragraph(
    bbox: tuple[float, float, float, float],
    page_height: float,
    page_width: float,
    font_size: float = 0.0,
    text_length: int = 0,
    num_lines: int = 1,
) -> str:
    x0, y0, x1, y1 = bbox
    h = y1 - y0

    if page_height > 0:
        if y0 < page_height * 0.08 and h < page_height * 0.06:
            return "header"

        if y1 > page_height * 0.92 and h < page_height * 0.06:
            return "footer"

    if font_size <= 0:
        font_size = h / max(1, num_lines)

    diag = _page_diagonal(page_width, page_height)
    relative_size = font_size / diag if diag > 0 else 0

    if relative_size > 0.030 and text_length < 120:
        return "title"

    if num_lines <= 1 and text_length < 80 and relative_size < 0.020:
        return "caption"

    return "body"


def classify_paragraphs(
    para_bboxes: list[tuple[float, float, float, float]],
    page_height: float,
    page_width: float,
    font_sizes: list[float] | None = None,
    text_lengths: list[int] | None = None,
    num_lines_list: list[int] | None = None,
) -> list[str]:
    types: list[str] = []
    for i, bbox in enumerate(para_bboxes):
        fs = font_sizes[i] if font_sizes and i < len(font_sizes) else bbox[3] - bbox[1]
        tl = text_lengths[i] if text_lengths and i < len(text_lengths) else 0
        nl = num_lines_list[i] if num_lines_list and i < len(num_lines_list) else 1
        types.append(classify_paragraph(bbox, page_height, page_width, font_size=fs, text_length=tl, num_lines=nl))
    return types
