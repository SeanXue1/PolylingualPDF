from __future__ import annotations

from ..models import Paragraph, TextBlock
from .geometry import bbox_height, bbox_union
from .reading_order import sort_boxes_in_reading_order
from .xycut import xycut_segment
from .line_detector import detect_lines
from .paragraph_detector import detect_paragraphs
from .layout_classifier import classify_paragraphs


def _estimate_font_size(bbox: tuple[float, float, float, float]) -> float:
    return bbox_height(bbox)


def run_cluster_pipeline(
    text_boxes: list[TextBlock],
    page_width: float,
    page_height: float,
    column_gap: float = 30.0,
    line_spacing_ratio: float = 1.5,
    min_region_size: int = 0,
) -> list[Paragraph]:
    if not text_boxes:
        return []

    # 1. Raw spatial bounding boxes (do NOT pre-sort, preserve 2D topology)
    bboxes = [tb.bbox for tb in text_boxes]

    # 2. Recursive XY-Cut Layout Segmentation
    region_idx_groups = xycut_segment(
        bboxes, page_width, page_height, min_region_size=min_region_size
    )

    all_para_data: list[dict] = []

    # 3. Process each region (Line Detection -> Paragraph Detection)
    for region in region_idx_groups:
        if not region:
            continue
        region_bboxes = [bboxes[i] for i in region]

        # 4. Text Line Detection
        line_idx_groups = detect_lines(region_bboxes)

        line_bbox_groups: list[list[tuple[float, float, float, float]]] = []
        for line_group in line_idx_groups:
            group_bboxes = [region_bboxes[bi] for bi in line_group]
            line_bbox_groups.append(group_bboxes)

        # 5. Paragraph Detection
        para_line_idx_groups = detect_paragraphs(line_bbox_groups, line_spacing_ratio=line_spacing_ratio)

        for para_line_indices in para_line_idx_groups:
            para_lines: list[tuple[float, float, float, float]] = []
            para_boxes: list[TextBlock] = []

            for li in para_line_indices:
                line = line_idx_groups[li]
                for bi in line:
                    global_idx = region[bi]
                    para_lines.append(bboxes[global_idx])
                    para_boxes.append(text_boxes[global_idx])

            if not para_lines:
                continue

            para_bbox = bbox_union(para_lines)
            para_text = "".join(tb.text for tb in para_boxes)

            line_tuples: list[tuple[tuple[float, float, float, float], str, float]] = []
            for b, tb in zip(para_lines, para_boxes):
                fs = tb.font_size if tb.font_size > 0 else _estimate_font_size(b)
                line_tuples.append((b, tb.text, fs))

            all_para_data.append({
                "bbox": para_bbox,
                "boxes": para_boxes,
                "text": para_text,
                "lines": line_tuples,
            })

    if not all_para_data:
        return []

    # 6. Layout Classification
    para_bboxes = [p["bbox"] for p in all_para_data]
    font_sizes = []
    text_lengths = []
    num_lines_list = []
    for p in all_para_data:
        fs_values = [fs for _, _, fs in p["lines"] if fs > 0]
        font_sizes.append(sum(fs_values) / len(fs_values) if fs_values else 0)
        text_lengths.append(len(p["text"]))
        num_lines_list.append(len(p["lines"]))

    layout_types = classify_paragraphs(
        para_bboxes, page_height, page_width, font_sizes, text_lengths, num_lines_list
    )

    # 7. Reading Order Detection on Paragraph Bounding Boxes
    reading_order = sort_boxes_in_reading_order(para_bboxes, column_gap=column_gap)

    paragraphs: list[Paragraph] = []
    for sorted_i in reading_order:
        p_data = all_para_data[sorted_i]
        ltype = layout_types[sorted_i]

        para = Paragraph(
            blocks=p_data["boxes"],
            line_bboxes=p_data["lines"],
        )
        para.layout_type = ltype
        paragraphs.append(para)

    return paragraphs
