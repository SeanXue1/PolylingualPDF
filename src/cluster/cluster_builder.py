from __future__ import annotations

from ..models import Paragraph, TextBlock
from .geometry import bbox_height, bbox_union, bbox_area, iou, intersection, horizontal_overlap, vertical_overlap
from .reading_order import sort_boxes_in_reading_order
from .xycut import xycut_segment
from .line_detector import detect_lines
from .paragraph_detector import detect_paragraphs
from .layout_classifier import classify_paragraphs


def _estimate_font_size(bbox: tuple[float, float, float, float]) -> float:
    return bbox_height(bbox)


def _containment_ratio(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Return the fraction of the smaller box's area that overlaps with the larger."""
    inter = intersection(a, b)
    if inter is None:
        return 0.0
    inter_area = bbox_area(inter)
    smaller_area = min(bbox_area(a), bbox_area(b))
    return inter_area / smaller_area if smaller_area > 0 else 0.0


def _merge_overlapping_paragraphs(
    all_para_data: list[dict],
    iou_threshold: float = 0.12,
    containment_threshold: float = 0.60,
) -> list[dict]:
    """Merge paragraph entries whose bounding boxes significantly overlap.

    Two paragraphs are merged when:
    - Their IoU exceeds *iou_threshold*, OR
    - One box is largely contained within the other (containment ratio > *containment_threshold*).

    Merging also catches line-split fragments that have the same column footprint
    and only a small vertical gap between them.
    """
    if len(all_para_data) <= 1:
        return all_para_data

    def _ordered_lines(lines: list[tuple[tuple[float, float, float, float], str, float]]):
        return sorted(lines, key=lambda t: (t[0][1], t[0][0]))

    def _median(values: list[float], default: float = 12.0) -> float:
        vals = sorted(v for v in values if v > 0)
        if not vals:
            return default
        return vals[len(vals) // 2]

    def _vertical_separation(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        if a[3] <= b[1]:
            return b[1] - a[3]
        if b[3] <= a[1]:
            return a[1] - b[3]
        return 0.0

    def _horizontal_separation(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        if a[2] <= b[0]:
            return b[0] - a[2]
        if b[2] <= a[0]:
            return a[0] - b[2]
        return 0.0

    def _merge_entry_group(group: list[dict]) -> dict:
        combined_boxes: list[TextBlock] = []
        combined_lines: list[tuple[tuple[float, float, float, float], str, float]] = []
        for entry in group:
            combined_boxes.extend(entry["boxes"])
            combined_lines.extend(entry["lines"])
        combined_boxes.sort(key=lambda tb: (tb.bbox[1], tb.bbox[0]))
        combined_lines = _ordered_lines(combined_lines)
        line_bboxes = [b for b, _, _ in combined_lines]
        if line_bboxes:
            merged_bbox = bbox_union(line_bboxes)
        else:
            merged_bbox = bbox_union([tb.bbox for tb in combined_boxes])
        return {
            "bbox": merged_bbox,
            "boxes": combined_boxes,
            "text": "".join(tb.text for tb in combined_boxes),
            "lines": combined_lines,
        }

    med_line_h = _median([bbox_height(b) for p in all_para_data for b, _, _ in p["lines"]], default=12.0)
    vertical_gap_limit = max(8.0, med_line_h * 0.95)
    horizontal_gap_limit = max(8.0, med_line_h * 0.75)

    def _should_merge(a: dict, b: dict) -> bool:
        a_bbox = a["bbox"]
        b_bbox = b["bbox"]
        if iou(a_bbox, b_bbox) > iou_threshold or _containment_ratio(a_bbox, b_bbox) > containment_threshold:
            return True

        a_w = max(1.0, a_bbox[2] - a_bbox[0])
        b_w = max(1.0, b_bbox[2] - b_bbox[0])
        a_h = max(1.0, a_bbox[3] - a_bbox[1])
        b_h = max(1.0, b_bbox[3] - b_bbox[1])

        x_overlap = horizontal_overlap(a_bbox, b_bbox)
        y_overlap = vertical_overlap(a_bbox, b_bbox)
        x_overlap_ratio = x_overlap / min(a_w, b_w)
        y_overlap_ratio = y_overlap / min(a_h, b_h)

        v_gap = _vertical_separation(a_bbox, b_bbox)
        h_gap = _horizontal_separation(a_bbox, b_bbox)

        # Same-column fragments: strong x overlap and only a small vertical gap.
        if x_overlap_ratio >= 0.55 and v_gap <= vertical_gap_limit:
            return True

        # Less common horizontal fragments: strong y overlap and only a small horizontal gap.
        if y_overlap_ratio >= 0.55 and h_gap <= horizontal_gap_limit:
            return True

        # Very thin cross-overlaps caused by XY-cut / OCR noise.
        if x_overlap_ratio >= 0.35 and y_overlap_ratio >= 0.35 and min(a_w, b_w, a_h, b_h) > 0:
            return True

        return False

    parent = list(range(len(all_para_data)))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(i: int, j: int) -> None:
        ri = _find(i)
        rj = _find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(all_para_data)):
        for j in range(i + 1, len(all_para_data)):
            if _should_merge(all_para_data[i], all_para_data[j]):
                _union(i, j)

    groups: dict[int, list[dict]] = {}
    for idx, entry in enumerate(all_para_data):
        root = _find(idx)
        groups.setdefault(root, []).append(entry)

    merged = [_merge_entry_group(group) for group in groups.values()]
    merged.sort(key=lambda p: (p["bbox"][1], p["bbox"][0]))
    return merged


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

    # 5.5. Merge paragraphs with overlapping bounding boxes
    all_para_data = _merge_overlapping_paragraphs(all_para_data)

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
