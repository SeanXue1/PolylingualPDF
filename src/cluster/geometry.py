from __future__ import annotations


def bbox_width(bbox: tuple[float, float, float, float]) -> float:
    return bbox[2] - bbox[0]


def bbox_height(bbox: tuple[float, float, float, float]) -> float:
    return bbox[3] - bbox[1]


def bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return bbox_width(bbox) * bbox_height(bbox)


def x_center(bbox: tuple[float, float, float, float]) -> float:
    return (bbox[0] + bbox[2]) / 2


def y_center(bbox: tuple[float, float, float, float]) -> float:
    return (bbox[1] + bbox[3]) / 2


def horizontal_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0]))


def vertical_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    return max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def horizontal_gap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    return b[0] - a[2]


def vertical_gap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    return b[1] - a[3]


def intersection(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x0 < x1 and y0 < y1:
        return (x0, y0, x1, y1)
    return None


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    inter = intersection(a, b)
    if inter is None:
        return 0.0
    inter_area = bbox_area(inter)
    union_area = bbox_area(a) + bbox_area(b) - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def is_same_line(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    y_overlap_ratio: float = 0.3,
) -> bool:
    a_h = bbox_height(a)
    b_h = bbox_height(b)
    if a_h == 0 or b_h == 0:
        return False
    overlap = vertical_overlap(a, b)
    min_h = min(a_h, b_h)
    return overlap / min_h >= y_overlap_ratio if min_h > 0 else False


def is_aligned_left(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    threshold: float = 5.0,
) -> bool:
    return abs(a[0] - b[0]) <= threshold


def is_aligned_right(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    threshold: float = 5.0,
) -> bool:
    return abs(a[2] - b[2]) <= threshold


def bbox_union(bboxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    if not bboxes:
        return (0.0, 0.0, 0.0, 0.0)
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    return (x0, y0, x1, y1)


def median_height(bboxes: list[tuple[float, float, float, float]], default: float = 12.0) -> float:
    heights = [bbox_height(b) for b in bboxes if bbox_height(b) > 0]
    if not heights:
        return default
    heights.sort()
    return heights[len(heights) // 2]

