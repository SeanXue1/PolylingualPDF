from __future__ import annotations

from .geometry import x_center, y_center, bbox_height


def sort_boxes_in_reading_order(
    bboxes: list[tuple[float, float, float, float]],
    column_gap: float = 30.0,
) -> list[int]:
    """Sort bounding boxes in natural reading order:

    Groups boxes by vertical bands / columns and sorts top-to-bottom, left-to-right.
    Returns list of original indices in sorted order.
    """
    if not bboxes:
        return []
    if len(bboxes) == 1:
        return [0]

    # Create indexed items
    items = [(i, b) for i, b in enumerate(bboxes)]

    # Sort primarily by y0, secondarily by x0
    items.sort(key=lambda item: (item[1][1], item[1][0]))

    # Group into lines/strips if vertical overlap is significant
    strips: list[list[tuple[int, tuple[float, float, float, float]]]] = []
    for idx, box in items:
        bh = bbox_height(box)
        placed = False
        for strip in strips:
            # Check overlap with strip average y
            strip_y0 = sum(b[1] for _, b in strip) / len(strip)
            strip_y1 = sum(b[3] for _, b in strip) / len(strip)
            strip_h = max(1.0, strip_y1 - strip_y0)

            # If box y_center falls within the vertical span of the strip
            if box[1] < strip_y1 and box[3] > strip_y0:
                overlap = min(box[3], strip_y1) - max(box[1], strip_y0)
                if overlap / min(max(1.0, bh), strip_h) >= 0.4:
                    strip.append((idx, box))
                    placed = True
                    break
        if not placed:
            strips.append([(idx, box)])

    result: list[int] = []
    for strip in strips:
        # Within each horizontal strip, sort left-to-right
        strip.sort(key=lambda item: item[1][0])
        result.extend(idx for idx, _ in strip)

    return result
