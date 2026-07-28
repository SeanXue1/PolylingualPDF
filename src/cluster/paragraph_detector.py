from __future__ import annotations

from .geometry import median_height


def detect_paragraphs(
    line_groups: list[list[tuple[float, float, float, float]]],
    line_spacing_ratio: float = 1.5,
    min_line_height: float = 4.0,
) -> list[list[int]]:
    if not line_groups:
        return []
    if len(line_groups) == 1:
        return [[0]]

    # Calculate line bounding boxes and metrics
    line_bboxes = []
    line_heights = []
    line_lefts = []
    line_rights = []

    for line in line_groups:
        x0 = min(b[0] for b in line)
        y0 = min(b[1] for b in line)
        x1 = max(b[2] for b in line)
        y1 = max(b[3] for b in line)
        line_bboxes.append((x0, y0, x1, y1))
        line_heights.append(y1 - y0)
        line_lefts.append(x0)
        line_rights.append(x1)

    # Estimate block-level dimensions
    block_left = min(line_lefts)
    block_right = max(line_rights)
    block_width = block_right - block_left

    # Estimate median line height and inter-line gaps
    inter_line_gaps = []
    for i in range(1, len(line_groups)):
        prev_bottom = line_bboxes[i - 1][3]
        curr_top = line_bboxes[i][1]
        gap = curr_top - prev_bottom
        if gap >= 0:
            inter_line_gaps.append(gap)

    med_line_h = sum(line_heights) / len(line_heights) if line_heights else 12.0
    med_line_gap = (sum(inter_line_gaps) / len(inter_line_gaps)) if inter_line_gaps else (med_line_h * 0.25)

    # Threshold for paragraph break: gap significantly larger than normal inter-line gap
    max_paragraph_gap = max(med_line_h * 0.8, med_line_gap * 2.0, 8.0)

    paragraphs: list[list[int]] = [[0]]

    for i in range(1, len(line_groups)):
        prev_b = line_bboxes[i - 1]
        curr_b = line_bboxes[i]

        prev_bottom = prev_b[3]
        curr_top = curr_b[1]
        gap = curr_top - prev_bottom

        prev_h = line_heights[i - 1]
        curr_h = line_heights[i]

        # 1. Large vertical gap -> split
        if gap > max_paragraph_gap:
            paragraphs.append([i])
            continue

        # 2. Font height discrepancy (> 25% difference, e.g. heading vs body) -> split
        if prev_h > 0 and curr_h > 0:
            ratio = max(prev_h, curr_h) / min(prev_h, curr_h)
            if ratio > 1.40:
                paragraphs.append([i])
                continue

        # 3. Short ending line signal: prev_line ends early before right margin of block
        # (Only if block is multi-character wide > 3 * line_h)
        if block_width > med_line_h * 3.0:
            prev_right = line_rights[i - 1]
            if block_right - prev_right > med_line_h * 2.5 and gap > 0:
                paragraphs.append([i])
                continue

        # 4. First-line indent signal: curr_line is indented from left margin
        if block_width > med_line_h * 3.0:
            curr_left = line_lefts[i]
            prev_left = line_lefts[i - 1]
            if curr_left - block_left > med_line_h * 1.2 and abs(prev_left - block_left) < med_line_h * 0.5:
                paragraphs.append([i])
                continue

        # No split trigger -> append to current paragraph
        paragraphs[-1].append(i)

    return paragraphs
