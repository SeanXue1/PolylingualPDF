from __future__ import annotations

from .geometry import bbox_height, vertical_overlap, horizontal_gap, median_height


def detect_lines(
    bboxes: list[tuple[float, float, float, float]],
    y_overlap_ratio: float = 0.4,
    max_h_gap_mult: float = 1.8,
) -> list[list[int]]:
    if not bboxes:
        return []

    med_h = median_height(bboxes, default=12.0)
    max_h_gap = max(15.0, med_h * max_h_gap_mult)

    sorted_indices = sorted(range(len(bboxes)), key=lambda i: (bboxes[i][1], bboxes[i][0]))
    lines: list[list[int]] = []

    for idx in sorted_indices:
        curr = bboxes[idx]
        curr_h = bbox_height(curr)
        if curr_h <= 0:
            continue

        best_line_idx = -1
        best_dist = float("inf")

        for li, line in enumerate(lines):
            # Check against the last added box in the candidate line
            ref_idx = line[-1]
            ref = bboxes[ref_idx]
            ref_h = bbox_height(ref)
            if ref_h <= 0:
                continue

            overlap = vertical_overlap(ref, curr)
            min_h = min(ref_h, curr_h)

            if min_h > 0 and overlap / min_h >= y_overlap_ratio:
                # Check horizontal gap between ref and curr
                # (since sorted by x, curr is to the right of ref or slightly overlapping)
                h_gap = curr[0] - ref[2]
                if h_gap <= max_h_gap:
                    if h_gap < best_dist:
                        best_dist = h_gap
                        best_line_idx = li

        if best_line_idx >= 0:
            lines[best_line_idx].append(idx)
        else:
            lines.append([idx])

    # Sort boxes in each line by x0
    for line in lines:
        line.sort(key=lambda i: bboxes[i][0])

    # Sort lines by average y0
    lines.sort(key=lambda line: sum(bboxes[i][1] for i in line) / len(line))

    return lines
