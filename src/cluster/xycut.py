from __future__ import annotations

from .geometry import median_height, bbox_width


def _project_x(
    bboxes: list[tuple[float, float, float, float]],
    page_width: float,
    exclude_wide_ratio: float = 0.0,
) -> list[int]:
    profile = [0] * max(1, int(page_width) + 1)
    max_wide = page_width * exclude_wide_ratio if exclude_wide_ratio > 0 else float("inf")
    for bx0, _, bx1, _ in bboxes:
        if bx1 - bx0 > max_wide:
            continue
        start = max(0, int(bx0))
        end = min(len(profile), int(bx1) + 1)
        for x in range(start, end):
            profile[x] += 1
    return profile


def _project_y(
    bboxes: list[tuple[float, float, float, float]],
    page_height: float,
) -> list[int]:
    profile = [0] * max(1, int(page_height) + 1)
    for _, by0, _, by1 in bboxes:
        start = max(0, int(by0))
        end = min(len(profile), int(by1) + 1)
        for y in range(start, end):
            profile[y] += 1
    return profile


def _find_gaps(profile: list[int], threshold: float = 0.0) -> list[tuple[int, int]]:
    gaps: list[tuple[int, int]] = []
    in_gap = False
    gap_start = 0
    for i, count in enumerate(profile):
        if count <= threshold:
            if not in_gap:
                gap_start = i
                in_gap = True
        else:
            if in_gap:
                gaps.append((gap_start, i))
                in_gap = False
    if in_gap:
        gaps.append((gap_start, len(profile)))
    return gaps


def _filter_internal_gaps(
    gaps: list[tuple[int, int]],
    min_size: int,
    min_coord: float,
    max_coord: float,
) -> list[tuple[int, int]]:
    """Filter gaps to only keep internal gaps that fall between boxes and have width >= min_size."""
    significant = []
    for s, e in gaps:
        if e - s < min_size:
            continue
        # Gap must lie inside the bounding box span of the content
        if e <= min_coord or s >= max_coord:
            continue
        significant.append((s, e))
    return significant


def _assign_to_region(
    idx: int, mid: float, gaps: list[tuple[int, int]], regions: list[list[int]],
) -> None:
    for gi, (gs, ge) in enumerate(gaps):
        if mid < gs:
            regions[gi].append(idx)
            return
        if gs <= mid <= ge:
            if mid - gs < ge - mid:
                regions[gi].append(idx)
            else:
                regions[gi + 1].append(idx)
            return
    regions[-1].append(idx)


def _split_by_vertical_gaps(
    bboxes: list[tuple[float, float, float, float]],
    gaps: list[tuple[int, int]],
) -> list[list[int]]:
    regions: list[list[int]] = [[] for _ in range(len(gaps) + 1)]
    for idx, (bx0, _, bx1, _) in enumerate(bboxes):
        x_mid = (bx0 + bx1) / 2
        _assign_to_region(idx, x_mid, gaps, regions)
    return [r for r in regions if r]


def _split_by_horizontal_gaps(
    bboxes: list[tuple[float, float, float, float]],
    gaps: list[tuple[int, int]],
) -> list[list[int]]:
    regions: list[list[int]] = [[] for _ in range(len(gaps) + 1)]
    for idx, (_, by0, _, by1) in enumerate(bboxes):
        y_mid = (by0 + by1) / 2
        _assign_to_region(idx, y_mid, gaps, regions)
    return [r for r in regions if r]


def xycut_segment(
    bboxes: list[tuple[float, float, float, float]],
    page_width: float,
    page_height: float,
    min_region_size: int = 0,
    gap_threshold: float = 0.0,
    wide_column_ratio: float = 0.45,
) -> list[list[int]]:
    if len(bboxes) <= 1:
        return [list(range(len(bboxes)))] if bboxes else []

    med_h = median_height(bboxes, default=12.0)
    # Adaptive gap sizes based on text size
    min_v_gap = max(4, int(med_h * 0.35))
    min_h_gap = max(5, int(med_h * 0.5))

    min_x = min(b[0] for b in bboxes)
    max_x = max(b[2] for b in bboxes)
    min_y = min(b[1] for b in bboxes)
    max_y = max(b[3] for b in bboxes)

    # 1. Try vertical cuts (split columns / side-by-side blocks)
    x_profile = _project_x(bboxes, page_width)
    gaps = _find_gaps(x_profile, gap_threshold)
    significant = _filter_internal_gaps(gaps, min_v_gap, min_x, max_x)

    # If no vertical gap found, try excluding wide banner boxes (headers/titles)
    if not significant and wide_column_ratio > 0:
        x_profile_ex = _project_x(bboxes, page_width, exclude_wide_ratio=wide_column_ratio)
        gaps_ex = _find_gaps(x_profile_ex, gap_threshold)
        significant_ex = _filter_internal_gaps(gaps_ex, min_v_gap, min_x, max_x)
        if significant_ex:
            significant = significant_ex

    if significant:
        regions = _split_by_vertical_gaps(bboxes, significant)
        if len(regions) > 1:
            result: list[list[int]] = []
            for r in regions:
                sub_bboxes = [bboxes[i] for i in r]
                sub_result = xycut_segment(sub_bboxes, page_width, page_height, min_region_size, gap_threshold, wide_column_ratio)
                for group in sub_result:
                    result.append([r[i] for i in group])
            return result

    # 2. Try horizontal cuts (split paragraphs / section breaks)
    y_profile = _project_y(bboxes, page_height)
    h_gaps = _find_gaps(y_profile, gap_threshold)
    significant_h = _filter_internal_gaps(h_gaps, min_h_gap, min_y, max_y)

    if significant_h:
        h_regions = _split_by_horizontal_gaps(bboxes, significant_h)
        if len(h_regions) > 1:
            result = []
            for r in h_regions:
                sub_bboxes = [bboxes[i] for i in r]
                sub_result = xycut_segment(sub_bboxes, page_width, page_height, min_region_size, gap_threshold, wide_column_ratio)
                for group in sub_result:
                    result.append([r[i] for i in group])
            return result

    return [list(range(len(bboxes)))]
